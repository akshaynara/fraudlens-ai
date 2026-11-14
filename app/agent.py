"""
Investigation pipeline for one incoming complaint.

    evidence  ->  [1] Gemini reconstructs incident (multimodal, structured)
              ->  [2] Gemini agent investigates with tools (lookups, similarity, report reads)
              ->  [3] Deterministic graph of connections (never hallucinated)
              ->  [4] Gemini Skeptic tries to break every connection
              ->  [5] Deterministic liability decision under RBI's 2027 framework (rules, never Gemini)
              ->  [6] Gemini writes the investigation brief, grounded in [1]-[5]

Every step appends to the case trace so the UI can show the agent working live.
If Gemini is unavailable, each step degrades to a rule-based fallback so the demo never dies.
"""
from __future__ import annotations

import json
import logging
import re
import traceback
from datetime import datetime, timezone

from . import graph as graphmod
from . import liability as liab
from .gemini_client import GeminiClient
from .models import AgentFindings, IncidentExtraction, LiabilityEvidence, SkepticReview
from .store import CaseStore, normalize_identifier

log = logging.getLogger("fraudlens.agent")

EXTRACT_SYSTEM = """You are an evidence analyst for a bank's financial-crime investigation unit in India.
You receive raw evidence from a fraud complaint: screenshots (UPI receipts, WhatsApp/SMS/Telegram chats,
bank statements), documents, audio, and the victim's own statement.
Reconstruct the incident as structured data. Rules:
- Extract EVERY payment identifier verbatim: UPI IDs (x@bank), phone numbers, bank account numbers, URLs,
  Telegram handles, emails. Record which evidence item each came from.
- Do NOT list the victim's own bank account, card or phone, or transaction/reference IDs, as identifiers.
- Separate what the evidence directly shows from what the victim assumes or you infer.
- Flag inconsistencies, e.g. amount in the statement differs from the receipt, dates that don't line up.
- Be neutral and precise. Never speculate about who the perpetrator is."""

AGENT_SYSTEM = """You are FraudLens, an AI investigation analyst working alongside a human investigator at a bank.
A new fraud complaint has been reconstructed into structured facts. Your job is to investigate whether it is
connected to previous complaints in the case database, using ONLY the tools provided.

Method:
1. Look up every payment identifier (UPI, phone, URL, account, Telegram) with lookup_identifier.
2. For each hit, read the matching report with get_report and look up THEIR identifiers too, following the chain
   until it stops (max 3 hops). Operations often rotate numbers but reuse UPI IDs, or vice versa.
3. Run find_similar_reports once with a description of the tactic to find reports that use the same playbook
   without sharing identifiers.
4. Identifiers marked known_legitimate (e.g. the 1930 helpline) are NOT links. Say so.
5. Then stop and answer with JSON matching this schema exactly:
   {"hypotheses": [..], "related_report_ids": [..], "unresolved_questions": [..], "suggested_next_steps": [..]}
Be economical: do not look up the same identifier twice. Never claim a connection you did not confirm via a tool."""

SKEPTIC_SYSTEM = """You are the Skeptic in a fraud-investigation unit. Your only job is to try to break the
proposed connections between a new complaint and previous reports before a human sees them.
For each related report, decide:
- strong: shared non-generic identifier(s) directly, or 2+ independent links
- plausible: 1-2 hop chain through non-generic identifiers, or strong tactic match plus timing fit
- weak: only narrative similarity, or a chain of 3+ hops
- reject: linked only through a legitimate/generic identifier (helplines, bank toll-free numbers, common merchants),
  or the link is contradicted by the evidence
Give one concrete reason per verdict in plain language a bank investigator would accept.
Never conclude that any person is guilty; you assess evidentiary links only."""

LIABILITY_SYSTEM = """You are an evidence analyst preparing a fraud complaint for a liability decision under
RBI's customer-protection framework for fraudulent electronic banking transactions. You do NOT decide liability.
You only report what the complaint's evidence shows about specific facts, and how strongly.
Rating rules, applied strictly because the burden of proof is on the bank:
- proven: the fact is directly visible in a screenshot, receipt or document you were given (quote or describe it).
- claimed: the fact is only stated, implied or inferred, including by the victim, with nothing that shows it.
- unknown: the evidence does not address it. Do not list unknown grounds.
A victim paying a fraudster is NOT by itself 'credentials_shared'; that ground needs a PIN, password or OTP being
disclosed. Give date-times only when the evidence states them; never guess a time."""

BRIEF_SYSTEM = """You write investigation briefs for a bank's financial-crime team in India. Write in clear,
plain English with short sections. Use markdown headings (##). Sections, in order:
Incident summary; Timeline; Evidence assessment (supported facts vs assumptions, inconsistencies);
Related complaints (list each with its link type and the Skeptic's verdict, strongest first; say explicitly which
proposed links were rejected and why); Liability assessment (only if a liability decision is provided: state the rule
matched, the findings relied on with their evidence, the amounts, the response deadline, and what is still unproven;
copy the amounts and the rule exactly, never recompute or change them); Pattern assessment (what the cluster suggests about the operation, hedged
appropriately); Unresolved questions; Recommended next steps (concrete: what to pull, whom to contact, what to
freeze-request through proper authorisation); Draft escalation note (3-4 sentences, ready for an authorised officer
to review and sign off). Ground every statement in the provided material. Never state or imply that a named
individual is guilty. Do not invent facts."""

PHONE_RE = re.compile(r"(?:\+91[\s-]?)?[6-9]\d{9}\b|\b1930\b|\b1800[-\s]?\d{3,4}[-\s]?\d{0,4}\b")
UPI_RE = re.compile(r"\b[a-zA-Z0-9._-]{2,}@[a-zA-Z]{2,}\b")
URL_RE = re.compile(r"https?://[^\s)]+|\b[a-z0-9-]+\.(?:xyz|top|online|site|link|gov\.in)\b")
TG_RE = re.compile(r"@[A-Za-z_][A-Za-z0-9_]{4,}")


class Investigator:
    def __init__(self, store: CaseStore, gemini: GeminiClient):
        self.store = store
        self.gemini = gemini

    # ------------------------------------------------------------------ entry
    def run(self, case_id: str):
        case = self.store.cases[case_id]
        trace = lambda kind, title, detail=None, payload=None: self.store.append_trace(
            case_id, {"kind": kind, "title": title, "detail": detail, "payload": payload,
                      "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        try:
            self.store.update_case(case_id, status="extracting")
            trace("phase", "Reconstructing the incident",
                  f"Gemini is reading {len(case['evidence_files'])} evidence file(s) and the victim statement")
            extraction = self._extract(case)
            self.store.update_case(case_id, extraction=extraction.model_dump())
            trace("note", f"Found {len(extraction.identifiers)} identifier(s), {len(extraction.inconsistencies)} inconsistency(ies)",
                  ", ".join(f"{i.type}: {i.value}" for i in extraction.identifiers) or "no identifiers extracted")

            self.store.update_case(case_id, status="investigating")
            trace("phase", "Investigating connections", "The agent decides which identifiers to follow and how far")
            findings = self._investigate(case_id, extraction, trace)
            self.store.update_case(case_id, findings=findings.model_dump())

            self.store.update_case(case_id, status="mapping")
            trace("phase", "Mapping the network", "Deterministic graph: shared identifiers and tactic similarity")
            cluster = graphmod.build_cluster(self.store, case_id, [i.model_dump() for i in extraction.identifiers],
                                             f"{extraction.scam_type}. {case['narrative']}")
            self.store.update_case(case_id, cluster=cluster)
            s = cluster["summary"]
            trace("note", f"{s['reports_in_cluster']} related report(s) through {s['identifiers_in_cluster']} identifier(s), up to {s['max_hops']} hop(s)",
                  f"Generic/legitimate identifiers ignored as links: {', '.join(cluster['generic_identifiers']) or 'none'}")

            self.store.update_case(case_id, status="reviewing")
            trace("phase", "Skeptic review", "A second Gemini pass tries to break every proposed connection")
            review = self._skeptic(case, extraction, cluster)
            self.store.update_case(case_id, review=review.model_dump())
            counts = {}
            for v in review.verdicts:
                counts[v.verdict] = counts.get(v.verdict, 0) + 1
            trace("note", "Skeptic verdicts: " + ", ".join(f"{k} {v}" for k, v in counts.items()), review.overall_assessment)

            self.store.update_case(case_id, status="deciding")
            trace("phase", "Deciding liability", "Deterministic rules on rated findings (RBI 2027 framework). Gemini never chooses the verdict")
            liability = self._liability(case, extraction, review)
            self.store.update_case(case_id, liability=liability)
            if liability["status"] == "decided":
                d = liability["decision"]
                trace("note", f"{d['rule']}: {d['headline']}",
                      f"Reverse Rs {d['reversal_inr']:,.0f} | compensation Rs {d['compensation_inr']:,.0f} | customer bears Rs {d['customer_bears_inr']:,.0f}")
            else:
                trace("note", "Liability not decided", "Missing: " + ", ".join(liability["missing"]))

            self.store.update_case(case_id, status="writing")
            trace("phase", "Writing the investigation brief")
            brief = self._brief(case, extraction, findings, cluster, review, liability)
            self.store.update_case(case_id, brief=brief)

            added = self.store.commit_case_to_corpus(case_id, extraction.model_dump())
            self.store.update_case(case_id, status="done", committed=added,
                                   finished_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
            trace("phase", "Done", "Case added to the corpus so future complaints can link to it" if added
                  else "Demo or duplicate statement, so it was not added to the corpus (re-runs stay clean)")
            if self.store.archive_case(case_id):
                trace("note", "Case saved to Firestore", "Investigation record persists across restarts")
        except Exception as e:  # keep the error visible in the UI
            log.error("investigation failed: %s\n%s", e, traceback.format_exc())
            trace("error", "Investigation failed", str(e))
            self.store.update_case(case_id, status="error", error=str(e))
            self.store.archive_case(case_id)

    # ------------------------------------------------------------------ [1] extraction
    def _extract(self, case: dict) -> IncidentExtraction:
        if self.gemini.available:
            parts = []
            for f in case["evidence_blobs"]:
                parts.append(self.gemini.file_part(f["bytes"], f["mime"]))
                parts.append(f"(The item above is evidence file: {f['name']})")
            parts.append(f"Today's date (UTC): {_today()}. The complaint was received at {case['created_at']}. "
                         "Dates up to today are not in the future.\n\n"
                         f"Victim statement:\n{case['narrative']}\n\nReconstruct the incident.")
            return self.gemini.generate_structured(parts, IncidentExtraction, EXTRACT_SYSTEM, model=self.gemini.model_pro)
        return self._extract_fallback(case)

    def _extract_fallback(self, case: dict) -> IncidentExtraction:
        text = case["narrative"]
        ids = []
        for m in PHONE_RE.findall(text):
            ids.append({"type": "phone", "value": m.strip(), "source": "victim statement"})
        for m in UPI_RE.findall(text):
            ids.append({"type": "upi", "value": m, "source": "victim statement"})
        for m in URL_RE.findall(text):
            ids.append({"type": "url", "value": m, "source": "victim statement"})
        for m in TG_RE.findall(text):
            if "@" in m and not UPI_RE.fullmatch(m):
                ids.append({"type": "telegram", "value": m, "source": "victim statement"})
        amounts = [{"value_inr": float(a.replace(",", "")), "source": "victim statement"}
                   for a in re.findall(r"Rs\.?\s?([\d,]{4,})", text)]
        return IncidentExtraction(
            summary=text[:300], scam_type=_guess_type(text), tactics=["(offline mode: tactics not analysed)"],
            timeline=[], identifiers=ids, amounts=amounts,
            facts_supported_by_evidence=["Identifiers and amounts were pattern-matched from the statement only."],
            assumptions_or_unverified=["Gemini unavailable: screenshots were not read."],
            missing_information=["Set GEMINI_API_KEY to enable multimodal evidence reading."], inconsistencies=[])

    # ------------------------------------------------------------------ [2] agent
    def _investigate(self, case_id: str, extraction: IncidentExtraction, trace) -> AgentFindings:
        store = self.store
        looked_up: set[str] = set()

        def lookup_identifier(value: str) -> dict:
            """Look up a payment identifier (UPI ID, phone number, URL, bank account, Telegram handle) in the
            complaint database. Returns matching report IDs, how many, their scam types, and whether the
            identifier is a known legitimate number such as a helpline (which must not be treated as a link)."""
            key = normalize_identifier(value)
            res = store.lookup_identifier(value, exclude=case_id)
            dup = key in looked_up
            looked_up.add(key)
            trace("tool", f"lookup_identifier({value})",
                  (f"{res['match_count']} match(es)" + (f" - known legitimate: {res['known_legitimate']}" if res['known_legitimate'] else ""))
                  + (" (repeat lookup)" if dup else ""), {"matches": res["matching_report_ids"]})
            return res

        def get_report(report_id: str) -> dict:
            """Read a previous complaint by its report ID. Returns its narrative, scam type, date, amount and
            the identifiers it contains, so you can follow the chain to further reports."""
            r = store.get_report(report_id)
            if not r:
                trace("tool", f"get_report({report_id})", "not found")
                return {"error": "not found"}
            trace("tool", f"get_report({report_id})", f"{r['scam_type']}, filed {r['filed_at'][:10]}, {len(r['identifiers'])} identifier(s)")
            return {k: r[k] for k in ("id", "filed_at", "scam_type", "channel", "amount_claimed_inr",
                                       "amount_on_receipt_inr", "narrative", "identifiers", "status")}

        def find_similar_reports(tactic_description: str) -> list[dict]:
            """Find previous complaints whose narrative describes a similar scam tactic, even if they share no
            identifiers with this case. Pass a 1-2 sentence description of the modus operandi."""
            res = store.find_similar(tactic_description, k=6, exclude=case_id, min_score=graphmod.semantic_min(store))
            trace("tool", "find_similar_reports(...)", f"{len(res)} report(s) above similarity {graphmod.semantic_min(store)}",
                  {"matches": [r["report_id"] for r in res]})
            return res

        if not self.gemini.available:
            return self._investigate_fallback(case_id, extraction, lookup_identifier, get_report, find_similar_reports)

        prompt = (f"Today's date (UTC): {_today()}.\n\nReconstructed complaint:\n" + extraction.model_dump_json(indent=1) +
                  "\n\nInvestigate connections now. Finish with the JSON object only.")
        text = self.gemini.run_agent(prompt, AGENT_SYSTEM, [lookup_identifier, get_report, find_similar_reports])
        try:
            return AgentFindings.model_validate_json(_json_block(text))
        except Exception:
            trace("note", "Agent reply was not valid JSON; keeping free-text findings", text[:400])
            return AgentFindings(hypotheses=[text[:600]], related_report_ids=[], unresolved_questions=[], suggested_next_steps=[])

    def _investigate_fallback(self, case_id, extraction, lookup, get_report, similar) -> AgentFindings:
        related, frontier, seen = set(), [i.value for i in extraction.identifiers], set()
        for _ in range(3):
            nxt = []
            for v in frontier:
                if normalize_identifier(v) in seen:
                    continue
                seen.add(normalize_identifier(v))
                res = lookup(v)
                if res["known_legitimate"]:
                    continue
                for rid in res["matching_report_ids"]:
                    if rid not in related:
                        related.add(rid)
                        nxt += [i["value"] for i in get_report(rid)["identifiers"]]
            frontier = nxt
        for s in similar(f"{extraction.scam_type} {extraction.summary}"):
            related.add(s["report_id"])
        return AgentFindings(
            hypotheses=["Offline mode: connections found by exhaustive identifier chaining and narrative similarity."],
            related_report_ids=sorted(related),
            unresolved_questions=["Gemini unavailable; investigative reasoning not generated."],
            suggested_next_steps=["Enable Gemini to get a reasoned brief."])

    # ------------------------------------------------------------------ [4] skeptic
    def _skeptic(self, case: dict, extraction: IncidentExtraction, cluster: dict) -> SkepticReview:
        related = cluster["related"]
        if not related:
            return SkepticReview(verdicts=[], overall_assessment="No related reports were found, so there is nothing to challenge.",
                                 what_would_change_my_mind=["A match on any payment identifier in a future complaint."])
        if self.gemini.available:
            material = {
                "new_case": {"summary": extraction.summary, "scam_type": extraction.scam_type, "tactics": extraction.tactics,
                             "identifiers": [i.model_dump() for i in extraction.identifiers], "statement": case["narrative"]},
                "proposed_connections": [{k: r[k] for k in ("report_id", "scam_type", "filed_at", "hops", "via", "only_generic_link", "narrative")} for r in related],
                "generic_or_legitimate_identifiers": cluster["generic_identifiers"],
            }
            return self.gemini.generate_structured([json.dumps(material, indent=1), "Review every proposed connection."],
                                                   SkepticReview, SKEPTIC_SYSTEM, model=self.gemini.model_pro)
        verdicts = []
        for r in related:
            if r["only_generic_link"]:
                verdicts.append({"report_id": r["report_id"], "verdict": "reject", "reason": "Linked only through a legitimate/generic identifier."})
            elif r["hops"] == 1:
                verdicts.append({"report_id": r["report_id"], "verdict": "strong", "reason": "Shares a non-generic payment identifier directly."})
            elif r["hops"] == 0:
                verdicts.append({"report_id": r["report_id"], "verdict": "weak", "reason": "Narrative similarity only."})
            else:
                verdicts.append({"report_id": r["report_id"], "verdict": "plausible", "reason": f"{r['hops']}-hop chain through non-generic identifiers."})
        return SkepticReview(verdicts=verdicts, overall_assessment="Rule-based review (Gemini unavailable).", what_would_change_my_mind=[])

    # ------------------------------------------------------------------ [5] liability
    def _liability(self, case: dict, extraction: IncidentExtraction, review: SkepticReview) -> dict:
        """Gather facts (bank record + evidence), then let the deterministic rules decide."""
        record = liab.BankRecord.model_validate(case.get("bank_record") or {})
        evidence = self._liability_evidence(case, extraction)

        txn_at = record.transaction_at or _parse_dt(evidence.transaction_at)
        receipt = [a for a in extraction.amounts if "receipt" in a.source.lower()]
        amount = record.amount_inr or next((a.value_inr for a in receipt + list(extraction.amounts)), None)
        missing = [name for name, v in (("transaction date and time", txn_at), ("transaction amount", amount)) if not v]
        if missing:
            return {"status": "needs_data", "missing": missing, "has_bank_record": bool(case.get("bank_record")),
                    "evidence": evidence.model_dump()}

        bank_f, cust_f, tp_rating, tp_evidence = liab.findings_from_bank_record(record)
        have = {f.ground for f in cust_f}
        cust_f += [liab.CustomerFinding(ground=g.ground, rating=g.rating, evidence=g.evidence)
                   for g in evidence.customer_grounds if g.rating != "unknown" and g.ground not in have]
        strong = sum(1 for v in review.verdicts if v.verdict == "strong")
        lcase = liab.LiabilityCase(
            case_id=case["id"], transaction_at=txn_at, amount_inr=amount, cross_border=record.cross_border,
            is_credit_card=record.is_credit_card, customer_type=record.customer_type,
            # with no bank record, the complaint reaching FraudLens is the report to the bank
            reported_to_bank_at=record.reported_to_bank_at or _parse_dt(case["created_at"]),
            reported_to_ncrp_at=record.reported_to_ncrp_at or _parse_dt(evidence.reported_to_ncrp_at),
            ncrp_acknowledgement=record.ncrp_acknowledgement or evidence.ncrp_acknowledgement,
            bank_findings=bank_f, customer_findings=cust_f, third_party_breach=tp_rating, third_party_evidence=tp_evidence,
            amount_recovered_inr=record.amount_recovered_inr, post_report_amount_inr=record.post_report_amount_inr,
            prior_compensation_claimed=record.prior_compensation_claimed, linked_complaints=strong)
        decision = liab.decide(lcase, preview=True)
        return {"status": "decided", "has_bank_record": bool(case.get("bank_record")),
                "case": lcase.model_dump(mode="json"), "decision": decision.model_dump(mode="json"),
                "coercion": {"rating": evidence.coercion, "evidence": evidence.coercion_evidence},
                "notes": evidence.notes, "letter": liab.customer_letter(decision)}

    def _liability_evidence(self, case: dict, extraction: IncidentExtraction) -> LiabilityEvidence:
        if self.gemini.available:
            try:
                parts = []
                for f in case["evidence_blobs"]:
                    parts.append(self.gemini.file_part(f["bytes"], f["mime"]))
                    parts.append(f"(The item above is evidence file: {f['name']})")
                parts.append(f"Today's date (UTC): {_today()}. The complaint was received at {case['created_at']}.\n\n"
                             f"Victim statement:\n{case['narrative']}\n\nReconstruction so far:\n{extraction.model_dump_json(indent=1)}"
                             "\n\nReport the facts for the liability decision.")
                return self.gemini.generate_structured(parts, LiabilityEvidence, LIABILITY_SYSTEM, model=self.gemini.model_pro)
            except Exception as e:  # the decision can still run on the bank record alone
                log.warning("liability evidence pass failed: %s", e)
        t = case["narrative"].lower()
        coerced = any(k in t for k in ("digital arrest", "could not hang up", "threat", "arrest"))
        return LiabilityEvidence(
            customer_grounds=[], coercion="claimed" if coerced else "unknown",
            coercion_evidence="Victim statement describes threats or confinement on a call" if coerced else "",
            notes=["Offline mode: screenshots were not read for the liability facts; only the bank record and statement were used."])

    # ------------------------------------------------------------------ [6] brief
    def _brief(self, case, extraction, findings, cluster, review, liability=None) -> str:
        material = {
            "today_utc": _today(), "case_id": case["id"], "received": case["created_at"], "victim_statement": case["narrative"],
            "reconstruction": extraction.model_dump(), "agent_findings": findings.model_dump(),
            "network": {"summary": cluster["summary"], "related": [{k: r[k] for k in ("report_id", "scam_type", "filed_at", "city", "amount_inr", "hops", "via")} for r in cluster["related"]],
                        "generic_identifiers_ignored": cluster["generic_identifiers"]},
            "skeptic_review": review.model_dump(),
        }
        if liability and liability.get("status") == "decided":
            material["liability_decision"] = {"decision": liability["decision"], "coercion": liability["coercion"], "notes": liability["notes"]}
        if self.gemini.available:
            return self.gemini.generate_text(json.dumps(material, indent=1, default=str) + "\n\nWrite the investigation brief.",
                                             BRIEF_SYSTEM, model=self.gemini.model_pro)
        lines = [f"## Incident summary\n{extraction.summary}", "## Related complaints"]
        vmap = {v.report_id: v for v in review.verdicts}
        for r in cluster["related"]:
            v = vmap.get(r["report_id"])
            via = " -> ".join(x["value"] for x in r["via"])
            lines.append(f"- {r['report_id']} ({r['scam_type']}, {r['filed_at'][:10]}): via {via}. Verdict: {v.verdict if v else 'n/a'}")
        if liability and liability.get("status") == "decided":
            d = liability["decision"]
            lines.append(f"## Liability assessment\n{d['rule']}: {d['headline']}\n\n" + "\n".join(f"- {r}" for r in d["reasons"]))
        lines.append("## Note\nGemini unavailable: this is a rule-based brief. Set GEMINI_API_KEY for the full analysis.")
        return "\n\n".join(lines)


def _parse_dt(value) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%d %B %Y")


def _guess_type(text: str) -> str:
    t = text.lower()
    for key, kind in [("digital arrest", "digital_arrest"), ("parcel", "courier_parcel"), ("electricity", "electricity_bill"),
                      ("task", "task_job"), ("loan app", "loan_app"), ("kyc", "kyc_update"), ("olx", "fake_buyer_olx"),
                      ("stock", "investment_group")]:
        if key in t:
            return kind
    return "other"


def _json_block(text: str) -> str:
    t = text.strip()
    if "```" in t:
        t = t.split("```")[1]
        t = t[4:] if t.startswith("json") else t
    start, end = t.find("{"), t.rfind("}")
    return t[start:end + 1] if start >= 0 else t
