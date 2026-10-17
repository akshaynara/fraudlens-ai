"""
Deterministic relationship analysis. Gemini never invents a connection: every edge here
is either a shared identifier (exact, after normalisation) or a narrative-similarity score
above a threshold. Gemini's job is to decide what to look up and to explain/critique.
"""
from __future__ import annotations

import networkx as nx

from .store import CaseStore, normalize_identifier

import os

# Gemini embeddings are dense (unrelated same-domain texts still score ~0.7), local hashed vectors are sparse.
_THRESHOLDS = {"gemini": float(os.getenv("SEMANTIC_MIN_GEMINI", "0.82")),
               "local": float(os.getenv("SEMANTIC_MIN_LOCAL", "0.40"))}
MAX_DEPTH = 4


def semantic_min(store: "CaseStore") -> float:
    return _THRESHOLDS.get(store.embedder.mode, 0.82)


def build_cluster(store: CaseStore, case_id: str, identifiers: list[dict], narrative: str) -> dict:
    G = nx.Graph()
    G.add_node(case_id, kind="case")
    frontier = [case_id]
    seen_reports = {case_id}
    ident_of = {case_id: [normalize_identifier(i["value"]) for i in identifiers]}
    # The new case's identifiers are not in the corpus index yet; keep their type and display text
    case_ids = {normalize_identifier(i["value"]): (i.get("type", "other"), i["value"]) for i in identifiers}
    generic = set()

    # ---- expand through shared identifiers (BFS, bounded) ---------------------------
    for depth in range(MAX_DEPTH):
        nxt = []
        for rid in frontier:
            keys = ident_of.get(rid) or [normalize_identifier(i["value"]) for i in store.reports[rid]["identifiers"]]
            for key in keys:
                if not key:
                    continue
                hits = store.ident_index.get(key, set())
                legit = store.legit.get(key)
                # Identifiers that appear in many reports across unrelated tactics are probably generic.
                types_ = {store.reports[h]["scam_type"] for h in hits if h in store.reports}
                is_generic = legit is not None or (len(hits) >= 6 and len(types_) >= 3)
                if is_generic:
                    generic.add(key)
                G.add_node(f"id:{key}", kind="identifier", itype=store.ident_type.get(key) or case_ids.get(key, ("other",))[0],
                           label=store.ident_display.get(key) or case_ids.get(key, (None, key))[1], generic=is_generic, legit=legit,
                           report_count=len(hits))
                G.add_edge(rid, f"id:{key}", kind="shares")
                if is_generic:
                    continue  # do not expand through generic identifiers
                for h in hits:
                    if h == rid:
                        continue
                    if h not in G:
                        G.add_node(h, kind="report")
                    G.add_edge(h, f"id:{key}", kind="shares")
                    if h not in seen_reports:
                        seen_reports.add(h)
                        nxt.append(h)
        frontier = nxt
        if not frontier:
            break

    # ---- semantic edges (narrative similarity) to the new case ---------------------
    for s in store.find_similar(narrative, k=8, exclude=case_id, min_score=semantic_min(store)):
        rid = s["report_id"]
        if rid not in G:
            G.add_node(rid, kind="report")
        G.add_edge(case_id, rid, kind="similar", score=s["similarity"])

    # ---- explanations per related report ---------------------------------------------
    # Prefer the identifier chain: a similarity edge is often a shorter path, but it would hide
    # the hard evidence (shared UPI / phone) that actually connects the report.
    # Generic/legitimate identifiers (helplines) are only used as a last resort, so that a
    # report reachable through real identifiers is never explained "via 1930".
    gen_nodes = {f"id:{g}" for g in generic}
    G_strict = G.subgraph([n for n in G if n not in gen_nodes])
    G_ids = G_strict.edge_subgraph([(a, b) for a, b, d in G_strict.edges(data=True) if d.get("kind") == "shares"])
    related = []
    for rid in G.nodes:
        if rid == case_id or G.nodes[rid].get("kind") != "report":
            continue
        path = None
        for graph_ in (G_ids, G_strict, G):
            if case_id in graph_ and rid in graph_:
                try:
                    path = nx.shortest_path(graph_, case_id, rid)
                    break
                except nx.NetworkXNoPath:
                    pass
        if path is None:
            continue
        via = []
        for a, b in zip(path, path[1:]):
            e = G.edges[a, b]
            if e.get("kind") == "similar":
                via.append({"type": "similar_tactic", "value": f"narrative similarity {e['score']:.2f}"})
            elif b.startswith("id:"):
                n = G.nodes[b]
                via.append({"type": n["itype"], "value": n["label"], "generic": n["generic"], "legit": n["legit"]})
        rep = store.reports[rid]
        # identifier hops; 0 means the only link is narrative similarity (never "strong")
        hops = sum(1 for n in path if n.startswith("id:"))
        related.append({
            "report_id": rid, "scam_type": rep["scam_type"], "filed_at": rep["filed_at"],
            "city": rep["victim_city"], "amount_inr": rep["amount_claimed_inr"],
            "hops": hops, "via": via,
            "only_generic_link": all(v.get("generic") for v in via if v["type"] != "similar_tactic") and
                                 not any(v["type"] == "similar_tactic" for v in via),
            "narrative": rep["narrative"],
        })
    related.sort(key=lambda r: (r["hops"] == 0, r["hops"], r["filed_at"]))

    return {
        "graph": to_vis(G, case_id),
        "related": related,
        "generic_identifiers": sorted(generic),
        "summary": {"reports_in_cluster": len([n for n in G if G.nodes[n].get("kind") == "report"]),
                    "identifiers_in_cluster": len([n for n in G if n.startswith("id:")]),
                    "max_hops": max([r["hops"] for r in related], default=0)},
    }


def to_vis(G: nx.Graph, case_id: str) -> dict:
    nodes, edges = [], []
    for n, d in G.nodes(data=True):
        if d.get("kind") == "case":
            nodes.append({"id": n, "label": "New case", "group": "case"})
        elif d.get("kind") == "report":
            nodes.append({"id": n, "label": n.replace("FR-2026-", "FR-"), "group": "report"})
        else:
            nodes.append({"id": n, "label": d["label"], "group": "generic" if d["generic"] else f"id_{d['itype']}",
                          "title": (d["legit"] and f"Known legitimate: {d['legit']}") or f"{d['itype']} seen in {d['report_count']} reports"})
    for a, b, d in G.edges(data=True):
        if d.get("kind") == "similar":
            edges.append({"from": a, "to": b, "dashes": True, "label": f"{d['score']:.2f}", "kind": "similar"})
        else:
            edges.append({"from": a, "to": b, "kind": "shares"})
    return {"nodes": nodes, "edges": edges}


def corpus_overview(store: CaseStore) -> dict:
    """The 'before' picture: every report as an isolated dot, so the cluster reveal lands."""
    G = nx.Graph()
    for r in store.reports.values():
        G.add_node(r["id"], kind="report")
        for i in r["identifiers"]:
            key = normalize_identifier(i["value"])
            if key in store.legit or len(store.ident_index[key]) < 2:
                continue
            G.add_node(f"id:{key}", kind="identifier", itype=i["type"], label=store.ident_display[key],
                       generic=False, legit=None, report_count=len(store.ident_index[key]))
            G.add_edge(r["id"], f"id:{key}", kind="shares")
    return to_vis(G, "")
