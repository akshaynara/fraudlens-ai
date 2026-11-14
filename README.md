# FraudLens AI

**One complaint is a case. Connected complaints reveal a network.**

An AI investigation agent for financial-crime teams. A new fraud complaint arrives as messy evidence —
UPI receipts, WhatsApp/SMS screenshots, a victim's statement. FraudLens reconstructs the incident with
Gemini, then an agent follows every payment identifier through earlier complaints, maps the hidden
network, lets a second Gemini "Skeptic" try to break each connection, and writes an evidence-grounded
investigation brief. Investigators get leads with a visible trail, not a black-box score.

AI Builder Cup 2026 · Theme: **BFSI — Intelligent Risk, Fraud & Financial Experiences**

## How it works

```
evidence (images / PDF / audio / text)
   │
   ▼
[1] Reconstruct ─ Gemini multimodal, structured JSON (identifiers, timeline, amounts, inconsistencies,
   │               facts vs assumptions)
   ▼
[2] Investigate ─ Gemini agent with tools: lookup_identifier · get_report · find_similar_reports
   │               It decides what to follow, how far, and when to stop (automatic function calling)
   ▼
[3] Map ────────── Deterministic graph (NetworkX): shared identifiers + narrative-similarity edges.
   │               Gemini never invents a connection. Generic/legitimate identifiers never expand.
   ▼
[4] Skeptic ────── Second Gemini pass: strong / plausible / weak / reject for every link, with a reason
   ▼
[5] Brief ──────── Gemini writes the investigation brief + draft escalation note, grounded in [1]–[4]
```

Every step streams into an **agent activity** timeline in the UI. The case is then committed to the
corpus so the next complaint can link to it, and the full investigation record is saved to **Firestore**.

### Google Cloud services

| Service | What it does in FraudLens |
|---|---|
| **Gemini 3.8 Flash** | Multimodal evidence extraction (structured output), the tool-using investigation agent (automatic function calling), the Skeptic review and the brief |
| **Gemini embeddings** (`gemini-embedding-001`) | Narrative similarity between complaints, to surface reports that share a playbook but no identifier |
| **Cloud Run** | Hosts the FastAPI backend and the investigator workbench (one warm instance, CPU always on for background investigations) |
| **Firestore** | Case archive: every investigation (reconstruction, trace, network, verdicts, brief) is persisted; on restart, case history and investigated complaints are restored into the corpus so future complaints keep linking to them |
| **Firebase Hosting** | Public front door: serves the app on a `web.app` domain and rewrites all traffic to the Cloud Run service |

### Why this needs Gen AI
* Evidence is unstructured and multimodal; identifiers live inside screenshots and chats.
* Which identifier to follow next is a judgment call; the agent makes it with tools, not memory.
* Scam rings rotate numbers but reuse UPI IDs (or vice versa). Only chaining across reports reveals them.
* Investigators must be able to defend a connection. The Skeptic produces that defence — or kills the link.

## Liability engine (RBI 2027 framework)

From 1 January 2027, RBI's revised customer-protection directions put the burden of proving customer
liability on the bank and set deadlines for every fraud complaint. `app/liability.py` turns one
complaint into a decision an officer can sign: who bears the loss, what compensation is due, and why.

Same rule as the graph: the verdict is deterministic code, never Gemini. Each negligence finding is
rated `proven`, `claimed` or `unknown`, and only `proven` findings can count against the customer.

| Rule | Finding | Outcome |
|---|---|---|
| R1 | Bank negligence proven | Zero liability, full reversal |
| R2 | Third-party breach, reported within 5 days | Zero liability, full reversal |
| R3 | Third-party breach, reported later | The bank's own policy (configurable) |
| R4 | Customer negligence proven, compensation conditions met | 85% of net loss, capped at Rs 25,000 |
| R5 | Customer negligence proven, conditions not met | Customer bears the loss; rejection with reasons |
| R6 | Transactions after the customer's report | Borne by the bank, on top of any outcome |
| R7 | Nothing proven | Bank has not met its burden; not the customer's liability |

```bash
pip install pytest && python -m pytest tests -q      # one test per rule, plus edge days
curl localhost:8080/api/liability/samples             # seven synthetic cases, one per rule
curl -X POST localhost:8080/api/liability/decide -H 'content-type: application/json' \
     -d '{"case": {"case_id": "T1", "transaction_at": "2027-01-18T16:40:00+05:30", "amount_inr": 48000}}'
```

In the pipeline it runs after the Skeptic review and before the brief. Facts come from two places: the
bank's own record (a `.json` file attached with the complaint: transaction time, alert log, scam-warning
log, reporting times) and a Gemini pass over the evidence that rates what the screenshots show. The
rules then decide, and the **Liability** tab shows the verdict, each finding with its evidence, the
compensation conditions and a draft letter to the customer. Transactions before 1 January 2027 are
decided as a clearly marked preview. See `demo/demo_bank_record.json` for the record format.

**Voice intake.** *Record complaint* takes a voice note in Hindi or English. The browser re-encodes it
as 16 kHz WAV, Gemini transcribes it and extracts what the caller stated (payment time, amount, 1930
report, acknowledgement number) and lists follow-up questions for anything missing. The server stamps
the time the recording arrived, and the liability engine uses that stamp as the time the fraud was
reported to the bank, because the outcome depends on it. Caller-stated times are flagged for the
officer to verify. `POST /api/intake` also accepts typed text.

**Letters.** The draft letter to the customer is a fixed template in English and Hindi, filled only
from the decision, so it cannot say more than the rules found.

The rule values come from published summaries of the directions and sit as constants at the top of
`app/liability.py`. Check them against the RBI text before relying on them. Not legal advice.

## Repository layout

```
app/
  main.py           FastAPI API + static hosting
  agent.py          the 5-step pipeline, prompts, tools, offline fallbacks
  gemini_client.py  google-genai wrapper (AI Studio key or Vertex AI)
  graph.py          deterministic relationship engine, vis-network output
  store.py          corpus store + identifier index + similarity
  archive.py        Firestore case archive (optional; app keeps working in memory without it)
  embeddings.py     Gemini embeddings with local fallback
  models.py         Pydantic schemas (structured outputs)
static/index.html   investigator workbench (vanilla JS + vis-network)
scripts/generate_dataset.py   designed synthetic corpus: 80 reports, 3 hidden clusters, decoys, contradictions
data/               reports.json · legit_identifiers.json · ground_truth.json
demo/               ready-made evidence bundle for the live demo
Dockerfile · deploy.sh · run_local.sh · requirements.txt
```

## Run locally

```bash
cp .env.example .env            # add your GEMINI_API_KEY (Google AI Studio)
./run_local.sh                  # http://localhost:8080
```
The default model is `gemini-3.8-flash` (Google lists the 2.5 models as available only to projects that
used them recently). Override with `GEMINI_MODEL` if your key lists a different one.
Without a key the app runs in an offline fallback (regex extraction, rule-based review) so the UI
and graph still work — but the real demo needs Gemini.

## Deploy to Cloud Run

```bash
gcloud auth login && gcloud config set project YOUR_PROJECT
export GEMINI_API_KEY=...        # or switch to Vertex AI env vars in deploy.sh
REGION=asia-south1 ./deploy.sh
```
`deploy.sh` enables Firestore, creates the database if needed, grants the runtime service account
`roles/datastore.user`, then runs `gcloud run deploy --source .` (Cloud Build builds the Dockerfile). First deploy
takes ~3–4 minutes. To use Vertex AI instead of a key, set `GOOGLE_GENAI_USE_VERTEXAI=true`,
`GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` and grant the Cloud Run service account
`roles/aiplatform.user`.

## Firebase Hosting (front door)

```bash
npm install -g firebase-tools     # or: curl -sL https://firebase.tools | bash
firebase login
firebase projects:addfirebase gen-lang-client-0815567638   # once, if the project is not a Firebase project yet
firebase deploy --only hosting
```
`firebase.json` rewrites every request to the `fraudlens` Cloud Run service in `asia-south1`.

## The demo (3 minutes)

1. Open the app. The graph shows 80 complaints as unconnected dots. Point out the header stats.
2. Click **Load demo case** → statement + two screenshots are attached. Show the screenshots briefly.
3. Click **Investigate**. Narrate the agent timeline as it appears:
   * Gemini reads the receipt and chat → pulls `sanjay.das72@upi`, the inspector's phone, ₹48,500, "digital arrest".
   * Agent looks up the UPI → 3 hits, reads them, finds a second phone, follows it → 3 more.
   * It looks up `1930` → marked legitimate, agent says so.
4. Graph reveals the 6-report cluster, including a **courier-parcel** complaint using a different script
   — the same money mule, a different story.
5. Connections tab: strong / plausible verdicts with the chain shown per report. The 1930 helpline appears as a
   grey "legitimate" node and is never followed, so the decoy complaints that mention 1930 stay out of the cluster.
   Story-only matches are shown dashed and capped at **weak**.
6. Brief tab: scroll to *Related complaints*, *Unresolved questions*, and the *Draft escalation note*.
7. Close: "Five minutes ago this was one more complaint. Now it's a case with six victims, two phones,
   two UPI IDs, and a draft escalation an officer can sign."

Ground truth for judges: `GET /api/truth` (planted clusters A/B/C, decoys, contradictions).

## Measured on the synthetic corpus

Each planted report was treated as a new complaint, and we checked what the deterministic graph recovered
(offline mode, local embeddings):

| Cluster | Linked by | Result |
|---|---|---|
| A (digital arrest ring, 6 reports) | shared phones + UPI IDs | 6 of 6 seeds recovered all 5 other members, 0 false identifier links |
| B (electricity-bill ring, 6 reports) | shared URL, UPI, phone | 6 of 6 seeds recovered all 5 other members, 0 false identifier links |
| C (task scam, tactic only) | narrative similarity | partial (1-3 of 4 per seed); similarity is a weak signal, so these links are capped at "weak" |
| Decoys (share only the 1930 helpline) | n/a | 0 pulled into any cluster |

Identifier chains are reliable; story similarity alone is not, and the product says so.

## Judging criteria → where to look

| Criterion | FraudLens |
|---|---|
| Technical merit & Gen AI (40%) | Multimodal structured extraction; tool-using agent with automatic function calling; Gemini embeddings; adversarial Skeptic pass; grounded generation. All on Cloud Run. |
| Problem alignment & impact (25%) | BFSI fraud: unstructured complaints → explainable cross-case leads. India's 1930/NCRP ecosystem. Cuts hours of manual cross-referencing to a minute. |
| Innovation (25%) | Evidence-first chains instead of scores; agent decides the investigation path; Skeptic kills false links (helplines, generic merchants) before a human sees them. |
| UX (10%) | One screen: intake → live agent trace → network reveal → verdicts → brief. Click a node to jump to its evidence trail. |

## Roadmap (for the deck)
* BigQuery analytics over the Firestore archive, Pub/Sub intake from the bank's complaint system and NCRP exports.
* Audio evidence (call recordings) and PDF statements — already supported by the extraction step.
* Investigator feedback loop: confirmed/rejected links retrain the Skeptic's thresholds.
* Victim-facing intake assistant that organises evidence and files the 1930/NCRP complaint.

## Safety and limits
Synthetic data only. The system identifies **evidentiary links between complaints**; it never asserts
guilt, never freezes accounts, and every freeze/escalation is a draft for an authorised officer.
