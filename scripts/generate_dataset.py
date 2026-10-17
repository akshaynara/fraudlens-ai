"""
Generate the synthetic fraud-report corpus FraudLens AI investigates against.

The dataset is designed, not random:
  * Cluster A  "digital arrest" ring   - 6 reports chained through 2 phones + 2 UPI IDs.
                                          No single report touches every identifier; the
                                          network only appears when you follow the chain.
  * Cluster B  electricity-bill ring    - 5 reports sharing a phishing URL, 3 share a UPI,
                                          1 bridges via phone to an unrelated loan-app report.
  * Cluster C  task/job scam            - 5 reports with the SAME tactic but NO shared
                                          identifiers (only narrative similarity), except two
                                          that share a Telegram handle.
  * Decoys     - pairs of unrelated reports that share a LEGITIMATE identifier
                 (1930 helpline, SBI toll-free). Must NOT be treated as a link.
  * Contradictions - reports whose narrative amount differs from the receipt amount.
  * Noise      - unique, unrelated reports.

Ground truth is stored under "_truth" and is stripped before the agent sees anything.

Usage:
    python scripts/generate_dataset.py            # writes data/reports.json + demo/ files
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

random.seed(2026)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEMO = ROOT / "demo"
DATA.mkdir(exist_ok=True)
DEMO.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- helpers
FIRST = ["rahul", "priya", "amit", "sneha", "vikas", "neha", "arjun", "pooja", "rohit", "kavya",
         "suresh", "anjali", "deepak", "meena", "karan", "ritu", "manoj", "divya", "sanjay", "isha"]
LAST = ["sharma", "verma", "patel", "singh", "iyer", "reddy", "nair", "gupta", "das", "mehta"]
UPI_SUFFIX = ["okaxis", "okhdfcbank", "oksbi", "ybl", "paytm", "ibl", "axl", "upi"]
CITIES = ["Bengaluru", "Hyderabad", "Pune", "Mumbai", "Chennai", "Delhi", "Kolkata", "Jaipur",
          "Ahmedabad", "Lucknow", "Kochi", "Indore"]
BANKS = ["SBI", "HDFC Bank", "ICICI Bank", "Axis Bank", "Kotak", "PNB", "Bank of Baroda"]

LEGIT_IDENTIFIERS = {
    "1930": "National cyber-crime helpline",
    "1800-1234": "SBI toll-free customer care",
    "1800-425-3800": "SBI toll-free customer care",
    "cybercrime.gov.in": "National Cyber Crime Reporting Portal",
}

_used: set[str] = set()


def fresh(gen):
    while True:
        v = gen()
        if v not in _used:
            _used.add(v)
            return v


def upi():
    return fresh(lambda: f"{random.choice(FIRST)}.{random.choice(LAST)}{random.randint(10, 99)}@{random.choice(UPI_SUFFIX)}")


def phone():
    return fresh(lambda: f"+91 {random.choice('6789')}{random.randint(100000000, 999999999)}")


def account():
    return fresh(lambda: str(random.randint(10**13, 10**14 - 1)))


def url():
    stems = ["bescom-bill-pay", "kseb-update", "mseb-recharge", "india-post-track", "kyc-verify-sbi",
             "parcel-release", "tax-refund-india", "fastag-renew", "pm-yojana-claim"]
    return fresh(lambda: f"https://{random.choice(stems)}-{random.randint(100, 999)}.{random.choice(['xyz', 'top', 'online', 'site', 'link'])}")


def tg():
    return fresh(lambda: f"@{random.choice(['hr', 'task', 'earn', 'promo', 'job'])}_{random.choice(['india', 'desk', 'team', 'pro'])}{random.randint(10, 999)}")


def amount(lo=4000, hi=250000):
    return int(round(random.randint(lo, hi), -2))


def date_between(days_back_lo=1, days_back_hi=120):
    d = datetime(2026, 10, 16) - timedelta(days=random.randint(days_back_lo, days_back_hi),
                                           hours=random.randint(0, 23))
    return d


# --------------------------------------------------------------------------- narratives
# Each template is (channel, evidence_types, text). Placeholders filled per report.
TEMPLATES = {
    "digital_arrest": [
        ("call+whatsapp", ["whatsapp_screenshot", "upi_receipt", "call_log"],
         "I got a call from {phone} claiming to be from the Mumbai Cyber Cell / CBI. They said a parcel "
         "in my name with drugs was seized and an arrest warrant was issued. They moved me to a WhatsApp "
         "video call where a man in uniform told me I was under 'digital arrest' and could not disconnect. "
         "They made me transfer Rs {amount} to {upi} as 'verification deposit' promising refund in 24 hours. "
         "I was on the call for almost {hours} hours."),
        ("call+whatsapp", ["whatsapp_screenshot", "upi_receipt"],
         "A man called from {phone} saying my Aadhaar was used for money laundering and I had to cooperate "
         "with a 'Supreme Court' video hearing on WhatsApp. They showed a fake ID card and FIR copy. Under "
         "pressure I paid Rs {amount} to UPI {upi} which they said would be returned after 'RBI verification'. "
         "They warned me not to tell family members."),
    ],
    "courier_parcel": [
        ("call", ["call_log", "upi_receipt"],
         "Received a call from {phone} about a FedEx/India Post parcel with illegal items booked in my name. "
         "They transferred me to a 'police officer' who asked me to pay Rs {amount} to {upi} to get the case "
         "closed quietly. After payment both numbers switched off."),
    ],
    "electricity_bill": [
        ("sms", ["sms_screenshot", "upi_receipt"],
         "Got an SMS saying my electricity connection would be disconnected tonight because last month's bill "
         "was not updated, with a link {url}. The page looked exactly like the board's website. I entered my "
         "details and paid Rs {amount} to {upi}. Then I got OTP messages and Rs {amount2} more was debited."),
        ("sms+call", ["sms_screenshot", "call_log", "bank_statement"],
         "SMS from an unknown number said 'Dear consumer your power will be cut at 9.30 PM, contact officer "
         "{phone}'. I called and he asked me to install a quick-support app from {url} to update the bill. "
         "Within 20 minutes Rs {amount} was transferred out of my {bank} account."),
    ],
    "task_job": [
        ("telegram", ["telegram_screenshot", "upi_receipt", "bank_statement"],
         "A WhatsApp message offered part-time work liking YouTube videos and rating hotels, Rs 50-150 per task. "
         "The first few tasks paid out small amounts. Then I was added to a Telegram group{tg_clause} and asked "
         "to complete 'prepaid merchant tasks'. I deposited Rs {amount} in total to {upi} and my wallet showed "
         "profit but withdrawal kept failing with 'credit score too low, deposit more'."),
        ("telegram", ["telegram_screenshot", "upi_receipt"],
         "Got a job offer for hotel review tasks via Telegram{tg_clause}. Small payments came first, building trust. "
         "Later tasks required a 'deposit' and I sent Rs {amount} to {upi}. The dashboard showed earnings of nearly "
         "double but the admin demanded more tax deposit before withdrawal. I realised it was fake when they "
         "asked for another Rs {amount2}."),
    ],
    "loan_app": [
        ("app", ["app_screenshot", "whatsapp_screenshot"],
         "I downloaded an instant loan app and got Rs {amount} but they demanded Rs {amount2} within 7 days. "
         "Recovery agents from {phone} started sending morphed photos to my contacts and threatening me on "
         "WhatsApp. They asked me to pay to {upi} to stop the harassment."),
    ],
    "kyc_update": [
        ("sms+call", ["sms_screenshot", "bank_statement"],
         "SMS said my {bank} account would be blocked for KYC non-compliance and to call {phone}. The caller "
         "asked me to share the OTP to 'verify PAN link'. Three transactions of Rs {amount} were made to "
         "{upi} within minutes."),
    ],
    "fake_buyer_olx": [
        ("call+whatsapp", ["whatsapp_screenshot", "upi_receipt"],
         "I listed a sofa on OLX. A buyer from {phone} said he was an Army officer being transferred and would "
         "pay in advance via UPI. He sent a QR code and said scanning it would credit me, but Rs {amount} was "
         "debited instead to {upi}. He insisted it was a mistake and sent another QR."),
    ],
    "investment_group": [
        ("whatsapp", ["whatsapp_screenshot", "bank_statement"],
         "I was added to a WhatsApp stock-tips group run by a 'SEBI-registered advisor'. Members posted daily "
         "profits. I invested Rs {amount} through their app by transferring to {upi} and account {account}. "
         "When I tried to withdraw they asked for 20% tax first."),
    ],
}

CONTACT_LEGIT = [
    " I later called {legit} and was told to file a complaint online.",
    " I reported it to {legit} the next morning.",
    "",
    "",
]


def make_narrative(scam_type, fill, legit=None, tmpl_idx=None):
    channel, evidence, text = random.choice(TEMPLATES[scam_type]) if tmpl_idx is None else TEMPLATES[scam_type][tmpl_idx]
    fill.setdefault("hours", random.randint(3, 9))
    fill.setdefault("amount2", amount(2000, 60000))
    fill.setdefault("bank", random.choice(BANKS))
    fill.setdefault("tg_clause", "")
    fill.setdefault("url", "")
    fill.setdefault("phone", "")
    fill.setdefault("upi", "")
    fill.setdefault("account", "")
    narrative = text.format(**fill)
    if legit:
        narrative += random.choice(CONTACT_LEGIT[:2]).format(legit=legit)
    return channel, evidence, narrative


counter = {"n": 0}


def report(scam_type, identifiers, truth, legit=None, contradiction=False, tmpl_idx=None, days=(1, 120)):
    counter["n"] += 1
    rid = f"FR-2026-{counter['n']:04d}"
    amt = amount()
    fill = {"amount": f"{amt:,}"}
    ids = []
    for t, v in identifiers:
        fill[t if t != "telegram" else "tg"] = v
        ids.append({"type": t, "value": v, "source": "narrative"})
    if "tg" in fill:
        fill["tg_clause"] = f" ({fill['tg']})"
    channel, evidence, narrative = make_narrative(scam_type, fill, legit, tmpl_idx)
    if legit:
        ids.append({"type": "phone" if legit[0].isdigit() else "url", "value": legit, "source": "narrative"})
    receipt_amt = amt
    if contradiction:
        receipt_amt = int(amt * random.choice([0.4, 0.55, 1.8]))
    filed = date_between(*days)
    return {
        "id": rid,
        "filed_at": filed.isoformat(timespec="minutes"),
        "victim_city": random.choice(CITIES),
        "scam_type": scam_type,
        "channel": channel,
        "amount_claimed_inr": amt,
        "amount_on_receipt_inr": receipt_amt,
        "evidence_types": evidence,
        "narrative": narrative,
        "identifiers": ids,
        "status": random.choice(["open", "open", "open", "under_review", "closed_unresolved"]),
        "_truth": truth,
    }


reports = []

# ---- Cluster A: digital-arrest ring (chained) --------------------------------------
pA1, pA2 = phone(), phone()
uA1, uA2 = upi(), upi()
reports += [
    report("digital_arrest", [("phone", pA1), ("upi", uA1)], {"cluster": "A"}, tmpl_idx=0, days=(40, 90)),
    report("digital_arrest", [("phone", pA1), ("upi", uA1)], {"cluster": "A"}, tmpl_idx=1, days=(35, 80)),
    report("digital_arrest", [("phone", pA2), ("upi", uA1)], {"cluster": "A"}, tmpl_idx=0, days=(20, 60)),
    report("digital_arrest", [("phone", pA2), ("upi", uA2)], {"cluster": "A"}, tmpl_idx=1, days=(10, 30)),
    report("digital_arrest", [("phone", pA2), ("upi", uA2)], {"cluster": "A"}, legit="1930", tmpl_idx=0, days=(5, 15)),
    report("courier_parcel", [("phone", phone()), ("upi", uA2)], {"cluster": "A", "note": "bridge via UPI, different tactic"}, days=(2, 9)),
]
# Demo case will mention a NEW phone pA3 and uA2 (so it only touches the right-hand end of the chain)
pA3 = phone()

# ---- Cluster B: electricity-bill ring ---------------------------------------------
wB, uB, pB1, pB2 = url(), upi(), phone(), phone()
reports += [
    report("electricity_bill", [("url", wB), ("upi", uB)], {"cluster": "B"}, tmpl_idx=0),
    report("electricity_bill", [("url", wB), ("upi", uB)], {"cluster": "B"}, tmpl_idx=0, contradiction=True),
    report("electricity_bill", [("url", wB), ("upi", uB), ("phone", pB1)], {"cluster": "B"}, tmpl_idx=0),
    report("electricity_bill", [("url", wB), ("phone", pB1)], {"cluster": "B"}, tmpl_idx=1),
    report("electricity_bill", [("url", wB), ("phone", pB2)], {"cluster": "B"}, tmpl_idx=1, legit="1800-1234"),
    report("loan_app", [("phone", pB2), ("upi", upi())], {"cluster": "B", "note": "bridge via phone, different tactic"}),
]

# ---- Cluster C: task/job scam, tactic-only similarity ------------------------------
tC = tg()
reports += [
    report("task_job", [("upi", upi())], {"cluster": "C", "link": "semantic"}, tmpl_idx=0),
    report("task_job", [("upi", upi())], {"cluster": "C", "link": "semantic"}, tmpl_idx=1),
    report("task_job", [("upi", upi()), ("telegram", tC)], {"cluster": "C", "link": "telegram+semantic"}, tmpl_idx=0),
    report("task_job", [("upi", upi()), ("telegram", tC)], {"cluster": "C", "link": "telegram+semantic"}, tmpl_idx=1, contradiction=True),
    report("task_job", [("upi", upi())], {"cluster": "C", "link": "semantic"}, tmpl_idx=1),
]

# ---- Decoys: share only a legitimate identifier ------------------------------------
reports += [
    report("kyc_update", [("phone", phone()), ("upi", upi())], {"decoy": "shares 1930 only"}, legit="1930"),
    report("fake_buyer_olx", [("phone", phone()), ("upi", upi())], {"decoy": "shares 1930 only"}, legit="1930"),
    report("investment_group", [("upi", upi()), ("account", account())], {"decoy": "shares SBI toll-free only"}, legit="1800-425-3800"),
    report("kyc_update", [("phone", phone()), ("upi", upi())], {"decoy": "shares SBI toll-free only"}, legit="1800-425-3800"),
]

# ---- Noise -------------------------------------------------------------------------
noise_types = ["kyc_update", "fake_buyer_olx", "investment_group", "loan_app", "courier_parcel",
               "electricity_bill", "digital_arrest", "task_job"]
while len(reports) < 80:
    st = random.choice(noise_types)
    ids = [("phone", phone()), ("upi", upi())]
    if st == "electricity_bill":
        ids.append(("url", url()))
    if st == "investment_group":
        ids.append(("account", account()))
    if st == "task_job":
        ids = [("upi", upi())] + ([("telegram", tg())] if random.random() < 0.5 else [])
    reports.append(report(st, ids, {"noise": True}, contradiction=random.random() < 0.06))

random.shuffle(reports)
# Re-number after shuffle so IDs don't leak cluster structure
for i, r in enumerate(sorted(reports, key=lambda r: r["filed_at"]), start=1):
    r["id"] = f"FR-2026-{i:04d}"
reports.sort(key=lambda r: r["id"])

(DATA / "reports.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False))
(DATA / "legit_identifiers.json").write_text(json.dumps(LEGIT_IDENTIFIERS, indent=2))

truth_summary = {
    "cluster_A_ids": [r["id"] for r in reports if r["_truth"].get("cluster") == "A"],
    "cluster_B_ids": [r["id"] for r in reports if r["_truth"].get("cluster") == "B"],
    "cluster_C_ids": [r["id"] for r in reports if r["_truth"].get("cluster") == "C"],
    "decoy_ids": [r["id"] for r in reports if r["_truth"].get("decoy")],
    "contradiction_ids": [r["id"] for r in reports if r["amount_claimed_inr"] != r["amount_on_receipt_inr"]],
    "demo_case_identifiers": {"phone_new": pA3, "upi_shared_with_cluster_A": uA2},
}
(DATA / "ground_truth.json").write_text(json.dumps(truth_summary, indent=2))

# --------------------------------------------------------------------------- demo evidence
DEMO_AMOUNT = 48500


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def receipt_png(path):
    img = Image.new("RGB", (720, 1280), "#f4f6f8")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 720, 140], fill="#1f6f43")
    d.text((40, 48), "Payment Successful", font=font(40), fill="white")
    d.text((40, 200), f"Rs {DEMO_AMOUNT:,}", font=font(72), fill="#111")
    d.text((40, 300), "Paid to", font=font(26), fill="#666")
    d.text((40, 340), "Verification Dept - Cyber Cell", font=font(34), fill="#111")
    d.text((40, 390), uA2, font=font(30), fill="#1f6f43")
    d.text((40, 480), "From", font=font(26), fill="#666")
    d.text((40, 520), "Meera Krishnan  |  HDFC Bank  XX4471", font=font(30), fill="#111")
    d.text((40, 620), "16 Oct 2026, 17:42", font=font(28), fill="#444")
    d.text((40, 670), "UPI transaction ID", font=font(26), fill="#666")
    d.text((40, 705), "629041238877", font=font(30), fill="#111")
    d.text((40, 780), "Note: RBI VERIFICATION DEPOSIT - REFUNDABLE", font=font(24), fill="#b00020")
    d.text((40, 1180), "Powered by UPI", font=font(24), fill="#999")
    img.save(path)


def whatsapp_png(path):
    img = Image.new("RGB", (720, 1280), "#e5ddd5")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 720, 120], fill="#075e54")
    d.text((90, 30), "Insp. R. Deshmukh  Cyber Cell", font=font(30), fill="white")
    d.text((90, 72), pA3, font=font(24), fill="#cfe9e3")
    msgs = [
        ("in", "This is Inspector Deshmukh, Mumbai Cyber Cell. A parcel with 400g MDMA booked on your Aadhaar was seized at Andheri."),
        ("in", "A warrant is issued. You are under DIGITAL ARREST. Do not disconnect this call or inform anyone."),
        ("out", "Sir I have never sent any parcel. Please check again."),
        ("in", "To verify your account is clean, transfer the verification deposit to the RBI escrow UPI below. It is refunded within 24 hrs."),
        ("in", uA2),
        ("out", "Done sir, Rs 48,500 sent."),
        ("in", "Good. Stay on the line. Senior officer will join from Supreme Court for your hearing."),
    ]
    y = 150
    for side, text in msgs:
        words, lines, cur = text.split(), [], ""
        for w in words:
            if len(cur + " " + w) > 38:
                lines.append(cur.strip())
                cur = w
            else:
                cur += " " + w
        lines.append(cur.strip())
        h = 30 * len(lines) + 30
        x0 = 40 if side == "in" else 220
        d.rounded_rectangle([x0, y, x0 + 460, y + h], radius=16, fill="white" if side == "in" else "#dcf8c6")
        for i, ln in enumerate(lines):
            d.text((x0 + 16, y + 14 + 30 * i), ln, font=font(24), fill="#111")
        y += h + 18
    img.save(path)


receipt_png(DEMO / "demo_upi_receipt.png")
whatsapp_png(DEMO / "demo_whatsapp_chat.png")

narrative = (
    f"Yesterday evening (16 Oct) I got a call from {pA3}. The man said he was from Mumbai Cyber Cell and that a "
    "parcel containing drugs booked using my Aadhaar had been seized. He moved me to a WhatsApp video call, "
    "showed an ID card and said I was under digital arrest and could not hang up or call anyone. After about "
    f"four hours he made me transfer Rs 48,500 as a 'verification deposit' to a UPI ID he sent in the chat, "
    "promising it would be refunded by RBI in 24 hours. After I paid they said a Supreme Court officer would join, "
    "then the call dropped and the number is now switched off. I have the WhatsApp screenshots and the payment "
    "receipt. I called 1930 this morning."
)
(DEMO / "demo_victim_statement.txt").write_text(narrative)
(DEMO / "README.md").write_text(
    "# Demo evidence bundle\n\n"
    "Upload these three files together as a new case:\n\n"
    "* `demo_upi_receipt.png`  - UPI payment success screen (amount, payee UPI ID)\n"
    "* `demo_whatsapp_chat.png` - WhatsApp conversation with the 'inspector' (phone number, tactic)\n"
    "* `demo_victim_statement.txt` - paste as the victim statement\n\n"
    "Expected outcome: the new case shares one UPI ID with the right-hand end of hidden Cluster A. "
    "Following the chain (UPI -> phone -> UPI -> phone) recovers all 6 related reports, including a "
    "courier-parcel report that uses a different tactic. The 1930 helpline mentioned in the statement "
    "is a known legitimate number, so it must never be followed as a link.\n"
)

print(f"Wrote {len(reports)} reports -> {DATA / 'reports.json'}")
print(json.dumps(truth_summary, indent=2))
