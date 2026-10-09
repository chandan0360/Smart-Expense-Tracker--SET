"""Load trained models and classify narration / bank header text with hybrid AI."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
import joblib

from ai_engine.categories import CATEGORIES

ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"


class ModelNotTrainedError(RuntimeError):
    pass


def clean_narration(text: str) -> str:
    """Normalize and clean bank transaction narration text."""
    t = (text or "").lower()
    # Replace delimiters and separators with spaces
    t = re.sub(r"[/_\-:@|\\#]+", " ", t)
    # Remove alphanumeric tokens containing 4 or more digits (RRN, UTR, Mandate ID, Account tails)
    t = re.sub(r"\b[a-z0-9]*\d{4,}[a-z0-9]*\b", " ", t)
    # Remove isolated punctuation
    t = re.sub(r"[^\w\s]", " ", t)
    # Collapse whitespace
    return re.sub(r"\s+", " ", t).strip()


@lru_cache(maxsize=1)
def _load(name: str):
    path = ARTIFACT_DIR / name
    if not path.exists():
        raise ModelNotTrainedError(
            f"Missing {path.name}. Run: python manage.py train_ai"
        )
    return joblib.load(path)


def clear_cache():
    """Clear cached models so retraining reloads immediately."""
    _load.cache_clear()


# High-precision keyword and merchant pattern rules tailored for Corporate & Business Bank Statements
_PATTERNS: list[tuple[str, str, float]] = [
    (
        r"\b(gst|gstin|cpin|gstpmt|gstr|tds|challan 281|challan 280|advance tax|epfo|epf contribution|ecr trrn|esic|esi corp|professional tax|pt payment|income tax|itns|customs duty|icegate|cbic|roc mca|mca21|stamp duty roc|statutory tax|tax assessment|direct tax challan)\b",
        "Taxes/GST",
        0.99,
    ),
    (
        r"\b(chartered accountant|ca fees|statutory audit|tax audit|audit fee|internal audit|secretarial audit|legal retainer|advocate fee|company secretary|cs fees|management consulting|patent trademark|architect design|legal counsel|valuation advisory|advisory fee)\b",
        "Professional Fees",
        0.98,
    ),
    (
        r"\b(aws cloud|amazon web services|google cloud|gcp|microsoft azure|azure cloud|zoho|tally solutions|tally software|github|slack tech|atlassian|jira|notion|figma|zoom video|canva|godaddy|digitalocean|cloudflare|postman|chatgpt|openai|claude|microsoft 365|adobe)\b",
        "Subscription",
        0.98,
    ),
    (
        r"\b(delhivery|bluedart|blue dart|porter courier|vrl logistics|tci freight|gati|fedex|dhl worldwide|freight|cargo shipping|trucking transport|parcel service|logistics|container corp|concor|safeexpress|dtdc|customs clearing agent|c&f agent)\b",
        "Logistics",
        0.98,
    ),
    (
        r"\b(salary|payroll|stipend|wages|company salary|monthly salary|direct salary dep|director remuneration|bulk salary|salary cms|bulk payroll|staff salary|worker wages|salary disbursement|executive compensation)\b",
        "Salary",
        0.99,
    ),
    (
        r"\b(working capital loan|cash credit|cc int|od int|overdraft interest|term loan|machinery loan|equipment finance|sidbi|emi|loan|repayment|repayments|borrowrep|borrowrepa|borrow repayment|slice|slice sm|cred repayment|onecard repayment|uni card|simpl repayment|lazypay|postpe|bajaj finance|bajaj finserv|hinduja ?leyland|hindujaleyland|tata capital|cholamandalam|hdbfs|hdb financial|kreditbee|navi finserv|hero fincorp|shriram finance|home credit|muthoot|manappuram|dmi finance|poonawalla|smart emi|card emi|loan account|loan repayment|achdr loan|nach loan|ecs loan|vehicle loan)\b",
        "EMI",
        0.98,
    ),
    (
        r"\b(razorpay|stripe payout|stripe|cashfree|payu|pine labs|billdesk|ccavenue|instamojo|paytm merchant|phonepe merchant|merchant settlement|merchant payout|client invoice|client payment|customer payment|consulting revenue|sales revenue|export receivables|inward remittance|firc|foreign inward|inward rtgs|inward neft|milestone payment|client settlement|sales proceeds)\b",
        "Client Inflow",
        0.98,
    ),
    (
        r"\b(commercial office lease|warehouse rent|wework india|awfis|smartworks|office rent|house rent|flat rent|room rent|rent upi|landlord|society maintenance|nobroker rent|commercial premises rent|commercial rent)\b",
        "Rent",
        0.98,
    ),
    (
        r"\b(vendor|supplier|raw material|procurement|inventory purchase|packaging|contractor|subcontractor|fabrication|job work|purchase order|b2b invoice|supply chain|oem parts|industrial spares|wholesale trader|distributors|wholesalers|traders|enterprises|pvt ltd|private limited|llp|commercial supplier|corp vendor|vendor cms|component supplies)\b",
        "Vendor Payment",
        0.98,
    ),
    (
        r"\b(bescom|tata power|adani elec|electricity|power bill|broadband|airtel broadband|jio fiber|act fiber|fibernet|bwssb|water bill|water tax|gail|gas bill|gas cylinder|dth|tata play|airtel dth|property tax|municipal tax|mobile recharge|utility bill|current acct|bank charges?|bank chg|consolidated charges?|pos rental|edc rental|mdr charges?|folio charges?|sms charges?|min bal|maintenance chg|annual maintenance)\b",
        "Bills",
        0.98,
    ),
    (
        r"\b(atm|cash wdl|cash withdrawal|nfs atm|cdm deposit|cash w/d|atm cash dispenser|petty cash)\b",
        "ATM/Cash",
        0.99,
    ),
    (
        r"\b(petrol|diesel|fuel|iocl|bpcl|hpcl|indian oil|bharat petroleum|hindustan petroleum|shell fuel|nayara|petrol pump|cng station|cng auto gas|ev fast charger|ather grid|statiq ev|fleet fuel)\b",
        "Fuel",
        0.98,
    ),
    (
        r"\b(group health insurance|commercial asset insurance|fire burglary insurance|directors officers d and o|transit insurance|lic|lic premium|life insurance|health insurance|hdfc life|icici lombard|star health|policybazaar|max life|tata aia|sbi life|acko|care health|niva bupa|hdfc ergo|new india assurance)\b",
        "Insurance",
        0.98,
    ),
    (
        r"\b(interest credit|int\.? ?pd|interest received|sb interest|savings bank interest|fd interest|term deposit interest|groww|zerodha|upstox|angel one|5paisa|dhan app|mutual fund|sip|securities|shares|stocks|nse|bse|ppf deposit|nps contribution|indmoney|kuvera|coin zerodha|demat amc|parag parikh|mirae asset|uti mutual fund|treasury fund|liquid fund|fixed deposit corporate)\b",
        "Investment",
        0.98,
    ),
    (
        r"\b(blinkit|zepto|bigbasket|instamart|dmart|reliance fresh|reliance smart|smart bazaar|spencers|natures basket|more supermarket|kirana|supermarket|vegetables|fruits|licious|freshtohome|country delight|amul|nandini|mother dairy|groceries|grocery|milkbasket|akshayakalpa|sabzi mandi|provision store|pantry supplies)\b",
        "Groceries",
        0.98,
    ),
    (
        r"\b(swiggy|zomato|dominos|pizza hut|mcdonalds|mcdonald|kfc|starbucks|burger king|cafe coffee day|subway|barbeque nation|haldiram|theobroma|biryani|chai point|chaayos|eat\.fit|eatclub|faasos|behrouz|oven story|restaurant|cafe|bakery|diner|kitchen|canteen|darshini|baskin robbins|dunkin donuts|wow momo|bikanervala|bikaji|saravana bhavan|sagar ratna|udupi hotel|dhaba|client lunch|office catering)\b",
        "Food",
        0.98,
    ),
    (
        r"\b(irctc|uber|ola|rapido|namma yatri|makemytrip|goibibo|cleartrip|yatra|red ?bus|abhibus|indigo|air india|vistara|spicejet|akasa air|flight|airlines|railways|metro|dmrc|bmrc|fastag|toll plaza|parking fee|airport parking|blusmart|bmtc bus|booking\.com|agoda hotel|oyo rooms|corporate flight|business hotel)\b",
        "Travel",
        0.98,
    ),
    (
        r"\b(amazon|flipkart|myntra|ajio|nykaa|meesho|tata cliq|croma|reliance digital|decathlon|ikea|uniqlo|zara|h&m|hm|westside|pantaloons|max fashion|zudio|lenskart|titan|tanishq|caratlane|snitch|bewakoof|urbanic|retail store|clothing|footwear|mall shopping|bata shoes|metro shoes|lifestyle store|shoppers stop|office furniture|stationery)\b",
        "Shopping",
        0.97,
    ),
    (
        r"\b(pvr|inox|cinepolis|bookmyshow|steam purchase|steam|spotify|netflix|hotstar|sonyliv|zee5|jiocinema|playstation|xbox|movie|cinema|theatre|gaming|carnival cinemas|wonderla|smaaash|timezone gaming|bowling|comedy show|concert|team offsite)\b",
        "Entertainment",
        0.98,
    ),
    (
        r"\b(apollo pharmacy|pharmeasy|1mg|tata 1mg|netmeds|medplus|fortis|manipal|max healthcare|practo|hospital|clinic|pharmacy|chemist|pathlab|dr lal|diagnostic|medicals|dentist|wellness forever|narayana health|aster cmi|medanta|blood test|srl diagnostics|metropolis lab|thyrocare|employee checkup)\b",
        "Healthcare",
        0.98,
    ),
    (
        r"\b(coursera|udemy|byjus|unacademy|physicswallah|khan academy|edx|upgrad|college fee|school tuition|university|tuition|coaching|exam fee|admission fee|semester fee|cbse fee|simplilearn|scaler academy|allen career|aakash institute|fiitjee|book depot|corporate training)\b",
        "Education",
        0.98,
    ),
]


def classify_category(description: str, user=None) -> tuple[str, float]:
    """
    Classify transaction narration using a hybrid pipeline:
    0. Check active learning CategoryRule database for user-learned overrides.
    1. Check for blank / table noise artifacts.
    2. Check high-precision keyword / merchant regex rules.
    3. Detect NPCI UPI P2P transfers (individual beneficiary/remitter).
    4. Fall back to trained scikit-learn TF-IDF + Logistic Regression model.
    """
    text = (description or "").strip()
    if not text:
        return "Other", 0.0

    cleaned = clean_narration(text)
    if not cleaned:
        return "Other", 0.0

    # 0. Active learning: check learned user rules first
    try:
        from django.db.models import Q
        from tracker.models import CategoryRule
        qs = CategoryRule.objects.all()
        if user and getattr(user, "is_authenticated", False):
            qs = qs.filter(Q(user=user) | Q(user__isnull=True))
        for rule in qs:
            kw = (rule.keyword or "").strip().lower()
            if kw and (kw in cleaned or kw in text.lower()):
                return rule.category, 1.0
    except Exception:
        pass

    # Filter obvious balance or header noise
    if any(term in cleaned for term in ("opening balance", "closing balance", "bualllance", "balance b f", "balance c f")):
        return "Other", 0.0

    # 1. High-precision rulebook matching
    for pattern, cat, conf in _PATTERNS:
        if re.search(pattern, cleaned):
            return cat, conf

    # 2. Check Indian NPCI UPI P2P patterns: UPI/(DR|CR)/<rrn>/<Entity>/...
    m = re.search(r"upi/(?:dr|cr)/\d+/([^/]+)", text, re.IGNORECASE)
    if m:
        entity_raw = m.group(1).strip()
        entity_cleaned = clean_narration(entity_raw)
        # Verify entity against merchant patterns first
        for pattern, cat, conf in _PATTERNS:
            if re.search(pattern, entity_cleaned):
                return cat, conf
        # If not matching any merchant, it's a person-to-person transfer
        return "Transfer", 0.92

    # Generic bank transfer signals
    if re.search(r"\b(neft|imps|rtgs|fund transfer|self transfer|account transfer|to savings|sent to|received from|phonepe transfer|gpay transfer|paytm transfer)\b", cleaned):
        return "Transfer", 0.90

    # 3. Machine Learning model fallback
    try:
        model = _load("category_model.joblib")
        # Cleaned narration vectorization
        pred_label = model.predict([cleaned])[0]
        proba = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba([cleaned])[0]
            classes = list(model.classes_)
            if pred_label in classes:
                proba = float(probs[classes.index(pred_label)])

        if pred_label in CATEGORIES and proba >= 0.20:
            # Calibrate confidence to make it user-intuitive
            calibrated_conf = min(0.95, round(proba * 1.3, 2)) if proba >= 0.4 else round(proba, 2)
            return str(pred_label), calibrated_conf
    except Exception:
        pass

    return "Other", 0.0


def classify_bank(header_text: str) -> tuple[str, float]:
    text = (header_text or "").strip()
    if not text:
        return "Unknown Bank", 0.0
    model = _load("bank_model.joblib")
    label = str(model.predict([text])[0])
    proba = 0.0
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba([text])[0]
        classes = list(model.classes_)
        if label in classes:
            proba = float(probs[classes.index(label)])
    return label, proba


def classify_transaction(text: str, user=None) -> tuple[str, float]:
    """Alias for classify_category."""
    return classify_category(text, user=user)


def predict_category(text: str, user=None) -> str:
    """Predict category string for given narration."""
    category, _ = classify_category(text, user=user)
    return category
