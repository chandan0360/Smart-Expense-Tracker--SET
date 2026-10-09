"""Intelligent Merchant, Vendor, Company, and Counterparty Extractor for SET.

Extracts canonical entity names (e.g. Swiggy, Uber, Netflix, AWS, Acme Pvt Ltd)
from unstructured Indian bank statement narrations and UPI/NEFT/RTGS/IMPS strings.
"""

from __future__ import annotations

import re

# Comprehensive canonical brand & company lookup (pattern -> Canonical Display Name)
_KNOWN_ENTITIES: list[tuple[str, str]] = [
    # Food & Quick Commerce
    (r"\bswiggy\b", "Swiggy"),
    (r"\bzomato\b", "Zomato"),
    (r"\bblinkit\b", "Blinkit"),
    (r"\bzepto\b", "Zepto"),
    (r"\bbigbasket\b|\bbbnow\b", "BigBasket"),
    (r"\bdunzo\b", "Dunzo"),
    (r"\bmcdonald'?s?\b|\bmcd\b", "McDonald's"),
    (r"\bdomino'?s?\b", "Domino's"),
    (r"\bstarbucks\b|\btata starbucks\b", "Starbucks"),
    (r"\bkfc\b", "KFC"),
    (r"\bpizza hut\b", "Pizza Hut"),
    (r"\bsubway\b", "Subway"),
    (r"\bburger king\b", "Burger King"),
    (r"\bhaldiram'?s?\b", "Haldiram's"),
    (r"\bchaayos\b", "Chaayos"),
    (r"\bchai point\b", "Chai Point"),
    (r"\bfreshmenu\b", "FreshMenu"),
    (r"\bfaasos\b|\brebel foods\b", "Faasos / Rebel Foods"),
    (r"\bbehrouz\b", "Behrouz Biryani"),

    # Mobility & Travel
    (r"\buber\b", "Uber"),
    (r"\bola\b|\bola cabs\b", "Ola"),
    (r"\brapido\b", "Rapido"),
    (r"\birctc\b", "IRCTC"),
    (r"\bmakemytrip\b|\bmmt\b", "MakeMyTrip"),
    (r"\bcleartrip\b", "Cleartrip"),
    (r"\byatra\b", "Yatra"),
    (r"\bindigo\b|\binterglobe\b", "IndiGo"),
    (r"\bair india\b", "Air India"),
    (r"\bvistara\b", "Vistara"),
    (r"\bspicejet\b", "SpiceJet"),
    (r"\bakasa\b", "Akasa Air"),
    (r"\bredbus\b", "redBus"),
    (r"\bfastag\b|\bnhai\b|\bnetc\b", "FASTag / Toll"),

    # Fuel & Energy
    (r"\bindian oil\b|\biocl\b", "Indian Oil"),
    (r"\bbharat petroleum\b|\bbpcl\b", "Bharat Petroleum (BPCL)"),
    (r"\bhindustan petroleum\b|\bhpcl\b", "HPCL"),
    (r"\bshell\b", "Shell"),

    # Tech & Subscriptions
    (r"\bamazon web services\b|\baws\b", "Amazon Web Services (AWS)"),
    (r"\bgoogle cloud\b|\bgcp\b", "Google Cloud"),
    (r"\bgoogle workspace\b|\bgsuite\b", "Google Workspace"),
    (r"\bgoogle(?:\s+india|\s+play)?\b", "Google"),
    (r"\bapple(?:\s+services|\s+store|\.com)?\b", "Apple"),
    (r"\bnetflix\b", "Netflix"),
    (r"\bspotify\b", "Spotify"),
    (r"\bamazon prime\b|\bprime video\b", "Amazon Prime"),
    (r"\byoutube\b", "YouTube"),
    (r"\bhotstar\b|\bdisney\b", "Disney+ Hotstar"),
    (r"\bgithub\b", "GitHub"),
    (r"\bchatgpt\b|\bopenai\b", "OpenAI / ChatGPT"),
    (r"\bcanva\b", "Canva"),
    (r"\bnotion\b", "Notion"),
    (r"\badobe\b", "Adobe"),
    (r"\bmicrosoft\b|\bmsft\b|\boffice 365\b", "Microsoft"),
    (r"\bslack\b", "Slack"),
    (r"\bzoom(?:\.us)?\b", "Zoom"),
    (r"\bdigitalocean\b", "DigitalOcean"),
    (r"\bcursor\b", "Cursor AI"),
    (r"\bfigma\b", "Figma"),
    (r"\blinkedin\b", "LinkedIn"),
    (r"\batlassian\b|\bjira\b", "Atlassian / Jira"),

    # E-Commerce & Retail
    (r"\bamazon\b", "Amazon"),
    (r"\bflipkart\b", "Flipkart"),
    (r"\bmyntra\b", "Myntra"),
    (r"\bnykaa\b", "Nykaa"),
    (r"\bajio\b", "Ajio"),
    (r"\btata cliq\b", "Tata CLiQ"),
    (r"\breliance retail\b|\breliance fresh\b|\breliance trends\b|\bsmart bazaar\b", "Reliance Retail"),
    (r"\bdmart\b|\bavenue supermarts\b", "DMart"),
    (r"\bcroma\b", "Croma"),
    (r"\bvijay sales\b", "Vijay Sales"),
    (r"\bdecathlon\b", "Decathlon"),
    (r"\bikea\b", "IKEA"),
    (r"\blenskart\b", "Lenskart"),
    (r"\burban company\b|\burbanclap\b", "Urban Company"),

    # FinTech & Payment Gateways
    (r"\bstripe(?:\s+payments)?\b", "Stripe"),
    (r"\brazorpay\b", "Razorpay"),
    (r"\bcashfree\b", "Cashfree"),
    (r"\bpayu\b", "PayU"),
    (r"\bphonepe\b", "PhonePe"),
    (r"\bpaytm\b", "Paytm"),
    (r"\bcred\b", "CRED"),
    (r"\bpine labs\b", "Pine Labs"),
    (r"\bbilldesk\b", "BillDesk"),
    (r"\bccavenue\b", "CCAvenue"),
    (r"\bzerodha\b", "Zerodha"),
    (r"\bgroww\b", "Groww"),
    (r"\bangel one\b", "Angel One"),

    # Logistics & Freight
    (r"\bdelhivery\b", "Delhivery"),
    (r"\bbluedart\b", "Blue Dart"),
    (r"\bporter\b", "Porter"),
    (r"\bvrl logistics\b", "VRL Logistics"),
    (r"\btci express\b|\btci freight\b", "TCI Freight"),
    (r"\bgati\b", "Gati Courier"),
    (r"\bfedex\b", "FedEx"),
    (r"\bdhl\b", "DHL Express"),

    # Hardware & IT
    (r"\bdell(?:\s+technologies)?\b", "Dell Technologies"),
    (r"\bhp india\b|\bhewlett packard\b", "HP"),
    (r"\blenovo\b", "Lenovo"),

    # Telecom & Utilities
    (r"\bairtel\b", "Airtel"),
    (r"\bjio fiber\b|\bjio\b|\breliance jio\b", "Jio"),
    (r"\bvodafone\b|\bvi india\b", "Vodafone Idea (Vi)"),
    (r"\btata power\b", "Tata Power"),
    (r"\badani electricity\b", "Adani Electricity"),
    (r"\bbescom\b", "BESCOM"),
    (r"\bact fibernet\b", "ACT Fibernet"),

    # Co-working & Real Estate
    (r"\bwework\b", "WeWork India"),
    (r"\bawfis\b", "Awfis"),
    (r"\bindiqube\b", "IndiQube"),

    # Statutory & Tax Authorities
    (r"\bgst payment\b|\bgstin\b|\bgood and services tax\b", "GST / Tax Authority"),
    (r"\btds challan\b|\badvance tax\b|\bincome tax\b", "Income Tax Dept"),
    (r"\bepfo\b|\bepf contribution\b", "EPFO (Provident Fund)"),
    (r"\besic\b|\besi\b", "ESIC"),
    (r"\bmca\b|\broc\b", "Ministry of Corporate Affairs (MCA)"),
]

_CLEAN_WORDS_RE = re.compile(
    r"\b(pvt ltd|private limited|ltd|limited|corp|corporation|inc|llp|"
    r"pos|ind|in|inward|outward|dr|cr|neft|rtgs|imps|upi|ach|nach|cms|"
    r"bill|pmt|payment|charges|transfer|trf|refuel|refueling|lease|"
    r"mumbai|bangalore|bengaluru|delhi|hyderabad|chennai|kolkata|pune|"
    r"services|solutions|enterprises|india)\b",
    re.I,
)


def extract_entity_name(narration: str, category: str = "") -> str:
    """Extracts a clean, human-readable company, vendor, or customer name from a narration."""
    if not narration:
        return "Unknown Entity"

    raw = narration.strip()
    low = raw.lower()

    # 1. Match known brands/merchants first (High Precision)
    for pattern, canonical_name in _KNOWN_ENTITIES:
        if re.search(pattern, low):
            return canonical_name

    # 2. UPI Pattern Extraction:
    # e.g. "UPI/RAMESH SHARMA/ramesh@okhdfcbank/Transfer" -> "Ramesh Sharma"
    # e.g. "UPI/SWIGGY/8821 POS IND" -> "Swiggy"
    # e.g. "UPI DR APEX ENTERPRISES BANGALORE" -> "Apex Enterprises"
    m_upi = re.search(
        r"upi(?:/|\s+(?:cr|dr)?\s+)(?:[a-z0-9_.\-]+@\w+|[0-9]{4,}/)?([a-zA-Z0-9\.\s\-&]+?)(?:/|[0-9]{4,}|\s+pos|\s+ind|\s+transfer|$)",
        low,
    )
    if m_upi:
        candidate = m_upi.group(1).strip()
        cleaned = _clean_candidate(candidate)
        if len(cleaned) >= 2:
            return cleaned

    # 3. NEFT / RTGS / IMPS Beneficiary or Remitter Extraction:
    # e.g. "NEFT CR-CITI0000001-STRIPE PAYMENTS-SAAS REVENUE"
    # e.g. "NEFT DR TATA MOTORS COMMERCIAL TRUCK EMI"
    # e.g. "NEFT CR CLIENT CLOUD SERVICES INV 8820"
    m_bank_wire = re.search(
        r"(?:neft|rtgs|imps)(?:\s+(?:cr|dr))?(?:[-/][a-z0-9]+)?[-/\s]+([a-zA-Z0-9\.\s\-&]{3,45}?)(?:[-/\s]+(?:inv|invoice|revenue|payment|charges|saas|cms|pmt|emi|\d{4,})|$)",
        low,
    )
    if m_bank_wire:
        candidate = m_bank_wire.group(1).strip()
        cleaned = _clean_candidate(candidate)
        if len(cleaned) >= 2:
            return cleaned

    # 4. Cheque / CHQ patterns:
    # e.g. "CHQ DR WEWORK INDIA MG ROAD LEASE JAN"
    # e.g. "CHQ DR BHIWANDI LOGISTICS PARK WAREHOUSE RENT"
    m_chq = re.search(
        r"(?:chq|cheque)(?:\s+(?:cr|dr))?[-/\s]+([a-zA-Z0-9\.\s\-&]{3,40}?)(?:[-/\s]+(?:lease|rent|payment|\d{4,})|$)",
        low,
    )
    if m_chq:
        candidate = m_chq.group(1).strip()
        cleaned = _clean_candidate(candidate)
        if len(cleaned) >= 2:
            return cleaned

    # 5. POS / Card patterns:
    # e.g. "POS INDIAN OIL BANGALORE"
    # e.g. "POS DR DELL TECHNOLOGIES DEV LAPTOPS"
    m_pos = re.search(
        r"pos(?:\s+(?:cr|dr))?[-/\s]+([a-zA-Z0-9\.\s\-&]{3,40}?)(?:[-/\s]+(?:mumbai|bangalore|delhi|ind|in|\d{4,})|$)",
        low,
    )
    if m_pos:
        candidate = m_pos.group(1).strip()
        cleaned = _clean_candidate(candidate)
        if len(cleaned) >= 2:
            return cleaned

    # 6. Specific Semantic Fallbacks
    if "atm" in low or "cash wdl" in low or category == "ATM/Cash":
        return "ATM Cash Withdrawal"
    if "salary" in low or "payroll" in low or category == "Salary":
        # Extract company or staff group if available
        sal_m = re.search(r"salary\s+(?:to\s+|for\s+)?([a-zA-Z\s]{3,30})", low)
        if sal_m:
            c = _clean_candidate(sal_m.group(1).strip())
            if len(c) > 2:
                return f"{c} (Payroll)"
        return "Staff Payroll / Salary"
    if "house rent" in low or "landlord" in low:
        return "House Landlord"
    if "fastag" in low or "toll" in low:
        return "NHAI FASTag Toll"

    # 7. Fallback: Clean words from narration
    cleaned = _clean_candidate(raw)
    if cleaned and len(cleaned) >= 3:
        words = cleaned.split()
        return " ".join(words[:4])

    return category or "General Counterparty"


def _clean_candidate(text: str) -> str:
    """Cleans a raw candidate substring into a crisp capitalized name."""
    if not text:
        return ""
    # Strip delimiters and digits >= 4 characters
    t = re.sub(r"[/_\-:@|\\#]+", " ", text)
    t = re.sub(r"\b[a-zA-Z0-9]*\d{3,}[a-zA-Z0-9]*\b", " ", t)
    t = re.sub(r"[^\w\s&.]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()

    # Filter out common banking noise words
    tokens = t.split()
    keep = []
    noise = {
        "neft", "rtgs", "imps", "upi", "pos", "dr", "cr", "ach", "nach", "cms",
        "ind", "in", "bill", "pmt", "ref", "rrn", "utr", "txn", "val", "chq",
        "transfer", "trf", "payment", "inv", "invoice", "charges", "charges",
        "mumbai", "bangalore", "bengaluru", "delhi", "chennai", "hyderabad", "pune",
        "kolkata", "branch", "settlement", "disbursement",
    }
    for tok in tokens:
        if tok.lower() not in noise:
            keep.append(tok)

    result = " ".join(keep).strip()
    # Format nicely in title case while preserving acronyms
    words = result.split()
    formatted = []
    acronyms = {"aws", "gst", "irctc", "ibm", "tcs", "hcl", "iocl", "bpcl", "hpcl", "mca", "roc", "fastag"}
    for w in words:
        if w.lower() in acronyms:
            formatted.append(w.upper())
        else:
            formatted.append(w.capitalize())

    return " ".join(formatted)
