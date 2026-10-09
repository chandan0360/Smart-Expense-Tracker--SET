"""Build a labelled narration dataset for expense category model (supports Business & Company Bank Statements)."""

from __future__ import annotations

import random
import re
from ai_engine.categories import CATEGORIES, INDIAN_BANKS

random.seed(42)

MERCHANTS: dict[str, list[str]] = {
    "Vendor Payment": [
        "SUPPLIER PAY", "VENDOR INVOICE SETTLEMENT", "RAW MATERIALS PROCUREMENT",
        "PACKAGING SUPPLIES", "INVENTORY PURCHASE", "JOB WORK CHARGES", "CONTRACTOR INVOICE",
        "FABRICATION EXPENSE", "HARDWARE SUPPLIERS", "TEXTILE VENDORS", "STEEL SUPPLIERS",
        "CHEMICAL DISTRIBUTORS", "PRINTING PRESS EXPENSES", "ELECTRICAL CONTRACTORS",
        "EQUIPMENT SPARES", "WHOLESALE TRADERS", "B2B INVOICE PAYMENT", "PURCHASE ORDER SETTLEMENT",
        "SHREE TRADERS", "GLOBAL ENTERPRISES", "BALAJI AGENCIES", "SUPPLY CHAIN DISBURSEMENT",
        "RAW MATERIAL INVOICE", "COMPONENT SUPPLIES", "OEM PARTS SUPPLIER", "INDUSTRIAL SPARES",
        "CORP VENDOR PAY", "VENDOR CMS PAYOUT", "B2B PROCUREMENT", "COMMERCIAL SUPPLIER",
        "SHENGZHEN SEMICONDUCTOR PARTS", "COPPER WIRE & COILS", "PLASTIC INJECTION MOULDING DIES",
        "LI-ION BATTERY CELL SUPPLIERS", "MICROCONTROLLER IC CHIPS", "SMT COMPONENT SOURCING",
        "CORRUGATED BOXES PACKAGING BATCH", "INDUSTRIAL FABRICATION CHARGES",
    ],
    "Taxes/GST": [
        "GST PMT", "GSTIN CHALLAN", "CPIN PAYMENT", "GST TAX LIABILITY", "GOODS AND SERVICES TAX",
        "GSTR 3B TAX PAYMENT", "IGST PAYMENT", "CGST SGST PAYMENT", "GST CHALLAN",
        "TDS CHALLAN 281", "TDS PAYMENT ITNS", "ADVANCE TAX CHALLAN 280", "INCOME TAX ASSESSMENT",
        "SELF ASSESSMENT TAX", "EPFO ELECTRONIC CHALLAN", "EPF CONTRIBUTION", "ESIC CHALLAN PMT",
        "PROFESSIONAL TAX PT", "ROC MCA FILING FEE", "STAMP DUTY ROC", "MUNICIPAL TRADE TAX",
        "CUSTOMS DUTY PAYMENT", "GOVERNMENT TAX CHALLAN", "DIRECT TAX DEPOSIT",
    ],
    "Client Inflow": [
        "CLIENT INVOICE RECEIPT", "CUSTOMER PAYMENT RECEIVED", "B2B CLIENT SETTLEMENT",
        "INWARD REMITTANCE CLIENT", "RETAINER FEE CREDIT", "CONSULTING REVENUE CR",
        "EXPORT RECEIVABLES", "PROJECT MILESTONE PMT", "CLIENT SETTLEMENT CR",
        "RAZORPAY SETTLEMENT", "STRIPE PAYOUT CR", "CASHFREE SETTLEMENT", "PAYU PAYMENTS CR",
        "PINE LABS POS SETTLEMENT", "PAYTM MERCHANT PAYOUT", "PHONEPE MERCHANT SETTLEMENT",
        "CCAVENUE PAYOUT", "BILLDESK SETTLEMENT", "CUSTOMER ADVANCE CR", "SALES REVENUE CR",
        "B2B SALES RECEIPT", "CLIENT INVOICE CLEARING", "COMMERCIAL SALES PROCEEDS",
        "INFOSYS B2B CLOUD RETAINER", "ACCENTURE CONSULTING SERVICES", "STRIPE PAYMENTS SAAS REVENUE",
        "CROMPTON GREAVES B2B WHOLESALE", "RELIANCE DIGITAL COMMERCIAL PURCHASE",
        "DUBAI ELECTRONICS TRADING LLC", "VIJAY SALES B2B INVOICE CLEARING",
        "AMAZON SELLER PAYOUT", "FLIPKART INDIA PVT LTD VENDOR SETTLEMENT",
    ],
    "Professional Fees": [
        "CHARTERED ACCOUNTANT FEES", "STATUTORY AUDIT FEES", "TAX AUDIT CHARGES",
        "LEGAL RETAINER FEE", "ADVOCATE LEGAL EXPENSES", "COMPANY SECRETARY CS FEES",
        "MANAGEMENT CONSULTING CHARGES", "PATENT TRADEMARK FILING", "VALUATION ADVISORY CHARGES",
        "ARCHITECT DESIGN FEES", "CONSULTANT PROFESSIONAL CHARGES", "INTERNAL AUDIT FEES",
        "SECRETARIAL AUDIT CHARGES", "LEGAL COUNSEL FEE", "RETAINERSHIP CHARGES",
    ],
    "Logistics": [
        "DELHIVERY LOGISTICS", "BLUEDART EXPRESS", "PORTER COURIER", "VRL LOGISTICS",
        "TCI FREIGHT", "GATI COURIER", "FEDEX EXPRESS", "DHL WORLDWIDE", "CARGO SHIPPING",
        "CONTAINER TRANSPORT", "CUSTOMS CLEARING AGENT", "TRUCKING TRANSPORT",
        "WAREHOUSE FREIGHT", "COURIER EXPENSES", "LOGISTICS DISBURSEMENT", "PARCEL SERVICE",
        "ROADWAYS TRANSPORT", "FAST TRACK CARGO", "DOMESTIC LOGISTICS",
    ],
    "Salary": [
        "BULK SALARY DISBURSEMENT", "PAYROLL BATCH", "STAFF SALARY CREDIT",
        "MONTHLY SALARY DISBURSEMENT", "DIRECTOR REMUNERATION", "CONTRACTOR STIPEND PAYROLL",
        "EMPLOYEE REIMBURSEMENT", "SALARY CREDIT", "PAYROLL", "NEFT SALARY", "COMPANY SALARY",
        "STAFF WAGES CREDIT", "EMPLOYER PAYROLL", "SALARY FOR MONTH", "STIPEND CREDIT",
        "BONUS DISBURSEMENT", "CONTRACT LABOUR SALARY", "EXECUTIVE COMPENSATION",
    ],
    "EMI": [
        "WORKING CAPITAL LOAN EMI", "CASH CREDIT CC INTEREST", "OVERDRAFT OD INTEREST",
        "BUSINESS LOAN REPAYMENT", "MACHINERY LOAN EMI", "EQUIPMENT FINANCE",
        "HINDUJA LEYLAND", "HINDUJALEYLAND FINANCE", "BAJAJ FINANCE EMI", "BAJAJ FINSERV LOAN",
        "TATA CAPITAL LOAN", "CHOLAMANDALAM FINANCE", "HDBFS LOAN", "HDB FINANCIAL SERVICES",
        "MUTHOOT FINANCE EMI", "MANAPPURAM FINANCE", "HOME CREDIT INDIA", "KREDITBEE LOAN",
        "NAVI FINSERV LOAN", "HERO FINCORP LOAN", "SHRIRAM FINANCE LOAN", "L&T FINANCE EMI",
        "SLICE REPAYMENT", "SLICE BORROWREP", "SLICE SM", "SLICE CREDIT", "CRED REPAYMENT",
        "ONECARD REPAYMENT", "UNI CARD REPAYMENT", "SIMPL REPAYMENT", "LAZYPAY REPAYMENT",
        "LOAN EMI DEBIT", "ACHDR LOAN REPAYMENT", "NACH LOAN DEBIT", "ECS LOAN DEBIT",
        "BANK LOAN INSTALLMENT", "TERM LOAN REPAYMENT", "PROJECT LOAN EMI", "VEHICLE LOAN EMI",
    ],
    "Rent": [
        "COMMERCIAL OFFICE LEASE", "WAREHOUSE RENT", "WEWORK INDIA LEASE",
        "AWFIS COWORKING RENT", "SMARTWORKS RENT", "OFFICE RENT EXPENSE", "HOUSE RENT",
        "RENT UPI LANDLORD", "MAINTENANCE SOCIETY", "FLAT RENT", "ROOM RENT", "OFFICE RENT",
        "SOCIETY MAINTENANCE", "APARTMENT MAINTENANCE", "NOBROKER RENT", "COMMERCIAL PREMISES RENT",
    ],
    "Bills": [
        "COMMERCIAL POWER BESCOM", "HIGH TENSION HT POWER", "TATA TELESERVICES LEASED LINE",
        "AIRTEL COMMERCIAL BROADBAND", "OFFICE WATER SUPPLY", "BESCOM ELECTRICITY", "TATA POWER",
        "AIRTEL BROADBAND", "JIO FIBER", "ACT FIBERNET", "BWSSB WATER", "GAIL GAS",
        "MUNICIPAL TAX", "BBMP PROPERTY TAX", "ADANI ELECTRICITY", "ELECTRICITY BILL",
        "WATER BILL", "IGL PIPED GAS", "MAHANAGAR GAS", "TATA PLAY DTH", "MOBILE RECHARGE",
        "COMMERCIAL LEASED LINE", "UTILITY BILL", "POWER BILL", "BROADBAND INTERNET",
    ],
    "Subscription": [
        "AWS CLOUD SERVICES", "GOOGLE WORKSPACE", "MICROSOFT AZURE", "ZOHO ONE SUITE",
        "SLACK TECHNOLOGIES", "TALLY SOFTWARE AMC", "GITHUB ENTERPRISE", "ZOOM VIDEO COMM",
        "ATLASSIAN JIRA", "GODADDY DOMAIN RENEWAL", "FIGMA ENTERPRISE", "MICROSOFT 365",
        "CHATGPT PLUS", "OPENAI SUBSCRIPTION", "CLAUDE PRO", "GOOGLE ONE STORAGE",
        "NOTION PLUS", "CANVA PRO", "LINKEDIN PREMIUM", "ADOBE CREATIVE CLOUD",
        "DIGITALOCEAN CLOUD", "CLOUDFLARE SERVICES", "POSTMAN API PLATFORM",
    ],
    "Travel": [
        "IRCTC", "UBER TRIP", "OLA RIDE", "MAKEMYTRIP", "GOIBIBO", "INDIGO AIRLINES",
        "AIR INDIA", "VISTARA", "RED BUS", "RAPIDO", "CLEARTRIP", "YATRA", "EASEMYTRIP",
        "AKASA AIR", "SPICEJET", "NAMMA YATRI", "BLUSMART", "METRO RAIL", "DMRC METRO",
        "FASTAG RECHARGE", "NHAI TOLL PLAZA", "AIRPORT PARKING", "BOOKING.COM",
        "CORPORATE FLIGHT BOOKING", "BUSINESS HOTEL STAY", "TAXI REIMBURSEMENT",
    ],
    "Food": [
        "SWIGGY", "ZOMATO", "DOMINOS PIZZA", "MCDONALDS", "KFC", "STARBUCKS",
        "CAFE COFFEE DAY", "BARBEQUE NATION", "HALDIRAM", "EATCLUB", "PIZZA HUT",
        "SUBWAY", "BURGER KING", "CHAI POINT", "CHAAYOS", "DARSHINI CAFE", "LOCAL RESTAURANT",
        "OFFICE PANTRY SNACKS", "TEA AND COFFEE VENDOR", "CLIENT LUNCH EXPENSE", "CATERING FOOD",
    ],
    "Groceries": [
        "BIGBASKET", "BLINKIT", "ZEPTO", "DMART", "RELIANCE FRESH", "MORE SUPERMARKET",
        "SPENCERS", "JIOMART GROCERY", "LOCAL KIRANA", "RELIANCE SMART", "INSTAMART",
        "AMUL MILK", "MOTHER DAIRY", "NANDINI MILK", "COUNTRY DELIGHT", "PANTRY MILK SUPPLIES",
    ],
    "Fuel": [
        "INDIAN OIL", "BPCL PETROL", "HP PETROL PUMP", "RELIANCE PETROL", "SHELL FUEL",
        "IOCL PETROL PUMP", "BHARAT PETROLEUM", "HINDUSTAN PETROLEUM", "NAYARA ENERGY",
        "DIESEL GENERATOR FUEL", "COMPANY VEHICLE FUEL", "PETROL BUNK", "EV CHARGING STATION",
    ],
    "Shopping": [
        "AMAZON", "FLIPKART", "MYNTRA", "AJIO", "CROMA", "IKEA", "DECATHLON",
        "RELIANCE DIGITAL", "VIJAY SALES", "OFFICE FURNITURE MART", "HARDWARE SUPPLIES",
        "COMPUTER PERIPHERALS", "STATIONERY AND PRINTING", "MALL RETAIL",
    ],
    "Entertainment": [
        "PVR CINEMAS", "INOX", "BOOKMYSHOW", "SPOTIFY PREMIUM", "NETFLIX", "HOTSTAR",
        "TEAM OFFSITE RESORT", "ANNUAL CELEBRATION EVENT", "ENTERTAINMENT PASS",
    ],
    "Healthcare": [
        "APOLLO PHARMACY", "PHARMEASY", "1MG", "FORTIS HOSPITAL", "MANIPAL HOSPITAL",
        "EMPLOYEE HEALTH CHECKUP", "FIRST AID SUPPLIES", "DIAGNOSTIC LAB", "CLINIC CONSULTATION",
    ],
    "Education": [
        "COURSERA", "UDEMY", "NPTEL", "CORPORATE TRAINING WORKSHOP", "SKILL CERTIFICATION",
        "EMPLOYEE LEARNING PROGRAM", "EXAM FEE", "COLLEGE FEE",
    ],
    "Investment": [
        "GROWW", "ZERODHA", "UPSTOX", "TREASURY MUTUAL FUND", "LIQUID FUND PURCHASE",
        "FIXED DEPOSIT CORPORATE", "COMMERCIAL PAPER", "GOVERNMENT SECURITIES", "PPF DEPOSIT",
    ],
    "Insurance": [
        "GROUP HEALTH INSURANCE", "COMMERCIAL ASSET INSURANCE", "FIRE BURGLARY INSURANCE",
        "DIRECTORS OFFICERS D AND O", "TRANSIT INSURANCE", "LIC PREMIUM", "HDFC LIFE",
        "ICICI LOMBARD", "STAR HEALTH", "POLICYBAZAAR", "VEHICLE INSURANCE",
    ],
    "Transfer": [
        "NEFT CR", "IMPS SENT", "RTGS CORPORATE TRANSFER", "INTER BANK FUNDS TRANSFER",
        "TO SAVINGS", "INTERNAL ACCOUNT TRANSFER", "DIRECTOR DRAWINGS", "PARTNER CAPITAL PAYOUT",
        "AKSHAY M", "CHANDAN M", "KUMAR J V", "VIJAYA", "ROHIT SHARMA", "ANANYA RAO",
        "RAHUL KUMAR", "PRIYA NAIR", "ARJUN MENON", "LOKESHAP", "BHARATHI P", "MR LAKS",
    ],
    "ATM/Cash": [
        "ATM WDL", "CASH WITHDRAWAL", "PETTY CASH DISPENSE", "NFS ATM", "SBI ATM WDL",
        "CDM CASH DEPOSIT", "VAULT CASH TRANSFER", "BRANCH CASH WITHDRAWAL",
    ],
    "Other": [
        "CORPORATE BANK AMC", "FOREX MARKUP FEE", "OUTWARD CHEQUE RETURN", "RTGS CHARGES",
        "NEFT SERVICE CHARGES", "MINIMUM BALANCE PENALTY", "GST ON BANK CHARGES",
        "CONSOLIDATED CHARGES", "MISC DEBIT", "MISC CREDIT",
    ],
}

UPI_PREFIXES = ["UPI/", "UPI-DR/", "UPI-CR/", "PAYTM/", "GPAY/", "PHONEPE/", "BHIM/"]
CHANNEL = ["POS", "ECOM", "NACH", "ACH", "IMPS", "NEFT", "RTGS", "UPI", "CMS", "CORP", "CHQ"]
BANKS_IFSC = ["SBIN", "NESF", "UTIB", "YESB", "BARB", "HDFC", "ICIC", "KKBK", "PUNB", "CNRB", "UBIN", "IDFB"]


def _vary(text: str) -> str:
    extras = [
        "",
        " REF " + str(random.randint(100000, 999999)),
        " UTR " + str(random.randint(10**11, 10**12 - 1)),
        " INV-" + str(random.randint(1000, 9999)),
        " PO-" + str(random.randint(1000, 9999)),
        " " + random.choice(CHANNEL),
        " IND",
        " BANGALORE",
        " MUMBAI",
        " DELHI",
        " CHENNAI",
    ]
    return f"{text}{random.choice(extras)}".strip()


def generate_samples(per_class: int = 300) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    for category in CATEGORIES:
        merchants = MERCHANTS.get(category, ["MISC TRANSACTIONS"])
        produced = 0
        while produced < per_class:
            merchant = random.choice(merchants)
            rrn = str(random.randint(10**11, 10**12 - 1))
            bank = random.choice(BANKS_IFSC)
            vpa_handle = re.sub(r"[^a-z0-9]", "", merchant.lower())[:8] + str(random.randint(10, 99))

            style = random.choice([
                "corp_cms",
                "neft_rtgs",
                "ach_nach",
                "upi_npci_dr",
                "upi_npci_cr",
                "plain",
            ])

            if style == "corp_cms":
                text = f"CMS/{bank}/{merchant}/INV-{random.randint(1000, 9999)}"
            elif style == "neft_rtgs":
                text = f"{random.choice(['NEFT', 'RTGS', 'IMPS'])}-{bank}-{merchant}-SETTLEMENT"
            elif style == "ach_nach":
                text = f"ACHDr {bank}{random.randint(10000, 99999)} {merchant}"
            elif style == "upi_npci_dr":
                text = f"UPI/DR/{rrn}/{merchant}/{bank}/{vpa_handle}/Payment"
            elif style == "upi_npci_cr":
                text = f"UPI/CR/{rrn}/{merchant}/{bank}/{vpa_handle}/Payment"
            else:
                text = merchant

            texts.append(_vary(text))
            labels.append(category)
            produced += 1

    return texts, labels


def generate_bank_headers(n: int = 500) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    templates = [
        "{bank}\nAccount Statement\nCustomer Name: {name}\nAccount No: {acct}\nBranch: {city}",
        "Statement of Account\n{bank}\nA/c Holder: {name} (Current Account)\nA/c Number XXXX{acct4}\nIFSC {ifsc}",
        "{bank} - Corporate eStatement\nCompany Name: {name}\nAccount {acct}\nPeriod {period}",
        "Welcome to {bank}\nCurrent & Business Account\n{name}\nAccount Number: {acct}",
    ]
    names = [
        "CHANDAN M", "ACME SOLUTIONS PVT LTD", "TECHNOVA INDIA ENTERPRISES",
        "APEX GLOBAL LOGISTICS", "SMARTWORKS TRADING CO", "ROHIT SHARMA", "ANANYA RAO"
    ]
    cities = ["Bengaluru", "Mumbai", "Chennai", "Hyderabad", "Pune", "Delhi"]
    for bank in INDIAN_BANKS:
        for _ in range(max(1, n // len(INDIAN_BANKS))):
            acct = str(random.randint(10**11, 10**12 - 1))
            text = random.choice(templates).format(
                bank=bank,
                name=random.choice(names),
                acct=acct,
                acct4=acct[-4:],
                city=random.choice(cities),
                ifsc=bank[:4].upper().replace(" ", "") + "0001234",
                period="01-01-2026 to 30-06-2026",
            )
            texts.append(text)
            labels.append(bank)
    return texts, labels
