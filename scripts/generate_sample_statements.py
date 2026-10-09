"""Generate realistic sample bank statements for Service-based & Product-based companies (3, 6, 12 months) and individuals."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
import random

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "sample_statements"


# ---------------------------------------------------------------------------
# Seed transaction generators
# ---------------------------------------------------------------------------

SERVICE_COMPANY_TXNS = [
    # Inflows (Client receipts, SaaS payouts, retainers)
    ("NEFT CR - INFOSYS B2B CLOUD RETAINER INVOICE INV-2026-081", 450000.0, "cr"),
    ("RTGS CR - ACCENTURE CONSULTING SERVICES SETTLEMENT", 620000.0, "cr"),
    ("ACH CR - STRIPE PAYMENTS SAAS REVENUE DISBURSEMENT", 380000.0, "cr"),
    ("UPI/RAZORPAY SOFTWARE REVENUE/771092", 215000.0, "cr"),
    ("INWARD REMITTANCE CR - TECHCORP US SOFTWARE CONTRACT", 750000.0, "cr"),
    ("NEFT CR - TATA CONSULTANCY SERVICES VENDOR SETTLEMENT", 480000.0, "cr"),
    ("IMPS CR - CLIENT ADVANCE REVENUE MILESTONE 2", 290000.0, "cr"),
    ("RAZORPAY GATEWAY PAYOUT - SAAS MRR SUBSCRIPTIONS", 340000.0, "cr"),
    # Cloud Infrastructure & Hosting
    ("AMAZON WEB SERVICES AWS CLOUD HOSTING BILL", 135000.0, "dr"),
    ("MICROSOFT AZURE ENTERPRISE CLOUD SERVICES", 68000.0, "dr"),
    ("GOOGLE CLOUD PLATFORM GCP INVOICE US-EAST", 42000.0, "dr"),
    # SaaS & Dev Tools Subscriptions
    ("GITHUB ENTERPRISE DEVELOPER SEATS SUBSCRIPTION", 22400.0, "dr"),
    ("ATLASSIAN JIRA CONFLUENCE ENTERPRISE CLOUD", 18500.0, "dr"),
    ("SLACK TECHNOLOGIES TEAM COLLABORATION PLAN", 14200.0, "dr"),
    ("FIGMA PROFESSIONAL DESIGN SEATS SUBSCRIPTION", 9800.0, "dr"),
    ("GOOGLE WORKSPACE BUSINESS SUITE 50 USERS", 16500.0, "dr"),
    ("ZOOM VIDEO COMM CORPORATE ENTERPRISE", 4200.0, "dr"),
    # HR & Payroll
    ("BULK SALARY DISBURSEMENT - CORE ENGINEERING STAFF", 680000.0, "dr"),
    ("DIRECTOR REMUNERATION - ANISH VERMA", 150000.0, "dr"),
    ("CONTRACTOR STIPEND PAYROLL - UI UX DESIGN BATCH", 85000.0, "dr"),
    ("EMPLOYEE MEDICAL INSURANCE NACH - STAR HEALTH", 36000.0, "dr"),
    ("EPFO ELECTRONIC CHALLAN DISBURSEMENT", 48000.0, "dr"),
    # Office Lease & Utilities
    ("COMMERCIAL OFFICE LEASE - WEWORK GALAXY BANGALORE", 185000.0, "dr"),
    ("AIRTEL COMMERCIAL LEASED LINE INTERNET 1GBPS", 12499.0, "dr"),
    ("BESCOM COMMERCIAL ELECTRICITY BILLING", 18500.0, "dr"),
    # Statutory Taxes & Professional Advisory
    ("TDS CHALLAN 281 - SECTION 194J PROFESSIONAL FEES", 42500.0, "dr"),
    ("GSTPMT-06 CHALLAN PMT - GSTR 3B TAX LIABILITY", 125000.0, "dr"),
    ("CHARTERED ACCOUNTANT FEES - STATUTORY TAX AUDIT", 35000.0, "dr"),
    ("LEGAL RETAINER FEE - SHARMA & ASSOCIATES ADVOCATES", 25000.0, "dr"),
    # Miscellaneous Office & Team
    ("SWIGGY TEAM ALL-HANDS LUNCH ORDER BANGALORE", 4850.0, "dr"),
    ("UBER TRIP CORPORATE CAB FARES FOR CLIENT VISITS", 3200.0, "dr"),
]

PRODUCT_COMPANY_TXNS = [
    # Inflows (Wholesale distributors, retail buyers, e-commerce)
    ("RTGS CR - CROMPTON GREAVES B2B WHOLESALE INVOICE PMT", 850000.0, "cr"),
    ("NEFT CR - RELIANCE DIGITAL COMMERCIAL PURCHASE CR", 680000.0, "cr"),
    ("RAZORPAY MERCHANT DISBURSEMENT - D2C ECOMMERCE SALES", 390000.0, "cr"),
    ("AMAZON SELLER PAYOUT - HARDWARE ELECTRONICS PROCEEDS", 460000.0, "cr"),
    ("FLIPKART INDIA PVT LTD - VENDOR SETTLEMENT CR", 410000.0, "cr"),
    ("EXPORT RECEIVABLES - DUBAI ELECTRONICS TRADING LLC", 950000.0, "cr"),
    ("NEFT CR - VIJAY SALES B2B INVOICE CLEARING", 520000.0, "cr"),
    # Raw Materials & Components Procurement
    ("RTGS DR - VENDOR INVOICE - SHENGZHEN SEMICONDUCTOR PARTS", 380000.0, "dr"),
    ("NEFT DR - RAW MATERIAL PROCUREMENT - COPPER WIRE & COILS", 210000.0, "dr"),
    ("IMPS DR - PLASTIC INJECTION MOULDING DIES & CASINGS", 145000.0, "dr"),
    ("RTGS DR - LI-ION BATTERY CELL SUPPLIERS GUJARAT", 295000.0, "dr"),
    ("NEFT DR - PACKAGING SUPPLIES - CORRUGATED BOXES BATCH", 65000.0, "dr"),
    ("PURCHASE ORDER SETTLEMENT - MICROCONTROLLER IC CHIPS", 185000.0, "dr"),
    ("VENDOR PAYMENT - SMT COMPONENT SOURCING DELHI", 160000.0, "dr"),
    # Logistics & Freight
    ("DELHIVERY LOGISTICS FREIGHT DISBURSEMENT", 54000.0, "dr"),
    ("BLUEDART EXPRESS B2B COURIER CHARGES", 32500.0, "dr"),
    ("VRL LOGISTICS FULL TRUCK LOAD FTL DISBURSEMENT", 68000.0, "dr"),
    ("TCI FREIGHT INTERSTATE CARGO TRANSPORT", 45000.0, "dr"),
    ("CUSTOMS CLEARING AGENT - IMPORT DUTY CLEARANCE", 125000.0, "dr"),
    # Manufacturing Plant & Working Capital
    ("INDUSTRIAL POWER BESCOM HT SUBSTATION TARIFF", 88500.0, "dr"),
    ("FACTORY WAREHOUSE LEASE - PEENYA INDUSTRIAL AREA", 160000.0, "dr"),
    ("SBI WORKING CAPITAL LOAN EMI DEBIT", 95000.0, "dr"),
    ("MACHINERY FINANCE EMI - SMT PICK AND PLACE LINE", 72000.0, "dr"),
    ("FACTORY LABOUR CONTRACTOR WAGES BATCH", 280000.0, "dr"),
    ("CORE STAFF MONTHLY SALARY DISBURSEMENT", 420000.0, "dr"),
    ("GST 3B CGST SGST INVENTORY TAX PAYMENT", 185000.0, "dr"),
    ("STATUTORY AUDIT FEES - CHARTERED ACCOUNTANTS", 40000.0, "dr"),
    ("DIESEL GENERATOR POWER BACKUP FUEL PURCHASE", 18500.0, "dr"),
]

PERSONAL_HDFC = [
    ("UPI/SWIGGY/8821 POS IND", 420.0, "dr"),
    ("NEFT SALARY ACME PVT LTD", 55000.0, "cr"),
    ("UPI/IRCTC/4412", 1850.0, "dr"),
    ("POS INDIAN OIL BANGALORE", 2100.0, "dr"),
    ("UPI/AMAZON/9001", 2499.0, "dr"),
    ("UPI/NETFLIX.COM/112", 199.0, "dr"),
    ("ATM WDL NFS", 5000.0, "dr"),
    ("UPI/BIGBASKET/330", 1760.0, "dr"),
]

PERSONAL_SBI = [
    ("UPI/ZOMATO/221", 560.0, "dr"),
    ("IMPS SENT FAMILY", 8000.0, "dr"),
    ("UPI/PVR CINEMAS/88", 780.0, "dr"),
    ("LIC PREMIUM NACH", 2450.0, "dr"),
    ("UPI/UBER TRIP/19", 340.0, "dr"),
    ("INTEREST CREDIT", 210.0, "cr"),
]

PERSONAL_ICICI = [
    ("UPI/BLINKIT/77", 890.0, "dr"),
    ("HOUSE RENT UPI LANDLORD", 15000.0, "dr"),
    ("GROWW SIP MUTUAL FUND", 3000.0, "dr"),
    ("UPI/APOLLO PHARMACY/4", 640.0, "dr"),
    ("UPI/JIO FIBER/91", 999.0, "dr"),
]


# ---------------------------------------------------------------------------
# Timeline-based generation helper
# ---------------------------------------------------------------------------

def generate_chronological_txns(
    start: date,
    end: date,
    pool: list[tuple[str, float, str]],
    initial_balance: float = 850000.0,
    interval_days: int = 3,
) -> list[dict]:
    """Generate chronological bank ledger rows between start and end dates."""
    data = []
    current_date = start
    balance = initial_balance
    idx = 0

    while current_date <= end:
        desc, base_amount, side = pool[idx % len(pool)]
        # Add slight realistic amount variation (+- 5%)
        variance = 1.0 + ((idx % 7) - 3) * 0.015
        amount = round(base_amount * variance, 2)

        debit = amount if side == "dr" else 0.0
        credit = amount if side == "cr" else 0.0
        balance = round(balance - debit + credit, 2)

        data.append(
            {
                "Date": current_date.strftime("%d-%m-%Y"),
                "Narration": desc,
                "Withdrawal(Dr)": debit or "",
                "Deposit(Cr)": credit or "",
                "Balance": balance,
            }
        )

        current_date += timedelta(days=interval_days)
        idx += 1

    return data


# ---------------------------------------------------------------------------
# File Writers: CSV, Excel, PDF
# ---------------------------------------------------------------------------

def write_csv(path: Path, bank: str, holder: str, acct: str, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"{bank}",
        f"Current Account Statement",
        f"Account Holder: {holder}",
        f"Account Number: {acct}",
        f"Branch: Bengaluru Main Branch",
        f"IFSC Code: {bank[:4].upper()}0001842",
        "",
    ]
    df = pd.DataFrame(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("\n".join(header) + "\n")
        df.to_csv(handle, index=False)


def write_excel(path: Path, bank: str, holder: str, acct: str, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    cover = pd.DataFrame(
        {
            "Field": ["Bank", "Account Holder", "Account Number", "Branch", "Statement Type"],
            "Value": [bank, holder, acct, "Bengaluru Main Branch", "Official Statement"],
        }
    )
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        cover.to_excel(writer, sheet_name="Account Details", index=False)
        df.to_excel(writer, sheet_name="Transactions", index=False)


def write_pdf(path: Path, bank: str, holder: str, acct: str, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    # A4: 595.27 x 841.89 points, margins 28pt left/right -> 539pt printable width
    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        leftMargin=28,
        rightMargin=28,
        topMargin=28,
        bottomMargin=28,
    )
    styles = getSampleStyleSheet()

    narration_style = ParagraphStyle(
        "NarrationStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7,
        leading=9,
        textColor=colors.HexColor("#0F172A"),
    )
    cell_style = ParagraphStyle(
        "CellStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor("#1E293B"),
    )
    header_style = ParagraphStyle(
        "HeaderStyle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10,
        textColor=colors.whitesmoke,
    )

    story = [
        Paragraph(f"<b>{bank}</b>", styles["Title"]),
        Paragraph("<b>Official Account Statement</b>", styles["Heading3"]),
        Paragraph(f"<b>Account Holder:</b> {holder} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Account No:</b> {acct}", styles["Normal"]),
        Paragraph(f"<b>Period:</b> {rows[0]['Date']} to {rows[-1]['Date']} &nbsp;&nbsp;|&nbsp;&nbsp; <b>Total Transactions:</b> {len(rows)}", styles["Normal"]),
        Spacer(1, 10),
    ]

    # Printable table with 539pt width: Date (58), Narration (245), Debit (75), Credit (75), Balance (86)
    table_data = [
        [
            Paragraph("Date", header_style),
            Paragraph("Narration / Description", header_style),
            Paragraph("Debit (Dr)", header_style),
            Paragraph("Credit (Cr)", header_style),
            Paragraph("Balance", header_style),
        ]
    ]

    for row in rows:
        dr_val = f"₹{row['Withdrawal(Dr)']}" if row["Withdrawal(Dr)"] else "-"
        cr_val = f"₹{row['Deposit(Cr)']}" if row["Deposit(Cr)"] else "-"
        bal_val = f"₹{row['Balance']}"

        table_data.append(
            [
                Paragraph(row["Date"], cell_style),
                Paragraph(row["Narration"], narration_style),
                Paragraph(dr_val, cell_style),
                Paragraph(cr_val, cell_style),
                Paragraph(bal_val, cell_style),
            ]
        )

    table = Table(table_data, colWidths=[58, 245, 75, 75, 86], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0F172A")),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 5),
                ("TOPPADDING", (0, 0), (-1, 0), 5),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.HexColor("#FFFFFF"), colors.HexColor("#F8FAFC")]),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 1), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 3),
            ]
        )
    )
    story.append(table)
    doc.build(story)


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    generated_manifest = []

    # 1. SERVICE-BASED COMPANY (Zenith Cloud Technologies Pvt Ltd - HDFC Bank)
    # 3 Months: Q1 2026 (Jan 1, 2026 -> Mar 31, 2026)
    svc_3m = generate_chronological_txns(date(2026, 1, 1), date(2026, 3, 31), SERVICE_COMPANY_TXNS, initial_balance=1250000.0, interval_days=2)
    write_csv(OUT / "Zenith_Cloud_Tech_Service_HDFC_3Months_Q1_2026.csv", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_3m)
    write_excel(OUT / "Zenith_Cloud_Tech_Service_HDFC_3Months_Q1_2026.xlsx", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_3m)
    write_pdf(OUT / "Zenith_Cloud_Tech_Service_HDFC_3Months_Q1_2026.pdf", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_3m)
    generated_manifest.append(("Zenith Cloud Tech (Service)", "3 Months (Q1 2026)", len(svc_3m)))

    # 6 Months: H1 2026 (Jan 1, 2026 -> Jun 30, 2026)
    svc_6m = generate_chronological_txns(date(2026, 1, 1), date(2026, 6, 30), SERVICE_COMPANY_TXNS, initial_balance=1500000.0, interval_days=3)
    write_csv(OUT / "Zenith_Cloud_Tech_Service_HDFC_6Months_H1_2026.csv", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_6m)
    write_excel(OUT / "Zenith_Cloud_Tech_Service_HDFC_6Months_H1_2026.xlsx", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_6m)
    write_pdf(OUT / "Zenith_Cloud_Tech_Service_HDFC_6Months_H1_2026.pdf", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_6m)
    generated_manifest.append(("Zenith Cloud Tech (Service)", "6 Months (H1 2026)", len(svc_6m)))

    # 12 Months: FY 2025-26 (Apr 1, 2025 -> Mar 31, 2026)
    svc_12m = generate_chronological_txns(date(2025, 4, 1), date(2026, 3, 31), SERVICE_COMPANY_TXNS, initial_balance=1800000.0, interval_days=3)
    write_csv(OUT / "Zenith_Cloud_Tech_Service_HDFC_12Months_FY2025_26.csv", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_12m)
    write_excel(OUT / "Zenith_Cloud_Tech_Service_HDFC_12Months_FY2025_26.xlsx", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_12m)
    write_pdf(OUT / "Zenith_Cloud_Tech_Service_HDFC_12Months_FY2025_26.pdf", "HDFC Bank", "ZENITH CLOUD TECHNOLOGIES PVT LTD", "50200084920194", svc_12m)
    generated_manifest.append(("Zenith Cloud Tech (Service)", "12 Months (FY2025-26)", len(svc_12m)))

    # 2. PRODUCT-BASED COMPANY (Nexis Hardware & Electronics Pvt Ltd - State Bank of India)
    # 3 Months: Q1 2026 (Jan 1, 2026 -> Mar 31, 2026)
    prod_3m = generate_chronological_txns(date(2026, 1, 1), date(2026, 3, 31), PRODUCT_COMPANY_TXNS, initial_balance=2200000.0, interval_days=2)
    write_csv(OUT / "Nexis_Hardware_Product_SBI_3Months_Q1_2026.csv", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_3m)
    write_excel(OUT / "Nexis_Hardware_Product_SBI_3Months_Q1_2026.xlsx", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_3m)
    write_pdf(OUT / "Nexis_Hardware_Product_SBI_3Months_Q1_2026.pdf", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_3m)
    generated_manifest.append(("Nexis Hardware (Product)", "3 Months (Q1 2026)", len(prod_3m)))

    # 6 Months: H1 2026 (Jan 1, 2026 -> Jun 30, 2026)
    prod_6m = generate_chronological_txns(date(2026, 1, 1), date(2026, 6, 30), PRODUCT_COMPANY_TXNS, initial_balance=2500000.0, interval_days=3)
    write_csv(OUT / "Nexis_Hardware_Product_SBI_6Months_H1_2026.csv", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_6m)
    write_excel(OUT / "Nexis_Hardware_Product_SBI_6Months_H1_2026.xlsx", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_6m)
    write_pdf(OUT / "Nexis_Hardware_Product_SBI_6Months_H1_2026.pdf", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_6m)
    generated_manifest.append(("Nexis Hardware (Product)", "6 Months (H1 2026)", len(prod_6m)))

    # 12 Months: FY 2025-26 (Apr 1, 2025 -> Mar 31, 2026)
    prod_12m = generate_chronological_txns(date(2025, 4, 1), date(2026, 3, 31), PRODUCT_COMPANY_TXNS, initial_balance=2800000.0, interval_days=3)
    write_csv(OUT / "Nexis_Hardware_Product_SBI_12Months_FY2025_26.csv", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_12m)
    write_excel(OUT / "Nexis_Hardware_Product_SBI_12Months_FY2025_26.xlsx", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_12m)
    write_pdf(OUT / "Nexis_Hardware_Product_SBI_12Months_FY2025_26.pdf", "State Bank of India", "NEXIS HARDWARE ELECTRONICS PVT LTD", "38920194821001", prod_12m)
    generated_manifest.append(("Nexis Hardware (Product)", "12 Months (FY2025-26)", len(prod_12m)))

    # 3. PERSONAL STATEMENTS (Kept for viva / test consistency)
    hdfc_p = generate_chronological_txns(date(2026, 1, 2), date(2026, 2, 28), PERSONAL_HDFC, initial_balance=85000.0, interval_days=4)
    write_csv(OUT / "HDFC_Chandan_JanFeb2026.csv", "HDFC Bank", "CHANDAN M", "50100123456789", hdfc_p)
    hdfc_p2 = generate_chronological_txns(date(2026, 3, 1), date(2026, 4, 30), PERSONAL_HDFC, initial_balance=92000.0, interval_days=4)
    write_pdf(OUT / "HDFC_Chandan_MarApr2026.pdf", "HDFC Bank", "CHANDAN M", "50100123456789", hdfc_p2)

    sbi_p = generate_chronological_txns(date(2026, 1, 5), date(2026, 3, 31), PERSONAL_SBI, initial_balance=75000.0, interval_days=4)
    write_pdf(OUT / "SBI_Ananya_Q1_2026.pdf", "State Bank of India", "ANANYA RAO", "38920111223344", sbi_p)

    icici_p = generate_chronological_txns(date(2026, 1, 5), date(2026, 3, 25), PERSONAL_ICICI, initial_balance=64000.0, interval_days=4)
    write_csv(OUT / "ICICI_Rohit_2026.csv", "ICICI Bank", "ROHIT SHARMA", "62481000998877", icici_p)
    write_excel(OUT / "ICICI_Rohit_2026.xlsx", "ICICI Bank", "ROHIT SHARMA", "62481000998877", icici_p)

    print("\n================ SAMPLE STATEMENTS GENERATION REPORT ================")
    for org, span, count in generated_manifest:
        print(f"[OK] {org:<32} | {span:<22} | {count:>3} transactions (CSV, PDF, XLSX)")
    print(f"[OK] Output directory: {OUT}")
    print("======================================================================\n")


if __name__ == "__main__":
    main()
