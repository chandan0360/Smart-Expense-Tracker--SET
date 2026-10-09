"""Parse bank statements from PDF, CSV, Excel and text into transaction rows."""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import pdfplumber
from dateutil import parser as date_parser

DATE_KEYS = ("date", "txn date", "transaction date", "value date", "posted date", "tran date", "trans date", "posting date", "txn_date", "valuedate")
DESC_KEYS = (
    "description",
    "narration",
    "particulars",
    "details",
    "remarks",
    "transaction remarks",
    "narration/description",
    "transaction details",
    "txn desc",
    "narrative",
    "payee",
    "beneficiary",
    "transaction reference",
    "trans reference",
    "txn reference",
    "transaction_reference",
    "reference",
    "txn particulars",
)
DEBIT_KEYS = ("debit", "withdrawal", "withdrawals", "dr", "debit amount", "withdrawal amt", "withdrawal(dr)", "withdrawal amount (inr)", "dr amount", "debit (inr)", "debit (rs)", "withdrawal (rs)")
CREDIT_KEYS = ("credit", "deposit", "deposits", "cr", "credit amount", "deposit amt", "deposit(cr)", "deposit amount (inr)", "cr amount", "credit (inr)", "credit (rs)", "deposit (rs)")
AMOUNT_KEYS = ("amount", "txn amount", "transaction amount", "amount (inr)", "amount (rs)")
TYPE_KEYS = ("type", "dr/cr", "cr/dr", "txn type", "indicator")
BALANCE_KEYS = ("balance", "closing balance", "available balance", "running balance", "balance (inr)", "account balance", "closing bal", "net balance")
CURRENCY_KEYS = ("currency", "curr", "ccy", "txn currency", "trans currency", "transaction currency")

# Standard Forex Baseline Exchange Rates to INR (Indian Rupees)
CURRENCY_RATES_TO_INR: dict[str, tuple[str, float]] = {
    "USD": ("USD", 83.50),
    "$": ("USD", 83.50),
    "EUR": ("EUR", 91.00),
    "€": ("EUR", 91.00),
    "GBP": ("GBP", 106.00),
    "£": ("GBP", 106.00),
    "AED": ("AED", 22.75),
    "SGD": ("SGD", 62.20),
    "CAD": ("CAD", 61.30),
    "AUD": ("AUD", 55.40),
    "JPY": ("JPY", 0.55),
    "CNY": ("CNY", 11.50),
    "SAR": ("SAR", 22.25),
    "QAR": ("QAR", 22.90),
}


class StatementParseError(ValueError):
    pass


@dataclass
class ParsedTransaction:
    txn_date: date | None
    description: str
    debit: float
    credit: float
    balance: float | None
    raw: dict = field(default_factory=dict)


@dataclass
class ParsedStatement:
    header_text: str
    transactions: list[ParsedTransaction]
    source_name: str


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (value or "").lower()).strip()


def _match_col(columns: list[str], keys: tuple[str, ...]) -> str | None:
    normalized = {_norm(c): c for c in columns}
    # 1. Exact match pass
    for key in keys:
        want = _norm(key)
        for n, original in normalized.items():
            if want == n:
                return original
    # 2. Word boundary or long phrase match pass
    for key in keys:
        want = _norm(key)
        for n, original in normalized.items():
            words = n.split()
            if want in words or (len(want) >= 5 and want in n):
                return original
    return None


def _parse_amount(value) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return 0.0
    text = str(value).strip()
    if not text or text in {"-", "nan", "None"}:
        return 0.0
    negative = text.startswith("(") and text.endswith(")")
    text = text.replace(",", "").replace("₹", "").replace("Rs.", "").replace("INR", "").replace("/-", "")
    text = re.sub(r"[^\d.\-]", "", text)
    if not text:
        return 0.0
    try:
        amount = float(text)
    except ValueError:
        return 0.0
    return -amount if negative else amount


def _parse_date(value) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for dayfirst in (True, False):
        try:
            return date_parser.parse(text, dayfirst=dayfirst, fuzzy=True).date()
        except (ValueError, OverflowError, TypeError):
            continue
    return None


def dataframe_to_transactions(df: pd.DataFrame) -> list[ParsedTransaction]:
    if df.empty:
        raise StatementParseError("No rows found in the statement.")
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    cols = list(df.columns)
    date_col = _match_col(cols, DATE_KEYS)
    desc_col = _match_col(cols, DESC_KEYS)
    debit_col = _match_col(cols, DEBIT_KEYS)
    credit_col = _match_col(cols, CREDIT_KEYS)
    amount_col = _match_col(cols, AMOUNT_KEYS)
    type_col = _match_col(cols, TYPE_KEYS)
    balance_col = _match_col(cols, BALANCE_KEYS)

    currency_col = _match_col(cols, CURRENCY_KEYS)

    if not desc_col and len(cols) >= 2:
        desc_col = cols[1]
    if not date_col:
        date_col = cols[0]

    rows: list[ParsedTransaction] = []
    for _, record in df.iterrows():
        description = str(record.get(desc_col, "")).strip() if desc_col else ""
        if not description or description.lower() in {"nan", "none"}:
            continue
        debit = _parse_amount(record.get(debit_col)) if debit_col else 0.0
        credit = _parse_amount(record.get(credit_col)) if credit_col else 0.0
        if debit == 0 and credit == 0 and amount_col:
            amount = _parse_amount(record.get(amount_col))
            kind = str(record.get(type_col, "")).upper() if type_col else ""
            if "CR" in kind or amount < 0:
                credit = abs(amount)
            else:
                debit = abs(amount)

        if abs(debit) == 0.0 and abs(credit) == 0.0:
            continue

        desc_lower = description.lower()
        if any(term in desc_lower for term in ("opening balance", "closing balance", "balance b/f", "balance c/f", "brought forward", "carried forward", "bualllance")):
            continue

        txn_date = _parse_date(record.get(date_col)) if date_col else None
        balance = _parse_amount(record.get(balance_col)) if balance_col else None

        # Multi-Currency Detection & Auto-Conversion to INR
        forex_code = None
        forex_rate = 1.0

        if currency_col:
            raw_curr = str(record.get(currency_col, "")).strip().upper()
            if raw_curr in CURRENCY_RATES_TO_INR:
                forex_code, forex_rate = CURRENCY_RATES_TO_INR[raw_curr]

        if not forex_code:
            for symbol, (c_code, rate) in CURRENCY_RATES_TO_INR.items():
                pattern = rf"(?:\b{re.escape(symbol)}\b|{re.escape(symbol)})\s*[\d,]+(?:\.\d+)?"
                if re.search(pattern, description, re.IGNORECASE):
                    forex_code = c_code
                    forex_rate = rate
                    break

        if forex_code and forex_rate != 1.0:
            orig_debit = debit
            orig_credit = credit
            if debit > 0:
                debit = round(debit * forex_rate, 2)
                description += f" [Converted: {forex_code} {orig_debit:,.2f} @ ₹{forex_rate:.2f}]"
            elif credit > 0:
                credit = round(credit * forex_rate, 2)
                description += f" [Converted: {forex_code} {orig_credit:,.2f} @ ₹{forex_rate:.2f}]"
            if balance is not None:
                balance = round(balance * forex_rate, 2)

        rows.append(
            ParsedTransaction(
                txn_date=txn_date,
                description=description,
                debit=abs(debit),
                credit=abs(credit),
                balance=balance if balance_col else None,
                raw={str(k): ("" if pd.isna(v) else str(v)) for k, v in record.items()},
            )
        )
    if not rows:
        raise StatementParseError("Could not read any transactions from this file.")
    return rows


def parse_tabular_file(path: Path) -> ParsedStatement:
    suffix = path.suffix.lower()
    header_bits = [path.stem]
    if suffix in {".csv", ".tsv", ".txt"}:
        sep = "\t" if suffix == ".tsv" else ","
        raw = path.read_bytes()
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                text = raw.decode(encoding)
                break
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        data_start = 0
        for i, line in enumerate(lines[:15]):
            if line.count(",") >= 2 or line.count("\t") >= 2 or line.count(";") >= 2 or line.count("|") >= 2:
                data_start = i
                break
            if line.strip():
                header_bits.append(line.strip())
        payload = "\n".join(lines[data_start:])
        df = pd.read_csv(io.StringIO(payload), sep=sep if suffix != ".csv" else None, engine="python")
    else:
        xl = pd.ExcelFile(path)
        header_bits.append("Excel statement")
        header_bits.extend(xl.sheet_names)
        best_df = None
        for sheet in xl.sheet_names:
            candidate = xl.parse(sheet)
            header_bits.append(candidate.head(8).to_csv(index=False))
            if best_df is None or len(candidate.index) > len(best_df.index):
                best_df = candidate
        df = best_df if best_df is not None else pd.DataFrame()
    return ParsedStatement(
        header_text="\n".join(header_bits),
        transactions=dataframe_to_transactions(df),
        source_name=path.name,
    )


def _pdf_header_text(pdf) -> str:
    chunks = []
    for page in pdf.pages[:10]:
        chunks.append(page.extract_text() or "")
    return "\n".join(chunks)


def _is_txn_table(header: list[str | None]) -> bool:
    if not header or len(header) < 2:
        return False
    cols = [_norm(str(c or "")) for c in header if c]
    has_date = any(any(k == c or k in c for k in DATE_KEYS) for c in cols)
    has_money = any(any(k == c or k in c for k in DEBIT_KEYS + CREDIT_KEYS + AMOUNT_KEYS) for c in cols)
    return has_date and has_money


def _tables_to_df(tables: list[list[list[str | None]]]) -> pd.DataFrame | None:
    """Concatenate tables across multi-page statements into a single unified DataFrame."""
    if not tables:
        return None

    # Identify candidate transaction tables
    candidate_tables = [t for t in tables if t and len(t) >= 2 and _is_txn_table(t[0])]
    if not candidate_tables:
        # Fallback to tables with at least 3 columns and at least 2 rows
        candidate_tables = [t for t in tables if t and len(t) >= 2 and len(t[0]) >= 3]
    if not candidate_tables:
        candidate_tables = tables

    primary_header: list[str] | None = None
    all_rows: list[list[str]] = []

    for table in candidate_tables:
        if not table or len(table) < 1:
            continue
        first_row = [str(c or "").strip() for c in table[0]]
        
        # Check if first row is a header
        if _is_txn_table(first_row):
            if primary_header is None:
                primary_header = first_row
            body = table[1:]
        elif primary_header is not None and len(first_row) == len(primary_header):
            body = table
        else:
            if primary_header is None and len([h for h in first_row if h]) >= 2:
                primary_header = first_row
                body = table[1:]
            else:
                body = table

        for r in body:
            row_str = [str(c or "").strip() for c in r]
            if any(cell for cell in row_str):
                if primary_header:
                    if len(row_str) == len(primary_header):
                        all_rows.append(row_str)
                    elif len(row_str) < len(primary_header):
                        row_str.extend([""] * (len(primary_header) - len(row_str)))
                        all_rows.append(row_str)
                    else:
                        all_rows.append(row_str[:len(primary_header)])
                else:
                    all_rows.append(row_str)

    if primary_header and all_rows:
        seen: dict[str, int] = {}
        unique_header: list[str] = []
        for h in primary_header:
            name = h if h else "unnamed"
            if name in seen:
                seen[name] += 1
                unique_header.append(f"{name}_{seen[name]}")
            else:
                seen[name] = 0
                unique_header.append(name)
        return pd.DataFrame(all_rows, columns=unique_header)

    # Fallback to largest single table
    best = None
    for table in tables:
        if not table or len(table) < 2:
            continue
        header = [str(c or "").strip() for c in table[0]]
        if len([h for h in header if h]) < 2:
            continue
        body = table[1:]
        df = pd.DataFrame(body, columns=header)
        if best is None or len(df) > len(best):
            best = df
    return best


def parse_pdf(path: Path, password: str | None = None) -> ParsedStatement:
    tables: list[list[list[str | None]]] = []
    try:
        with pdfplumber.open(path, password=password) as pdf:
            header_text = _pdf_header_text(pdf)
            for page in pdf.pages:
                for table in page.extract_tables() or []:
                    tables.append(table)
    except Exception as exc:
        err_str = str(exc).lower()
        if "password" in err_str or "encrypted" in err_str or "authenticate" in err_str or "incorrect" in err_str:
            raise StatementParseError(
                "This bank statement PDF is password-protected. Please enter the password (e.g. PAN or DOB) in the upload form."
            )
        raise

    df = _tables_to_df(tables)
    if df is None or df.empty:
        raise StatementParseError(
            "No transaction table found in this PDF. Try CSV/Excel or a clearer statement scan."
        )
    return ParsedStatement(
        header_text=header_text or path.stem,
        transactions=dataframe_to_transactions(df),
        source_name=path.name,
    )


def parse_statement(path: Path, password: str | None = None) -> ParsedStatement:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return parse_pdf(path, password=password)
    if suffix in {".csv", ".tsv", ".txt", ".xlsx", ".xls"}:
        parsed = parse_tabular_file(path)
        if suffix in {".csv", ".txt"} and "bank" not in parsed.header_text.lower():
            # prepend filename cues for identity model
            parsed.header_text = f"{path.stem}\n{parsed.header_text}"
        return parsed
    raise StatementParseError(f"Unsupported file type: {suffix}")
