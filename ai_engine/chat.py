"""SET conversational engine — multi-turn chat over this project's ledger."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from datetime import date
from decimal import Decimal
from typing import Any

from django.conf import settings
from tracker.models import BankAccount, StatementFile, Transaction

from ai_engine.categories import CATEGORIES
from ai_engine.memory import SessionMemoryBuffer
from ai_engine.rag import get_user_vector_store


MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

BANK_ALIASES = {
    "hdfc": "HDFC",
    "icici": "ICICI",
    "sbi": "State Bank",
    "state bank": "State Bank",
    "axis": "Axis",
    "kotak": "Kotak",
    "pnb": "Punjab National",
    "punjab national": "Punjab National",
    "canara": "Canara",
    "baroda": "Baroda",
    "bob": "Baroda",
    "union": "Union Bank",
    "yes bank": "Yes Bank",
    "idfc": "IDFC",
    "indusind": "IndusInd",
    "federal": "Federal",
    "indian bank": "Indian Bank",
    "bank of india": "Bank of India",
}

STOP_WORDS = {
    "how", "much", "did", "spend", "spent", "what", "show", "tell", "about", "total",
    "were", "was", "my", "the", "for", "and", "any", "our", "are", "can", "you",
    "which", "took", "place", "between", "these", "data", "date", "dates", "amount",
    "transferred", "transfer", "those", "that", "this", "please", "would", "like",
    "look", "into", "check", "give", "me", "set", "expense", "tracker", "hello",
    "hey", "hi", "thanks", "thank", "yes", "yeah", "okay", "ok", "sure", "from",
    "with", "have", "has", "had", "been", "your", "yourself", "who", "when",
    "where", "why", "does", "done", "want", "need", "help", "ask", "just", "also",
    "next", "last", "first", "all", "some", "more", "list", "them", "then", "than",
    "out", "in", "on", "to", "of", "or", "if", "is", "it", "its", "a", "an",
    "rupees", "rupee", "inr", "rs", "debit", "credit", "transaction", "transactions",
    "account", "accounts", "bank", "statement", "statements", "cash", "flow",
    "summary", "overview", "details", "info", "information", "report", "today",
    "yesterday", "month", "year", "week", "range", "period", "during", "until",
    "till", "since", "after", "before", "under", "over", "above", "below",
    "greater", "less", "paid", "pay", "got", "received", "there", "here",
    "something", "anything", "everything", "nothing", "could", "should",
    "will", "shall", "do", "get", "got", "see", "saw", "find", "found",
}

FOLLOWUP_RE = re.compile(
    r"^\s*(yes|yeah|yep|yup|ok|okay|sure|please|go ahead|do it|"
    r"list (them|those|it)|show (me )?(those|them|more|it)|"
    r"details|more( info)?|tell me more|and\b|what about|"
    r"same (for|with)|how much( was)? (that|it|those)|"
    r"those|them|that one|continue)\b",
    re.I,
)

BYE_RE = re.compile(
    r"\b(bye|goodbye|good bye|see you|stop listening|end conversation|that's all|thats all|quit)\b",
    re.I,
)


def _format_inr(val: float | Decimal) -> str:
    try:
        f = float(val)
        return f"₹{f:,.2f}"
    except (ValueError, TypeError):
        return f"₹{val}"


def _user_name(user, detected_name: str | None = None) -> str:
    if detected_name:
        return detected_name.strip().title()
    if not user:
        return ""
    first = getattr(user, "first_name", "") or ""
    name = first.strip() or (user.username if getattr(user, "username", None) else "")
    if name.lower() in {"admin", "administrator", "root"}:
        return ""
    return name.title()


def _call_name(user, detected_name: str | None = None) -> str:
    name = _user_name(user, detected_name)
    return f", {name}" if name else ""


def _speak_text(reply: str) -> str:
    text = re.sub(r"[*_`#]", "", reply or "")
    text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)
    text = text.replace("₹", "rupees ")
    text = re.sub(r"[•⚠️✅🤝💼🏛️📈👥💳🏷️📊🔍📋📅🗓️💰🏦📄💡]", " ", text)
    text = re.sub(r"\n+", ". ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 900:
        text = text[:880].rsplit(" ", 1)[0] + "."
    return text


def _normalize_history(history: list[dict] | None, query: str) -> list[dict]:
    cleaned: list[dict] = []
    if not history:
        return cleaned
    for turn in history:
        role_raw = (turn.get("role") or turn.get("sender") or "").lower()
        text = (turn.get("text") or turn.get("content") or "").strip()
        if not text:
            continue
        role = "assistant" if role_raw in {"assistant", "bot", "ai", "set"} else "user"
        cleaned.append({"role": role, "text": text})
    if cleaned and cleaned[-1]["role"] == "user" and cleaned[-1]["text"].strip() == (query or "").strip():
        cleaned = cleaned[:-1]
    return cleaned[-10:]


def _infer_year(txns: list[dict], fallback: int | None = None) -> int:
    years = []
    for t in txns:
        d = t.get("date") or ""
        if len(d) >= 4 and d[:4].isdigit():
            years.append(int(d[:4]))
    if years:
        return max(years)
    return fallback or date.today().year


def _extract_dates(query: str, default_year: int) -> list[str]:
    q = query.lower()
    found: list[str] = []

    for m in re.finditer(r"\b(20\d\d)[-/](0?[1-9]|1[0-2])[-/](0?[1-9]|[12]\d|3[01])\b", q):
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        found.append(f"{y:04d}-{mo:02d}-{d:02d}")

    for m in re.finditer(r"\b(0?[1-9]|[12]\d|3[01])[-/](0?[1-9]|1[0-2])[-/](20\d\d)\b", q):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        found.append(f"{y:04d}-{mo:02d}-{d:02d}")

    month_regex = (
        r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec|"
        r"january|february|march|april|june|july|august|september|october|november|december)"
    )
    for m in re.finditer(rf"\b(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+({month_regex})(?:\s+(20\d\d))?\b", q):
        d = int(m.group(1))
        mo = MONTH_MAP.get(m.group(2), 1)
        y = int(m.group(3)) if m.group(3) else default_year
        found.append(f"{y:04d}-{mo:02d}-{d:02d}")

    for m in re.finditer(rf"\b({month_regex})\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?(?:\s+(20\d\d))?\b", q):
        mo = MONTH_MAP.get(m.group(1), 1)
        d = int(m.group(2))
        y = int(m.group(3)) if m.group(3) else default_year
        found.append(f"{y:04d}-{mo:02d}-{d:02d}")

    return found


def _extract_month(query: str, default_year: int) -> tuple[int, int] | None:
    q = query.lower()
    months = (
        r"january|february|march|april|june|july|august|september|october|november|december|"
        r"jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
    )
    m = re.search(rf"\b(?:in|during|for|of)\s+({months}|may)(?:\s+(20\d\d))?\b", q)
    if not m:
        m = re.search(rf"\b({months})(?:\s+(20\d\d))\b", q)
    if not m:
        return None
    return MONTH_MAP.get(m.group(1), 1), int(m.group(2)) if m.group(2) else default_year


def _extract_bank(query: str) -> str | None:
    q = query.lower()
    for alias, needle in sorted(BANK_ALIASES.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(alias)}\b", q):
            return needle
    return None


def _extract_category(query: str) -> str | None:
    q = query.lower()
    extra = {
        "gst": "Taxes/GST",
        "tax": "Taxes/GST",
        "taxes": "Taxes/GST",
        "tds": "Taxes/GST",
        "vendor": "Vendor Payment",
        "vendors": "Vendor Payment",
        "supplier": "Vendor Payment",
        "payroll": "Salary",
        "salary": "Salary",
        "salaries": "Salary",
        "atm": "ATM/Cash",
        "cash": "ATM/Cash",
        "client": "Client Inflow",
        "revenue": "Client Inflow",
        "subscription": "Subscription",
        "saas": "Subscription",
        "professional": "Professional Fees",
        "logistics": "Logistics",
        "freight": "Logistics",
    }
    for cat in CATEGORIES:
        if cat.lower() in q:
            return cat
    for word, cat in extra.items():
        if re.search(rf"\b{re.escape(word)}\b", q):
            return cat
    return None


def _empty_state() -> dict[str, Any]:
    return {
        "intent": "open",
        "bank": None,
        "category": None,
        "party": None,
        "date_from": None,
        "date_to": None,
        "month": None,
        "year": None,
        "channel": None,
        "want_list": False,
    }


def _merge_state(prev: dict | None, incoming: dict) -> dict:
    base = _empty_state()
    if prev:
        for k in base:
            if prev.get(k) not in (None, ""):
                base[k] = prev[k]
    incoming = incoming or {}
    for k, v in incoming.items():
        if k == "intent" and v in {"open", "followup"}:
            continue
        if k == "want_list":
            base[k] = bool(v) or bool(base.get(k))
            continue
        if v not in (None, "", False):
            base[k] = v
    if incoming.get("intent") not in {None, "open", "followup"}:
        base["intent"] = incoming["intent"]
    elif incoming.get("intent") == "followup":
        base["want_list"] = True
        if base.get("intent") in {"open", None}:
            base["intent"] = "period"
    return base


def _parse_query_slots(query: str, default_year: int) -> dict[str, Any]:
    q = (query or "").strip()
    ql = q.lower()
    slots = _empty_state()

    if BYE_RE.search(ql):
        slots["intent"] = "bye"
        return slots
    if re.search(r"\b(who are you|what are you|your name|introduce yourself|what can you do)\b", ql):
        slots["intent"] = "identity"
        return slots
    
    # Catch conversational greetings and introductions like "Hello, I am Chetan", "Hi I'm Chetan"
    greet_m = re.search(r"\b(hello|hi|hey|good morning|good afternoon|good evening|greetings|howdy)\b", ql)
    intro_m = re.search(r"\b(?:i am|i'm|my name is|this is)\s+([a-zA-Z]{2,20})\b", ql)
    if intro_m:
        slots["detected_name"] = intro_m.group(1).title()
    if greet_m and not re.search(r"\b(spend|spent|pay|paid|balance|account|statement|transaction|how much)\b", ql):
        slots["intent"] = "greet"
        return slots

    if re.search(r"\b(thank you|thanks|cheers|appreciated)\b", ql) and len(ql.split()) <= 6:
        slots["intent"] = "thanks"
        return slots
    if re.search(r"\b(how are you|status report|system status)\b", ql):
        slots["intent"] = "status"
        return slots

    dates = _extract_dates(q, default_year)
    if len(dates) >= 2:
        d1, d2 = sorted(dates[:2])
        slots["date_from"], slots["date_to"] = d1, d2
    elif len(dates) == 1:
        slots["date_from"] = slots["date_to"] = dates[0]
    month_year = _extract_month(q, default_year)
    if month_year and not slots["date_from"]:
        slots["month"], slots["year"] = month_year

    slots["bank"] = _extract_bank(ql)
    slots["category"] = _extract_category(ql)

    for mode in ["upi", "neft", "imps", "rtgs", "pos", "atm", "cheque", "card"]:
        if re.search(rf"\b{mode}\b", ql):
            slots["channel"] = mode
            break

    from_to = re.search(
        r"\bfrom\s+([a-zA-Z0-9_.\- ]{2,40}?)\s+to\s+([a-zA-Z0-9_.\- ]{2,40}?)(?:\s+how|\s+what|\s*\?|\s*$)",
        ql,
    )
    to_m = re.search(
        r"\b(?:transferred|transfer|paid|sent|disbursed|given|spent)\s+(?:to|on|for)\s+([a-zA-Z0-9_.\- ]{2,40}?)(?:\s+how|\s*\?|\s*$)",
        ql,
    )
    from_m = re.search(
        r"\b(?:transferred|received|got|credited|came)\s+from\s+([a-zA-Z0-9_.\- ]{2,40}?)(?:\s+how|\s*\?|\s*$)",
        ql,
    )
    if from_to:
        slots["party"] = from_to.group(2).strip()
        slots["intent"] = "transfer"
    elif to_m:
        party = to_m.group(1).strip()
        if party not in {"me", "him", "her", "them", "it", "us", "this", "that"}:
            slots["party"] = party
            slots["intent"] = "transfer"
    elif from_m:
        party = from_m.group(1).strip()
        if party not in {"me", "him", "her", "them", "it", "us", "this", "that"}:
            slots["party"] = party
            slots["intent"] = "inflow"

    if re.search(r"\b(how many accounts|which accounts|list accounts|my accounts|bank accounts)\b", ql):
        slots["intent"] = "accounts"
    elif re.search(r"\b(how many statements|uploaded files|statement files|uploaded statements)\b", ql):
        slots["intent"] = "statements"
    elif re.search(r"\b(balance|closing balance|current balance|available balance|net balance)\b", ql):
        slots["intent"] = "balance"
    elif re.search(r"\b(cashflow|cash flow|runway|burn rate|financial health|pnl|profit|summary|overview)\b", ql):
        slots["intent"] = "summary"
    elif re.search(r"\b(highest|largest|biggest|max|maximum|top expense)\b", ql):
        slots["intent"] = "max"
    elif re.search(r"\b(smallest|lowest|minimum|min expense)\b", ql):
        slots["intent"] = "min"
    elif re.search(r"\b(average spend|average debit|average expense|avg debit)\b", ql):
        slots["intent"] = "average"
    elif re.search(r"\b(how many transactions|total transactions|number of transactions)\b", ql):
        slots["intent"] = "count"
    elif re.search(r"\b(subscription|subscriptions|recurring|monthly bills|autopay|repeat payments)\b", ql):
        slots["intent"] = "subscriptions"
    elif re.search(r"\b(merchant|merchants|top vendors|vendor breakdown|payee breakdown|who did i pay)\b", ql):
        slots["intent"] = "merchants"
    elif re.search(r"\b(spending speed|velocity|spending trend|monthly trend|am i spending more|trend analysis)\b", ql):
        slots["intent"] = "velocity"
    elif re.search(r"\b(advice|saving tips|how to save|recommendation|reduce expense|reduce spending|budget advice)\b", ql):
        slots["intent"] = "advice"
    elif slots["intent"] == "open" and slots["category"]:
        slots["intent"] = "category"
    elif slots["intent"] == "open" and (slots["date_from"] or slots["month"]):
        slots["intent"] = "period"
    elif slots["intent"] == "open" and slots["channel"]:
        slots["intent"] = "channel"
    elif slots["intent"] == "open" and slots["bank"]:
        slots["intent"] = "bank"

    if re.search(r"\b(show|list|itemi[sz]e|break ?down|details|transactions)\b", ql):
        slots["want_list"] = True

    if slots["intent"] == "open" and not slots["party"]:
        words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]{2,}\b", ql)
        candidates = [w for w in words if w not in STOP_WORDS]
        if candidates and re.search(r"\b(spend|spent|paid|pay|transfer|show|find|search|about|to|from)\b", ql):
            slots["party"] = max(candidates, key=len)
            slots["intent"] = "search"

    if FOLLOWUP_RE.search(ql) and slots["intent"] == "open":
        slots["intent"] = "followup"
        slots["want_list"] = True

    return slots


def _filter_txns(txns: list[dict], state: dict) -> list[dict]:
    matched = txns
    if state.get("bank"):
        needle = state["bank"].lower()
        matched = [t for t in matched if needle in (t.get("bank") or "").lower()]
    if state.get("category"):
        cat = state["category"].lower()
        matched = [t for t in matched if (t.get("category") or "").lower() == cat]
    if state.get("party"):
        party = state["party"].lower()
        matched = [
            t for t in matched
            if party in (t.get("description") or "").lower()
            or party in (t.get("bank") or "").lower()
            or party in (t.get("category") or "").lower()
        ]
    if state.get("channel"):
        ch = state["channel"]
        matched = [
            t for t in matched
            if ch in (t.get("description") or "").lower()
            or (ch == "atm" and (t.get("category") or "") == "ATM/Cash")
        ]
    if state.get("date_from") and state.get("date_to"):
        d1, d2 = state["date_from"], state["date_to"]
        matched = [t for t in matched if t.get("date") and d1 <= t["date"] <= d2]
    elif state.get("month") and state.get("year"):
        prefix = f"{int(state['year']):04d}-{int(state['month']):02d}"
        matched = [t for t in matched if (t.get("date") or "").startswith(prefix)]
    return matched


def _scope_label(state: dict) -> str:
    bits = []
    if state.get("bank"):
        bits.append(state["bank"])
    if state.get("category"):
        bits.append(state["category"])
    if state.get("party"):
        bits.append(state["party"].title())
    if state.get("channel"):
        bits.append(state["channel"].upper())
    if state.get("date_from") and state.get("date_to"):
        if state["date_from"] == state["date_to"]:
            bits.append(f"on {state['date_from']}")
        else:
            bits.append(f"{state['date_from']} to {state['date_to']}")
    elif state.get("month") and state.get("year"):
        bits.append(f"{date(2000, int(state['month']), 1):%B} {state['year']}")
    return " · ".join(bits) if bits else "all indexed records"


def _totals(txns: list[dict]) -> tuple[float, float, float]:
    debit = sum(t["debit"] for t in txns)
    credit = sum(t["credit"] for t in txns)
    return debit, credit, credit - debit


def _list_lines(txns: list[dict], limit: int = 10) -> list[str]:
    lines = []
    for t in txns[:limit]:
        amt = f"Debit {_format_inr(t['debit'])}" if t["debit"] > 0 else f"Credit {_format_inr(t['credit'])}"
        lines.append(f"• **{t['date'] or 'N/A'}** | {amt} | *{t['category']}* — {t['description']}")
    if len(txns) > limit:
        lines.append(f"\n*(Showing {limit} of {len(txns)} transactions. Ask me to list more, or narrow by date or bank.)*")
    return lines


def _compose_ledger_answer(query: str, txns: list[dict], metrics: dict, user, state: dict) -> str:
    user_name = _user_name(user, state.get("detected_name"))
    name_clause = f", {user_name}" if user_name else ""
    intent = state.get("intent") or "open"
    scoped = _filter_txns(txns, state)
    debit, credit, net = _totals(scoped)
    label = _scope_label(state)

    if intent == "bye":
        return f"Conversation paused{name_clause}. Ask SET anytime you want to pick this back up!"

    if intent == "greet":
        intro = f"Hey {user_name}! Great to meet you. " if user_name else "Hello! "
        return (
            f"{intro}I'm **SET**, your Smart Expense Tracker assistant.\n\n"
            f"I have your financial ledger loaded: **{metrics.get('count', len(txns))}** transactions "
            f"(inflows **{_format_inr(metrics.get('credit', 0))}**, outflows **{_format_inr(metrics.get('debit', 0))}**).\n\n"
            "How can I help you today? You can ask me about recent expenses, top merchants, recurring subscriptions, or monthly balance summaries!"
        )

    if intent == "identity":
        intro = f"I'm **SET**, your personal financial assistant{name_clause}.\n\n"
        return (
            f"{intro}"
            "I analyze your bank statements, track spending trends, identify recurring bills, and help you manage your accounts. "
            "Feel free to ask me anything about your transactions, vendors, or cashflow!"
        )

    if intent == "thanks":
        return f"You're very welcome{name_clause}! Let me know if you need help with anything else in your ledger."

    if intent == "status":
        health = "surplus" if metrics.get("net", 0) >= 0 else "deficit"
        return (
            f"SET is online and ready{name_clause}. Your ledger shows a net **{health} of {_format_inr(metrics.get('net', 0))}** "
            f"across **{metrics.get('count', len(txns))}** transactions. What shall we review first?"
        )

    if intent == "accounts" and user:
        accs = BankAccount.objects.filter(user=user)
        if accs.exists():
            lines = [f"Here are the **{accs.count()}** bank accounts connected to your profile{name_clause}:\n"]
            for a in accs:
                lines.append(f"• **{a.bank_name}** | Holder: *{a.account_holder}* | `{a.display_account}`")
            lines.append("\nSay a bank name anytime and I will filter our conversation to that account.")
            return "\n".join(lines)

    if intent == "statements" and user:
        stmts = StatementFile.objects.filter(batch__user=user)
        if stmts.exists():
            lines = [f"SET has processed **{stmts.count()}** statement files{name_clause}:\n"]
            for s in stmts:
                acc_info = f" ({s.account.bank_name} · {s.account.display_account})" if s.account else ""
                lines.append(f"• **{s.original_name}**{acc_info} — {s.created_at.strftime('%d %b %Y')}")
            lines.append("\nWould you like to inspect transactions from any specific file?")
            return "\n".join(lines)

    if intent == "balance":
        with_bal = [t for t in scoped if t.get("balance") is not None]
        if with_bal:
            latest = with_bal[0]
            return (
                f"Your latest **closing balance**{name_clause} ({label}):\n\n"
                f"• **{_format_inr(latest['balance'])}** as of **{latest['date'] or 'N/A'}**\n"
                f"• Bank: {latest.get('bank') or '—'}\n"
                f"• Last narration: {latest['description']}\n\n"
                "Would you like me to show the recent transactions leading up to this balance?"
            )
        return f"Your net position for {label}{name_clause} is **{_format_inr(net)}** (credits {_format_inr(credit)} − debits {_format_inr(debit)})."

    if intent == "max":
        debits = [t for t in scoped if t["debit"] > 0]
        if debits:
            max_t = max(debits, key=lambda x: x["debit"])
            return (
                f"Your largest outflow{name_clause} ({label}) was:\n\n"
                f"• **{_format_inr(max_t['debit'])}** — {max_t['description']}\n"
                f"• Category: {max_t['category']} · Date: {max_t['date'] or 'N/A'}\n\n"
                "Would you like to see other large payments as well?"
            )

    if intent == "min":
        debits = [t for t in scoped if t["debit"] > 0]
        if debits:
            min_t = min(debits, key=lambda x: x["debit"])
            return (
                f"Your smallest outflow{name_clause} ({label}) was:\n\n"
                f"• **{_format_inr(min_t['debit'])}** — {min_t['description']}\n"
                f"• Category: {min_t['category']} · Date: {min_t['date'] or 'N/A'}"
            )

    if intent == "average":
        debits = [t["debit"] for t in scoped if t["debit"] > 0]
        if debits:
            avg = sum(debits) / len(debits)
            return (
                f"Your average expense{name_clause} for {label} is **{_format_inr(avg)}** "
                f"across {len(debits)} debits. Would you like to compare another category or month?"
            )

    if intent == "count":
        txn_p = "transaction" if len(scoped) == 1 else "transactions"
        return (
            f"You have **{len(scoped)}** {txn_p} {label}{name_clause}. "
            f"Inflows: **{_format_inr(credit)}**, Outflows: **{_format_inr(debit)}**. "
            "Would you like me to list them out or group them by category?"
        )

    if intent == "subscriptions":
        subs = [
            t for t in scoped
            if t["category"] in {"Subscription", "Bills"}
            or any(w in t["description"].lower() for w in ["netflix", "spotify", "prime", "aws", "swiggy one", "zomato gold", "hotstar", "google", "apple", "adobe", "wifi", "broadband", "electricity", "rent"])
        ]
        if subs:
            total_sub = sum(t["debit"] for t in subs)
            unique_desc: dict[str, float] = {}
            for t in subs:
                key = t["description"].strip()
                unique_desc[key] = unique_desc.get(key, 0.0) + float(t["debit"])
            sub_lines = [f"• **{k}**: {_format_inr(v)}" for k, v in sorted(unique_desc.items(), key=lambda x: x[1], reverse=True)[:7]]
            return (
                f"Here are your **recurring bills & subscriptions**{name_clause} ({label}):\n\n"
                f"• Total Subscriptions & Bills: **{_format_inr(total_sub)}** across {len(subs)} payments\n\n"
                "**Breakdown:**\n" + "\n".join(sub_lines) + "\n\n"
                "Would you like details on any specific bill?"
            )
        return f"I didn't find any recurring subscriptions or bill payments for {label}{name_clause}."

    if intent == "merchants":
        debits = [t for t in scoped if t["debit"] > 0]
        if debits:
            merchants: dict[str, dict] = {}
            for t in debits:
                desc = t["description"].strip()
                key = desc.split()[0].title() if len(desc.split()) > 0 else desc
                if key not in merchants:
                    merchants[key] = {"total": 0.0, "count": 0, "sample": desc}
                merchants[key]["total"] += float(t["debit"])
                merchants[key]["count"] += 1
            sorted_merchants = sorted(merchants.items(), key=lambda x: x[1]["total"], reverse=True)[:5]
            lines = [f"• **{m}** (*{info['sample'][:30]}*): {_format_inr(info['total'])} ({info['count']} txns)" for m, info in sorted_merchants]
            return (
                f"Here are your top **merchants & payees**{name_clause} ({label}):\n\n"
                + "\n".join(lines) + "\n\n"
                "Would you like to list transactions for any of these merchants?"
            )

    if intent == "velocity":
        by_month: dict[str, float] = {}
        for t in scoped:
            if t["debit"] > 0 and t.get("date"):
                m_key = t["date"][:7]
                by_month[m_key] = by_month.get(m_key, 0.0) + float(t["debit"])
        sorted_months = sorted(by_month.items())
        if len(sorted_months) >= 2:
            m1, v1 = sorted_months[-2]
            m2, v2 = sorted_months[-1]
            diff = v2 - v1
            pct = ((v2 - v1) / v1 * 100) if v1 > 0 else 0
            direction = "increased" if diff > 0 else "decreased"
            return (
                f"Here is your **spending velocity trend**{name_clause}:\n\n"
                f"• **{m1}**: {_format_inr(v1)}\n"
                f"• **{m2}**: {_format_inr(v2)}\n\n"
                f"Your outflows {direction} by **{_format_inr(abs(diff))} ({abs(pct):.1f}%)** between {m1} and {m2}.\n"
                "Would you like a breakdown of what drove this change?"
            )
        return f"There is insufficient multi-month data to compute spending velocity for {label}{name_clause}."

    if intent == "advice":
        by_cat: dict[str, float] = {}
        for t in scoped:
            if t["debit"] > 0:
                by_cat[t["category"]] = by_cat.get(t["category"], 0.0) + float(t["debit"])
        top_cats = sorted(by_cat.items(), key=lambda x: x[1], reverse=True)[:2]
        health_tip = "You have a positive net surplus! Consider moving excess funds to investments or high-yield savings." if net >= 0 else "Your outflows exceed inflows. Consider capping optional spending."
        cat_tips = ""
        if top_cats:
            cat_tips = f"\n• Your highest spending goes to **{top_cats[0][0]}** ({_format_inr(top_cats[0][1])}). Setting a 15% budget cap here could save you **{_format_inr(top_cats[0][1] * 0.15)}** monthly."
        return (
            f"**Financial Advisory & Smart Tips**{name_clause} ({label}):\n\n"
            f"• {health_tip}"
            f"{cat_tips}\n"
            f"• Track unallocated ATM cash withdrawals to maintain 100% auditable accounting.\n\n"
            "Ask me to analyze any specific spending category to explore further optimization!"
        )

    if intent == "summary":
        by_cat: dict[str, float] = {}
        for t in scoped:
            if t["debit"] > 0:
                by_cat[t["category"]] = by_cat.get(t["category"], 0.0) + float(t["debit"])
        cat_lines = [f"• **{c}**: {_format_inr(a)}" for c, a in sorted(by_cat.items(), key=lambda x: x[1], reverse=True)[:6]]
        health = "surplus" if net >= 0 else "deficit"
        return (
            f"Here is your cashflow overview for **{label}**{name_clause}:\n\n"
            f"• Inflows: {_format_inr(credit)}\n"
            f"• Outflows: {_format_inr(debit)}\n"
            f"• Net Position: {_format_inr(net)} ({health})\n"
            f"• Total Transactions: {len(scoped)}\n\n"
            + ("**Top spend categories:**\n" + "\n".join(cat_lines) + "\n\n" if cat_lines else "")
            + "Which category or bank would you like to look into next?"
        )

    if not scoped and (state.get("bank") or state.get("category") or state.get("party") or state.get("date_from") or state.get("month")):
        return (
            f"I didn't find any matching transactions for **{label}** in your ledger{name_clause}. "
            "Try specifying another date, bank, or merchant — or ask me to show all transactions!"
        )

    label_str = f"for **{label}**" if label != "all indexed records" else "in your ledger"
    txn_plural = "transaction" if len(scoped) == 1 else "transactions"
    lines = [
        f"Here is what I found {label_str}{name_clause}: **{len(scoped)}** {txn_plural} totaling **{_format_inr(debit)}** in outflows and **{_format_inr(credit)}** in inflows.\n"
    ]
    if state.get("want_list") or intent in {"period", "transfer", "inflow", "search", "channel", "followup", "category"}:
        lines.append("**Matching entries:**")
        lines.extend(_list_lines(scoped, 12 if state.get("want_list") else 8))
    else:
        lines.append("**Recent entries:**")
        lines.extend(_list_lines(scoped, 5))
    lines.append(f"\nWhat would you like to review next{name_clause}?")
    return "\n".join(lines)


def _try_eval_math(query: str, user_name: str = "") -> str | None:
    q = query.lower().strip()
    name_suffix = f", {user_name}" if user_name else ""
    m_pct = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of)\s*(?:₹|rs\.?)?\s*(\d+(?:,\d+)*(?:\.\d+)?)", q)
    if m_pct:
        pct = float(m_pct.group(1))
        val = float(m_pct.group(2).replace(",", ""))
        res = (pct / 100.0) * val
        return f"**{pct}% of {_format_inr(val)}** is **{_format_inr(res)}**{name_suffix}. Apply that rate to a category in this ledger?"
    m_math = re.search(r"(?:what is|calculate|solve|evaluate)?\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*([\+\-\*\/])\s*(\d+(?:,\d+)*(?:\.\d+)?)\s*\??$", q)
    if m_math:
        a = float(m_math.group(1).replace(",", ""))
        op = m_math.group(2)
        b = float(m_math.group(3).replace(",", ""))
        if op == "/" and b == 0:
            return f"Division by zero is undefined{name_suffix}."
        ops = {"+": a + b, "-": a - b, "*": a * b, "/": a / b}
        res = ops.get(op)
        if res is not None:
            return f"**{a:,.2f} {op} {b:,.2f}** = **{res:,.2f}**{name_suffix}."
    return None


def _call_groq_api(api_key: str, system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        for turn in history:
            messages.append({"role": turn["role"], "content": turn["text"]})
    messages.append({"role": "user", "content": user_prompt})
    last_err = None
    for model in models:
        payload = {"model": model, "messages": messages, "temperature": 0.35, "max_tokens": 900}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "SET-ExpenseTracker/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                choices = data.get("choices", [])
                if choices:
                    return choices[0].get("message", {}).get("content", "").strip()
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}"
            if e.code not in {404, 400}:
                break
        except Exception as e:
            last_err = str(e)
            break
    raise RuntimeError(f"Groq API error: {last_err}")


def _call_gemini_api(api_key: str, system_prompt: str, user_prompt: str, history: list[dict] | None = None) -> str:
    models = ["gemini-2.5-flash", "gemini-1.5-flash"]
    last_err = None
    contents = []
    if history:
        for turn in history:
            role = "model" if turn["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": turn["text"]}]})
    contents.append({"role": "user", "parts": [{"text": user_prompt}]})
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {"temperature": 0.35, "maxOutputTokens": 900},
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts and "text" in parts[0]:
                        return parts[0]["text"].strip()
        except Exception as e:
            last_err = str(e)
            continue
    raise RuntimeError(f"Gemini API error: {last_err}")


def _local_fallback_answer(query: str, txns: list[dict], metrics: dict, user=None, state: dict | None = None) -> str:
    math_ans = _try_eval_math(query, user_name=_user_name(user))
    if math_ans:
        return math_ans
    return _compose_ledger_answer(query, txns, metrics, user, state or _empty_state())


def answer_transaction_query(
    user,
    query: str,
    account_id: str | None = None,
    client_api_key: str | None = None,
    history: list[dict] | None = None,
    client_state: dict | None = None,
) -> dict[str, Any]:
    """Answer as SET in a multi-turn conversation over this user's ledger."""
    qs = Transaction.objects.filter(user=user)
    if account_id:
        try:
            qs = qs.filter(account_id=int(account_id))
        except (ValueError, TypeError):
            pass

    txns_data = []
    total_debit = Decimal("0.00")
    total_credit = Decimal("0.00")
    cat_totals: dict[str, Decimal] = {}

    for t in qs.order_by("-txn_date", "-id"):
        d = float(t.debit)
        c = float(t.credit)
        total_debit += t.debit
        total_credit += t.credit
        if t.debit > 0:
            cat_totals[t.category] = cat_totals.get(t.category, Decimal("0.00")) + t.debit
        txns_data.append({
            "id": t.id,
            "date": str(t.txn_date or ""),
            "description": t.description,
            "debit": d,
            "credit": c,
            "balance": float(t.balance) if t.balance is not None else None,
            "category": t.category,
            "bank": t.account.bank_name if t.account else "",
        })

    net = total_credit - total_debit
    metrics = {
        "debit": float(total_debit),
        "credit": float(total_credit),
        "net": float(net),
        "count": len(txns_data),
    }

    # 1. Conversational Memory Buffer & Query Condensation
    mem_buffer = SessionMemoryBuffer(max_turns=8)
    mem_buffer.load_from_history(history or [])
    rewritten_query = mem_buffer.rewrite_query_with_context(query)

    # 2. Local Semantic Vector Store (RAG Search)
    parsed_acc_id = None
    if account_id:
        try:
            parsed_acc_id = int(account_id)
        except (ValueError, TypeError):
            parsed_acc_id = None
    vstore = get_user_vector_store(user, account_id=parsed_acc_id)
    rag_results = vstore.search(rewritten_query or query, top_k=6)
    rag_chunks = [f"- {r['chunk']}" for r in rag_results]

    prior = _normalize_history(history, query)
    default_year = _infer_year(txns_data)
    incoming = _parse_query_slots(query, default_year)
    ql = (query or "").lower()
    if re.search(r"\b(all accounts|reset filters|start over|clear filter|whole ledger)\b", ql):
        client_state = None
    if incoming.get("intent") in {"greet", "identity", "thanks", "bye", "status"}:
        state = incoming
    else:
        state = _merge_state(client_state, incoming)
        if incoming.get("date_from"):
            state["month"] = None
        if incoming.get("month") and not incoming.get("date_from"):
            state["date_from"] = state["date_to"] = None

    bank_accs = BankAccount.objects.filter(user=user)
    acc_summary_lines = [
        f"- Bank: {a.bank_name} | Holder: {a.account_holder} | Account: {a.display_account}"
        for a in bank_accs
    ]
    user_name = _user_name(user) or "User"
    cat_lines = [f"- {cat}: {_format_inr(amt)}" for cat, amt in sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)]
    filtered = _filter_txns(txns_data, state)
    preview_src = filtered[:40] if filtered else txns_data[:40]
    txn_lines = [
        f"- Date: {t['date'] or 'N/A'} | Narration: {t['description']} | Debit: {_format_inr(t['debit'])} | Credit: {_format_inr(t['credit'])} | Balance: {_format_inr(t['balance']) if t['balance'] is not None else 'N/A'} | Category: {t['category']} | Bank: {t['bank']}"
        for t in preview_src
    ]

    system_prompt = f"""You are SET (Smart Expense Tracker AI) inside this Django project. You are talking with {user_name}.

You hold a live conversation. Remember the last topic. If they say "list those", "what about February", or "same for HDFC", continue from the current filters.

Rules:
- Use only this project's ledger. Amounts in Indian Rupees (₹).
- Address {user_name} naturally. Never say "boss".
- Answer the latest question first, then one short follow-up question so the dialogue can continue.
- If they go off-topic, steer back to their statements, accounts, categories, and cashflow.
- If no rows match, say so clearly and suggest another filter.
- Keep replies readable: short paragraphs and bullets, not a dump.

Conversation filters in play:
{json.dumps({k: v for k, v in state.items() if v}, ensure_ascii=False)}

Ledger totals:
- Outflows: {_format_inr(total_debit)}
- Inflows: {_format_inr(total_credit)}
- Net: {_format_inr(net)}
- Rows: {len(txns_data)}

Accounts:
{chr(10).join(acc_summary_lines) if acc_summary_lines else "None uploaded yet."}

Category spend:
{chr(10).join(cat_lines) if cat_lines else "None"}

Semantically retrieved RAG records (Top matches for: "{rewritten_query}"):
{chr(10).join(rag_chunks) if rag_chunks else "No specific semantic matches found."}

Recent transactions in focus:
{chr(10).join(txn_lines) if txn_lines else "No transactions recorded."}
"""

    groq_key = (
        (client_api_key or "").strip() if "gsk_" in (client_api_key or "") else ""
    ) or getattr(settings, "GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")

    gemini_key = (
        (client_api_key or "").strip() if "AIza" in (client_api_key or "") else ""
    ) or getattr(settings, "GEMINI_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")

    reply = None
    model_name = "set-local"
    if groq_key:
        try:
            reply = _call_groq_api(groq_key, system_prompt, query, history=prior)
            model_name = "set-groq-llama3"
        except Exception:
            reply = None
    if not reply and gemini_key:
        try:
            reply = _call_gemini_api(gemini_key, system_prompt, query, history=prior)
            model_name = "set-gemini-flash"
        except Exception:
            reply = None
    if not reply:
        reply = _local_fallback_answer(query, txns_data, metrics, user=user, state=state)
        model_name = "set-local"

    return {
        "status": "ok",
        "reply": reply,
        "speak": _speak_text(reply),
        "state": state,
        "model": model_name,
        "has_api_key": bool(groq_key or gemini_key),
        "end_conversation": state.get("intent") == "bye",
    }
