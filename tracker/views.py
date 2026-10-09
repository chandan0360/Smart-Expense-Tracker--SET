from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db import transaction as db_transaction
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.views.decorators.http import require_POST

from ai_engine.categories import CATEGORIES
from tracker.analytics import account_metrics, get_user_account, metrics_for_charts, user_accounts
from tracker.filters import (
    apply_transaction_filters,
    date_bounds,
    query_string,
    requested_filters,
    user_banks,
    user_statements,
)
from tracker.models import BankAccount, StatementFile, Transaction, UploadBatch
from tracker.pipeline import PipelineError, process_batch


def _allowed(name: str) -> bool:
    return Path(name).suffix.lower() in settings.ALLOWED_STATEMENT_EXTENSIONS


def _filter_context(request, user):
    filters = requested_filters(request)
    first, last = date_bounds(user)
    return {
        "filters": filters,
        "filter_query": query_string(filters),
        "qs_no_account": query_string({**filters, "account_id": ""}),
        "accounts": user_accounts(user),
        "banks": user_banks(user),
        "statements": user_statements(user),
        "categories": CATEGORIES,
        "date_min": first.isoformat() if first else "",
        "date_max": last.isoformat() if last else "",
    }


@login_required
def home(request):
    error = request.session.pop("pipeline_error", None)
    last_batch = UploadBatch.objects.filter(user=request.user).first()
    return render(
        request,
        "tracker/home.html",
        {
            "pipeline_error": error,
            "accounts": user_accounts(request.user),
            "last_batch": last_batch,
            "has_transactions": Transaction.objects.filter(user=request.user).exists(),
        },
    )


@login_required
def upload_view(request):
    if request.method == "POST":
        files = request.FILES.getlist("statements")
        if not files:
            request.session["pipeline_error"] = "Please choose at least one bank statement file."
            return redirect("home")
        if len(files) > settings.MAX_UPLOAD_FILES:
            request.session["pipeline_error"] = "Too many files in one batch."
            return redirect("home")
        bad = [f.name for f in files if not _allowed(f.name)]
        if bad:
            request.session["pipeline_error"] = (
                "Only bank statements are allowed (PDF, CSV, Excel, TSV, TXT). Rejected: "
                + ", ".join(bad)
            )
            return redirect("home")

        batch = UploadBatch.objects.create(user=request.user, file_count=len(files))
        for uploaded in files:
            StatementFile.objects.create(
                batch=batch,
                file=uploaded,
                original_name=uploaded.name,
            )
        pdf_password = (request.POST.get("pdf_password") or "").strip() or None
        try:
            process_batch(batch, password=pdf_password)
            from ai_engine.rag import get_user_vector_store
            get_user_vector_store(request.user, force_refresh=True)
        except PipelineError as exc:
            request.session["pipeline_error"] = exc.message
            return redirect("home")
        request.session["active_batch_id"] = batch.id
        if batch.duplicates_skipped:
            messages.success(
                request,
                f"{batch.transaction_count} new transaction(s) added, "
                f"{batch.duplicates_skipped} duplicate(s) skipped (already in your account)."
            )
        else:
            messages.success(request, "Statements extracted. Review transactions, then open the dashboard.")
        return redirect("transactions")
    return redirect("home")


@login_required
def transactions_view(request):
    base = Transaction.objects.filter(user=request.user).select_related("account", "statement")
    if not base.exists():
        request.session["pipeline_error"] = "No transactions yet. Upload a bank statement first."
        return redirect("home")
    ctx = _filter_context(request, request.user)
    qs = apply_transaction_filters(base, ctx["filters"])

    paginator = Paginator(qs, 50)  # 50 rows per page keeps the ledger table readable and fast
    page_number = request.GET.get("page") or 1
    page_obj = paginator.get_page(page_number)

    ctx.update({"transactions": page_obj.object_list, "page_obj": page_obj, "paginator": paginator})
    return render(request, "tracker/transactions.html", ctx)


from decimal import Decimal
import re

from tracker.models import BankAccount, CategoryRule, StatementFile, Transaction, UploadBatch


@login_required
@require_POST
def update_category(request, pk: int):
    txn = get_object_or_404(Transaction, pk=pk, user=request.user)
    category = (request.POST.get("category") or "").strip()
    if category in CATEGORIES:
        txn.category = category
        txn.confidence = 1.0
        txn.save(update_fields=["category", "confidence"])

        # Extract keyword for Active Learning / Continuous Improvement
        from ai_engine.classifier import clean_narration
        raw_desc = txn.description
        m = re.search(r"upi/(?:dr|cr)/\d+/([^/]+)", raw_desc, re.IGNORECASE)
        if m:
            keyword = m.group(1).strip()
        else:
            cleaned = clean_narration(raw_desc)
            words = [w for w in cleaned.split() if len(w) >= 3 and not w.isdigit()]
            keyword = " ".join(words[:3]) if words else cleaned[:30]

        if keyword:
            CategoryRule.objects.update_or_create(
                user=request.user,
                keyword=keyword.lower(),
                defaults={"category": category},
            )
            # Reclassify other matching transactions for this user
            updated_count = Transaction.objects.filter(
                user=request.user, description__icontains=keyword
            ).exclude(pk=txn.pk).update(category=category, confidence=1.0)

            msg = f"Category updated to '{category}'."
            if updated_count > 0:
                msg += f" AI learned '{keyword}' and updated {updated_count} matching transaction(s)."
            else:
                msg += f" AI learned rule for '{keyword}'."
            messages.success(request, msg)
        else:
            messages.success(request, "Category updated.")

    next_url = request.POST.get("next") or reverse("transactions")
    return redirect(next_url)


@login_required
@require_POST
def add_manual_transaction(request):
    txn_date_str = request.POST.get("txn_date")
    description = (request.POST.get("description") or "").strip()
    amount_str = request.POST.get("amount") or "0"
    is_credit = request.POST.get("txn_type") == "credit"
    category = (request.POST.get("category") or "Other").strip()
    account_id = request.POST.get("account_id")
    parent_id = request.POST.get("parent_id")
    notes = (request.POST.get("notes") or "").strip()

    if not description:
        messages.error(request, "Description/Payee is required.")
        return redirect(request.POST.get("next") or "transactions")

    try:
        amount = abs(Decimal(amount_str))
    except Exception:
        amount = Decimal("0.00")

    debit = Decimal("0.00") if is_credit else amount
    credit = amount if is_credit else Decimal("0.00")

    account = None
    if account_id:
        try:
            account = BankAccount.objects.get(id=int(account_id), user=request.user)
        except (BankAccount.DoesNotExist, ValueError):
            pass
    if not account:
        account = BankAccount.objects.filter(user=request.user).first()
        if not account:
            account = BankAccount.objects.create(
                user=request.user,
                bank_name="Cash / Petty Cash Wallet",
                account_holder=request.user.username,
                account_number="CASH-001",
                display_account="Petty Cash",
                fingerprint=f"cash_{request.user.id}",
            )

    parent = None
    if parent_id:
        try:
            parent = Transaction.objects.get(id=int(parent_id), user=request.user)
        except (Transaction.DoesNotExist, ValueError):
            pass

    from datetime import date
    from dateutil import parser as d_parser
    try:
        txn_date = d_parser.parse(txn_date_str).date() if txn_date_str else date.today()
    except Exception:
        txn_date = date.today()

    Transaction.objects.create(
        user=request.user,
        account=account,
        txn_date=txn_date,
        description=description,
        debit=debit,
        credit=credit,
        category=category if category in CATEGORIES else "Other",
        confidence=1.0,
        is_manual=True,
        parent_transaction=parent,
        notes=notes,
    )
    if parent:
        messages.success(request, f"Cash spend of ₹{amount:,.2f} recorded and linked to withdrawal '{parent.description[:25]}'.")
    else:
        messages.success(request, f"Manual transaction '{description}' (₹{amount:,.2f}) added.")

    return redirect(request.POST.get("next") or "transactions")


@login_required
@require_POST
def clear_extracted(request):
    user = request.user
    with db_transaction.atomic():
        files = list(StatementFile.objects.filter(batch__user=user))
        Transaction.objects.filter(user=user).delete()
        for statement in files:
            if statement.file:
                statement.file.delete(save=False)
        StatementFile.objects.filter(batch__user=user).delete()
        UploadBatch.objects.filter(user=user).delete()
        BankAccount.objects.filter(user=user).delete()
    request.session.pop("active_batch_id", None)
    messages.success(request, "Extracted transactions were cleared. Upload only the statements you need.")
    return redirect("home")


@login_required
def dashboard_view(request):
    base = Transaction.objects.filter(user=request.user)
    if not base.exists():
        request.session["pipeline_error"] = "Dashboard needs extracted transactions. Upload statements first."
        return redirect("home")
    ctx = _filter_context(request, request.user)
    qs = apply_transaction_filters(base, ctx["filters"])
    metrics = account_metrics(request.user, qs=qs)
    ctx.update(
        {
            "mode": "overall",
            "metrics": metrics,
            "chart_data": metrics_for_charts(metrics),
            "active_account": None,
        }
    )
    return render(request, "tracker/dashboard.html", ctx)


@login_required
def dashboard_account_view(request, pk: int):
    base = Transaction.objects.filter(user=request.user)
    if not base.exists():
        request.session["pipeline_error"] = "Dashboard needs extracted transactions. Upload statements first."
        return redirect("home")
    account = get_user_account(request.user, pk)
    ctx = _filter_context(request, request.user)
    ctx["filters"]["account_id"] = str(account.id)
    qs = apply_transaction_filters(base, ctx["filters"])
    metrics = account_metrics(request.user, qs=qs)
    ctx.update(
        {
            "mode": "account",
            "metrics": metrics,
            "chart_data": metrics_for_charts(metrics),
            "active_account": account,
            "filter_query": query_string(ctx["filters"]),
        }
    )
    return render(request, "tracker/dashboard.html", ctx)


@login_required
def chat_api_view(request):
    import json
    from django.http import JsonResponse
    from ai_engine.chat import answer_transaction_query

    if request.method != "POST":
        return JsonResponse({"status": "error", "error": "Only POST requests are supported."}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        body = request.POST.dict()

    query = (body.get("message") or body.get("query") or "").strip()
    if not query:
        return JsonResponse({"status": "error", "error": "Question cannot be empty."}, status=400)

    account_id = body.get("account_id") or request.GET.get("account")
    api_key = body.get("api_key")
    history = body.get("history")
    client_state = body.get("state") if isinstance(body.get("state"), dict) else None

    result = answer_transaction_query(
        user=request.user,
        query=query,
        account_id=account_id,
        client_api_key=api_key,
        history=history,
        client_state=client_state,
    )
    return JsonResponse(result)


_TTS_AUDIO_CACHE: dict[str, bytes] = {}
_MAX_TTS_CACHE_ENTRIES = 150


@login_required
def tts_api_view(request):
    import asyncio
    import hashlib
    import json
    import os
    import re
    import urllib.request
    from django.conf import settings
    from django.http import HttpResponse, JsonResponse
    import edge_tts

    if request.method != "POST":
        return JsonResponse({"status": "error", "error": "Only POST requests allowed."}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
    except Exception:
        body = request.POST.dict()

    text = (body.get("text") or "").strip()
    if not text:
        return JsonResponse({"status": "error", "error": "No text provided."}, status=400)

    # Clean text for neural speech synthesis (spoken conversation, not raw markdown)
    clean_text = re.sub(r"<[^>]*>", "", text)
    clean_text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", clean_text)
    clean_text = re.sub(r"https?://\S+", "", clean_text)
    clean_text = re.sub(r"[*_#`~]", "", clean_text)
    clean_text = re.sub(r"[•⚠️✅🤝💼🏛️📈👥💳🏷️📊🔍📋📅🗓️💰🏦📄🔗]", " ", clean_text)
    clean_text = clean_text.replace("₹", " Rupees ")
    clean_text = re.sub(r"\bRs\.?\b", " Rupees ", clean_text, flags=re.I)
    clean_text = re.sub(r"\bINR\b", " Rupees ", clean_text, flags=re.I)
    clean_text = re.sub(r"\bSET\b", "Set", clean_text)
    clean_text = re.sub(r"\bUPI\b", "U P I", clean_text, flags=re.I)
    clean_text = re.sub(r"\bDR\b", "Debit", clean_text)
    clean_text = re.sub(r"\bCR\b", "Credit", clean_text)
    clean_text = re.sub(r"\bATM\b", "A T M", clean_text, flags=re.I)
    clean_text = re.sub(r"\bIMPS\b", "I M P S", clean_text, flags=re.I)
    clean_text = re.sub(r"\bNEFT\b", "N E F T", clean_text, flags=re.I)
    clean_text = re.sub(r"\bRTGS\b", "R T G S", clean_text, flags=re.I)
    clean_text = re.sub(r"\bEMI\b", "E M I", clean_text, flags=re.I)
    clean_text = re.sub(r"\bGST\b", "G S T", clean_text, flags=re.I)
    clean_text = re.sub(r"\bTDS\b", "T D S", clean_text, flags=re.I)
    clean_text = re.sub(r"\bFD\b", "Fixed Deposit", clean_text, flags=re.I)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()
    if len(clean_text) > 1200:
        clean_text = clean_text[:1180].rsplit(" ", 1)[0] + "."
    if not clean_text:
        return JsonResponse({"status": "error", "error": "Empty sanitized text."}, status=400)

    voice_name = body.get("voice") or "en-IN-NeerjaNeural"
    cache_key = hashlib.md5(f"{voice_name}:{clean_text}".encode("utf-8")).hexdigest()
    if cache_key in _TTS_AUDIO_CACHE:
        return HttpResponse(_TTS_AUDIO_CACHE[cache_key], content_type="audio/mpeg")

    deepgram_key = (
        (body.get("deepgram_api_key") or "").strip()
        or getattr(settings, "DEEPGRAM_API_KEY", "")
        or os.environ.get("DEEPGRAM_API_KEY", "")
    )

    # 1. Deepgram Aura Voice TTS (Asteria / Luna model)
    if deepgram_key:
        try:
            url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
            payload_data = json.dumps({"text": clean_text}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload_data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Token {deepgram_key}",
                    "User-Agent": "SET-Voice/1.0",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                audio_bytes = resp.read()
                if len(_TTS_AUDIO_CACHE) >= _MAX_TTS_CACHE_ENTRIES:
                    _TTS_AUDIO_CACHE.pop(next(iter(_TTS_AUDIO_CACHE)), None)
                _TTS_AUDIO_CACHE[cache_key] = audio_bytes
                return HttpResponse(audio_bytes, content_type="audio/mp3")
        except Exception:
            pass

    # 2. SET Neural edge_tts Engine (Free high quality neural synthesis)
    try:
        async def _synthesize():
            try:
                communicate = edge_tts.Communicate(clean_text, voice=voice_name, pitch="+2Hz", rate="+5%")
                audio_stream = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_stream.extend(chunk["data"])
                return bytes(audio_stream)
            except Exception:
                # Fallback to Emily Neural if Neerja has temporary connectivity hiccup
                communicate = edge_tts.Communicate(clean_text, voice="en-IE-EmilyNeural", pitch="+1Hz", rate="+4%")
                audio_stream = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_stream.extend(chunk["data"])
                return bytes(audio_stream)

        audio_bytes = asyncio.run(_synthesize())
        if len(_TTS_AUDIO_CACHE) >= _MAX_TTS_CACHE_ENTRIES:
            _TTS_AUDIO_CACHE.pop(next(iter(_TTS_AUDIO_CACHE)), None)
        _TTS_AUDIO_CACHE[cache_key] = audio_bytes
        return HttpResponse(audio_bytes, content_type="audio/mpeg")
    except Exception as exc:
        return JsonResponse({"status": "error", "error": str(exc)}, status=500)


@login_required
def export_tax_audit_pdf(request):
    """Exports official CA / Tax Audit summary PDF for the user."""
    from tracker.reports import generate_tax_audit_pdf

    account_id = request.GET.get("account")
    parsed_acc_id = None
    if account_id:
        try:
            parsed_acc_id = int(account_id)
        except (ValueError, TypeError):
            parsed_acc_id = None

    pdf_bytes = generate_tax_audit_pdf(request.user, account_id=parsed_acc_id)
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="SET_Tax_Audit_Summary.pdf"'
    return response


@login_required
def export_transactions_csv(request):
    """Exports filtered transactions to CSV format."""
    import csv

    base = Transaction.objects.filter(user=request.user).select_related("account", "statement")
    ctx = _filter_context(request, request.user)
    qs = apply_transaction_filters(base, ctx["filters"])

    response = HttpResponse(content_type="text/csv; charset=utf-8")
    response["Content-Disposition"] = 'attachment; filename="SET_Transactions_Export.csv"'

    writer = csv.writer(response)
    writer.writerow([
        "Transaction ID", "Date", "Bank", "Account Number",
        "Narration / Description", "Company / Vendor / Customer",
        "Category", "AI Confidence",
        "Debit (INR)", "Credit (INR)", "Balance (INR)",
    ])

    for t in qs:
        writer.writerow([
            t.id,
            t.txn_date.isoformat() if t.txn_date else "",
            t.account.bank_name if t.account else "Default Bank",
            t.account.display_account if t.account else "N/A",
            t.description,
            t.entity_name,
            t.category,
            f"{float(t.confidence or 0.0):.2f}",
            float(t.debit or 0.0),
            float(t.credit or 0.0),
            float(t.balance) if t.balance is not None else "",
        ])
    return response


@login_required
def export_transactions_excel(request):
    """Exports filtered transactions to Excel (.xlsx) format."""
    import io
    import pandas as pd

    base = Transaction.objects.filter(user=request.user).select_related("account", "statement")
    ctx = _filter_context(request, request.user)
    qs = apply_transaction_filters(base, ctx["filters"])

    rows = []
    for t in qs:
        rows.append({
            "Transaction ID": t.id,
            "Date": t.txn_date.isoformat() if t.txn_date else "",
            "Bank": t.account.bank_name if t.account else "Default Bank",
            "Account Number": t.account.display_account if t.account else "N/A",
            "Narration / Description": t.description,
            "Company / Vendor / Customer": t.entity_name,
            "Category": t.category,
            "AI Confidence": round(float(t.confidence or 0.0), 2),
            "Debit (INR)": float(t.debit or 0.0),
            "Credit (INR)": float(t.credit or 0.0),
            "Balance (INR)": float(t.balance) if t.balance is not None else None,
        })

    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Transactions")

    response = HttpResponse(
        buffer.getvalue(),
        content_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
    response["Content-Disposition"] = 'attachment; filename="SET_Transactions_Export.xlsx"'
    return response
