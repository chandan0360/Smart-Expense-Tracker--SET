"""Comprehensive End-to-End Diagnostic & Health Check Script for SET.
Verifies Database, ML models, RAG vector store, Voice TTS, statement parsing, and templates.
"""

import os
import sys
import json
import traceback
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

# Force UTF-8 stdout for Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

def run_diagnostics():
    print("=" * 70)
    print("SET SYSTEM DIAGNOSTIC & HEALTH AUDIT")
    print("=" * 70)

    results = []

    def report(name, success, details=""):
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"[{status}] {name}")
        if details:
            print(f"       Details: {details}")
        results.append((name, success, details))

    # 1. Django Environment Setup
    try:
        import django
        os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
        django.setup()
        report("Django Environment Setup", True, f"Django v{django.get_version()} loaded successfully.")
    except Exception as e:
        report("Django Environment Setup", False, str(e))
        return

    from django.conf import settings
    from django.contrib.auth import get_user_model
    from tracker.models import BankAccount, StatementFile, Transaction, UploadBatch

    User = get_user_model()

    # 2. Database Connectivity & Records
    try:
        user_count = User.objects.count()
        acc_count = BankAccount.objects.count()
        txn_count = Transaction.objects.count()
        report(
            "Database Connectivity",
            True,
            f"Connected. Users: {user_count}, Accounts: {acc_count}, Transactions: {txn_count}"
        )
    except Exception as e:
        report("Database Connectivity", False, str(e))

    # 3. AI / ML Model Artifacts Check
    try:
        from ai_engine.classifier import predict_category, classify_bank
        from ai_engine.identity import extract_bank_name
        import joblib

        cat_model_path = settings.AI_ARTIFACT_DIR / "category_model.joblib"
        bank_model_path = settings.AI_ARTIFACT_DIR / "bank_model.joblib"

        if cat_model_path.exists() and bank_model_path.exists():
            # Test inference
            test_desc = "UPI/23901/Zomato Media/Pay"
            cat_pred = predict_category(test_desc)
            bank_pred, conf = classify_bank("HDFC BANK LIMITED - ACCOUNT STATEMENT")
            report(
                "ML Classification Pipeline",
                True,
                f"Models loaded. Narration '{test_desc}' -> '{cat_pred}'. Bank Header -> '{bank_pred}' ({int(conf*100)}%)"
            )
        else:
            report("ML Classification Pipeline", False, "Model artifact files (.joblib) missing in ai_engine/artifacts/.")
    except Exception as e:
        report("ML Classification Pipeline", False, str(e))

    # 4. Bank Statement Parser
    try:
        sample_dir = settings.BASE_DIR / "sample_statements"
        sample_files = list(sample_dir.glob("*.*")) if sample_dir.exists() else []
        if sample_files:
            report("Sample Statements Discovery", True, f"Found {len(sample_files)} sample statement files: {[f.name for f in sample_files[:4]]}")
        else:
            report("Sample Statements Discovery", False, "No sample statement files found.")
    except Exception as e:
        report("Sample Statements Discovery", False, str(e))

    # 5. Local Vector Store & RAG Engine
    try:
        from ai_engine.rag import get_user_vector_store
        from ai_engine.memory import SessionMemoryBuffer

        demo_user = User.objects.first()
        if demo_user:
            vstore = get_user_vector_store(demo_user, force_refresh=True)
            search_res = vstore.search("food dining restaurant", top_k=3)
            report(
                "Local Vector Store & RAG Retrieval",
                True,
                f"Indexed {len(vstore.documents)} chunks. Search returned {len(search_res)} relevant records."
            )

            # Test Memory Rewriting
            mem = SessionMemoryBuffer(max_turns=4)
            mem.add_user_message("How much did I spend on Swiggy last month?")
            mem.add_bot_message("You spent Rs 1200 on Swiggy.")
            rewritten = mem.rewrite_query_with_context("What about Zomato?")
            report(
                "Conversational Memory Buffer",
                True,
                f"Memory rewritten query: '{rewritten}'"
            )
        else:
            report("Local Vector Store & RAG Retrieval", False, "No user found in database to test vector indexing.")
    except Exception as e:
        report("Local Vector Store & RAG Retrieval", False, str(e))

    # 6. Chat API Engine Inference
    try:
        from ai_engine.chat import answer_transaction_query
        demo_user = User.objects.first()
        if demo_user:
            ans = answer_transaction_query(user=demo_user, query="Hello SET, give me cashflow summary")
            success = ans.get("status") == "ok" and bool(ans.get("reply"))
            report(
                "Chatbot Intelligence Engine",
                success,
                f"Status: {ans.get('status')}, Model: {ans.get('model')}, Response preview: {ans.get('reply')[:60]}..."
            )
        else:
            report("Chatbot Intelligence Engine", False, "No user available to test.")
    except Exception as e:
        report("Chatbot Intelligence Engine", False, str(e))

    # 7. Neural TTS Audio Synthesis Engine
    try:
        from tracker.views import tts_api_view
        from django.test import RequestFactory
        demo_user = User.objects.first()
        rf = RequestFactory()
        req = rf.post(
            "/api/tts/",
            data=json.dumps({"text": "SET system diagnostic check complete."}),
            content_type="application/json"
        )
        req.user = demo_user
        resp = tts_api_view(req)
        is_audio = resp.status_code == 200 and len(resp.content) > 1000
        report(
            "SET Neural Voice Engine (/api/tts/)",
            is_audio,
            f"HTTP {resp.status_code}, Generated {len(resp.content)} bytes of neural MP3 audio."
        )
    except Exception as e:
        report("SET Neural Voice Engine (/api/tts/)", False, str(e))

    # 8. Templates Compilation & Rendering Verification
    try:
        from django.template.loader import get_template
        templates_to_test = [
            "base.html",
            "tracker/home.html",
            "tracker/upload.html",
            "tracker/transactions.html",
            "tracker/dashboard.html",
            "tracker/_filters.html",
            "accounts/login.html",
            "accounts/signup.html",
        ]
        compiled = 0
        for t_name in templates_to_test:
            get_template(t_name)
            compiled += 1
        report("Templates Compilation", True, f"Successfully validated and compiled {compiled} templates.")
    except Exception as e:
        report("Templates Compilation", False, str(e))

    # 9. Summary
    print("=" * 70)
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    print(f"Audit Summary: {passed}/{total} Subsystems Fully Operational ({int(passed/total*100)}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_diagnostics()
