from decimal import Decimal
from pathlib import Path

from django.db import transaction as db_transaction

from ai_engine.classifier import classify_category
from ai_engine.extractor import StatementParseError, parse_statement
from ai_engine.identity import identify_statement
from tracker.models import BankAccount, StatementFile, Transaction, UploadBatch


class PipelineError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def process_batch(batch: UploadBatch, password: str | None = None) -> UploadBatch:
    try:
        with db_transaction.atomic():
            total = 0
            duplicates_skipped = 0
            for statement in batch.files.all():
                path = Path(statement.file.path)
                parsed = parse_statement(path, password=password)
                identity = identify_statement(parsed.header_text, statement.original_name)
                account, _ = BankAccount.objects.get_or_create(
                    user=batch.user,
                    fingerprint=identity["fingerprint"],
                    defaults={
                        "bank_name": identity["bank_name"],
                        "account_holder": identity["account_holder"],
                        "account_number": identity["account_number"],
                        "display_account": identity["display_account"],
                    },
                )
                statement.account = account
                statement.header_excerpt = parsed.header_text[:2000]
                statement.save(update_fields=["account", "header_excerpt"])

                batch.status = UploadBatch.Status.EXTRACTED
                batch.save(update_fields=["status"])

                for row in parsed.transactions:
                    debit = Decimal(str(round(row.debit, 2)))
                    credit = Decimal(str(round(row.credit, 2)))
                    fingerprint = Transaction.build_fingerprint(
                        account.id, row.txn_date, row.description, debit, credit
                    )
                    if Transaction.objects.filter(user=batch.user, fingerprint=fingerprint).exists():
                        duplicates_skipped += 1
                        continue

                    category, confidence = classify_category(row.description, user=batch.user)
                    Transaction.objects.create(
                        user=batch.user,
                        account=account,
                        batch=batch,
                        statement=statement,
                        txn_date=row.txn_date,
                        description=row.description,
                        debit=debit,
                        credit=credit,
                        balance=None if row.balance is None else Decimal(str(round(row.balance, 2))),
                        category=category,
                        confidence=confidence,
                        raw_json=row.raw,
                        fingerprint=fingerprint,
                    )
                    total += 1

            if total == 0 and duplicates_skipped == 0:
                raise PipelineError("No transactions could be extracted from the uploaded files.")
            if total == 0 and duplicates_skipped > 0:
                raise PipelineError(
                    f"All {duplicates_skipped} transaction(s) in this upload already exist in your account — nothing new to add."
                )

            batch.status = UploadBatch.Status.ANALYZED
            batch.transaction_count = total
            batch.duplicates_skipped = duplicates_skipped
            batch.save(update_fields=["status", "transaction_count", "duplicates_skipped"])
            return batch
    except (StatementParseError, PipelineError) as exc:
        batch.status = UploadBatch.Status.ERROR
        batch.error_message = str(exc)
        batch.save(update_fields=["status", "error_message"])
        raise PipelineError(str(exc)) from exc
    except Exception as exc:
        batch.status = UploadBatch.Status.ERROR
        batch.error_message = f"Processing failed: {exc}"
        batch.save(update_fields=["status", "error_message"])
        raise PipelineError(batch.error_message) from exc
