import hashlib

from django.contrib.auth.models import User
from django.db import models


class BankAccount(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="bank_accounts")
    bank_name = models.CharField(max_length=120)
    account_holder = models.CharField(max_length=160)
    account_number = models.CharField(max_length=32)
    display_account = models.CharField(max_length=32)
    fingerprint = models.CharField(max_length=64, db_index=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("user", "fingerprint")
        ordering = ["bank_name", "account_holder"]

    def __str__(self):
        return f"{self.bank_name} · {self.account_holder} · {self.display_account}"


class UploadBatch(models.Model):
    class Status(models.TextChoices):
        PROCESSING = "processing", "Processing"
        EXTRACTED = "extracted", "Extracted"
        ANALYZED = "analyzed", "Analyzed"
        ERROR = "error", "Error"

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="upload_batches")
    status = models.CharField(max_length=20, choices=Status.choices, default=Status.PROCESSING)
    error_message = models.TextField(blank=True)
    file_count = models.PositiveIntegerField(default=0)
    transaction_count = models.PositiveIntegerField(default=0)
    duplicates_skipped = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]


class StatementFile(models.Model):
    batch = models.ForeignKey(UploadBatch, on_delete=models.CASCADE, related_name="files")
    account = models.ForeignKey(
        BankAccount, on_delete=models.SET_NULL, null=True, blank=True, related_name="statements"
    )
    file = models.FileField(upload_to="uploads/%Y/%m/")
    original_name = models.CharField(max_length=255)
    header_excerpt = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)


class Transaction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="transactions")
    account = models.ForeignKey(
        BankAccount, on_delete=models.SET_NULL, null=True, blank=True, related_name="transactions"
    )
    batch = models.ForeignKey(
        UploadBatch, on_delete=models.SET_NULL, null=True, blank=True, related_name="transactions"
    )
    statement = models.ForeignKey(
        StatementFile, on_delete=models.SET_NULL, null=True, blank=True, related_name="transactions"
    )
    txn_date = models.DateField(null=True, blank=True)
    description = models.TextField()
    debit = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    credit = models.DecimalField(max_digits=14, decimal_places=2, default=0)
    balance = models.DecimalField(max_digits=14, decimal_places=2, null=True, blank=True)
    category = models.CharField(max_length=40, default="Other")
    confidence = models.FloatField(default=0)
    raw_json = models.JSONField(default=dict, blank=True)
    fingerprint = models.CharField(max_length=64, db_index=True, default="")
    is_manual = models.BooleanField(default=False)
    parent_transaction = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True, related_name="child_expenses"
    )
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["txn_date", "id"]
        unique_together = ("user", "fingerprint")

    @staticmethod
    def build_fingerprint(account_id, txn_date, description, debit, credit, is_manual=False):
        """
        Identifies a transaction well enough to catch re-uploads of the same
        (or overlapping) statement, without needing a bank-issued transaction
        ID (which most statements don't provide). Same account + date +
        description + amounts = treated as the same transaction.
        """
        import uuid
        if is_manual:
            return hashlib.sha256(f"manual|{uuid.uuid4()}|{account_id}|{txn_date}|{description}|{debit}|{credit}".encode()).hexdigest()
        raw = f"{account_id}|{txn_date}|{(description or '').strip().lower()}|{debit}|{credit}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def save(self, *args, **kwargs):
        if not self.fingerprint:
            self.fingerprint = self.build_fingerprint(
                self.account_id, self.txn_date, self.description, self.debit, self.credit, self.is_manual
            )
        super().save(*args, **kwargs)

    @property
    def entity_name(self) -> str:
        """Name of the company, vendor, merchant, or customer."""
        from ai_engine.entity_extractor import extract_entity_name
        return extract_entity_name(self.description, self.category)

    @property
    def counterparty(self) -> str:
        return self.entity_name


class CategoryRule(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="category_rules", null=True, blank=True)
    keyword = models.CharField(max_length=200, db_index=True)
    category = models.CharField(max_length=40)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]
        unique_together = ("user", "keyword")

    def __str__(self):
        return f"Rule: '{self.keyword}' -> {self.category} ({self.user})"

