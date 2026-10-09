from django.urls import path

from tracker import views

urlpatterns = [
    path("", views.home, name="home"),
    path("upload/", views.upload_view, name="upload"),
    path("transactions/", views.transactions_view, name="transactions"),
    path("transactions/<int:pk>/category/", views.update_category, name="update_category"),
    path("transactions/manual/", views.add_manual_transaction, name="add_manual_transaction"),
    path("clear/", views.clear_extracted, name="clear_extracted"),
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("dashboard/account/<int:pk>/", views.dashboard_account_view, name="dashboard_account"),
    path("dashboard/export/tax-audit/", views.export_tax_audit_pdf, name="export_tax_audit_pdf"),
    path("transactions/export/csv/", views.export_transactions_csv, name="export_transactions_csv"),
    path("transactions/export/excel/", views.export_transactions_excel, name="export_transactions_excel"),
    path("chat/", views.chat_api_view, name="chat_api"),
    path("api/tts/", views.tts_api_view, name="tts_api"),
]
