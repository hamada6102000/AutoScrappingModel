"""Django admin registration for scraper models."""

from django.contrib import admin

from . import models


class TargetFieldInline(admin.TabularInline):
    model = models.TargetField
    extra = 1


@admin.register(models.ScrapeTarget)
class ScrapeTargetAdmin(admin.ModelAdmin):
    list_display = ("name", "category", "url", "is_active", "created_at")
    list_filter = ("category", "is_active")
    search_fields = ("name", "url")
    inlines = [TargetFieldInline]


@admin.register(models.TargetField)
class TargetFieldAdmin(admin.ModelAdmin):
    list_display = ("target", "name", "field_type", "is_required", "order")
    list_filter = ("field_type", "is_required")
    search_fields = ("name", "target__name")


@admin.register(models.XPathRule)
class XPathRuleAdmin(admin.ModelAdmin):
    list_display = ("field", "version", "is_active", "source", "xpath", "created_at")
    list_filter = ("is_active", "source")
    search_fields = ("xpath", "field__name")


@admin.register(models.ScrapeRun)
class ScrapeRunAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "target",
        "status",
        "health_score",
        "fields_extracted",
        "fields_total",
        "healed",
        "started_at",
    )
    list_filter = ("status", "healed")
    date_hierarchy = "started_at"


@admin.register(models.ExtractedRecord)
class ExtractedRecordAdmin(admin.ModelAdmin):
    list_display = ("run", "field", "normalized_value", "success")
    list_filter = ("success",)


@admin.register(models.ChangeEvent)
class ChangeEventAdmin(admin.ModelAdmin):
    list_display = (
        "target",
        "change_type",
        "resolution",
        "structural_distance",
        "time_to_recovery_ms",
        "detected_at",
    )
    list_filter = ("change_type", "resolution")


@admin.register(models.FailureRecord)
class FailureRecordAdmin(admin.ModelAdmin):
    list_display = ("field", "run", "reason", "created_at")
    search_fields = ("field__name", "reason")


@admin.register(models.LLMCallLog)
class LLMCallLogAdmin(admin.ModelAdmin):
    list_display = (
        "run",
        "provider",
        "model",
        "purpose",
        "prompt_tokens",
        "completion_tokens",
        "latency_ms",
        "success",
    )
    list_filter = ("provider", "purpose", "success")
