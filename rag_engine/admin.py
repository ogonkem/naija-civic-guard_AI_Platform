from django.contrib import admin

from .models import EvalResult, RequestMetric


@admin.register(RequestMetric)
class RequestMetricAdmin(admin.ModelAdmin):
    list_display = (
        "timestamp", "short_id", "provider", "model",
        "embedding_time_ms", "retrieval_time_ms", "generation_time_ms", "total_time_ms",
        "tokens_generated", "is_estimate", "tokens_per_second", "errored",
    )
    list_filter = ("provider", "model", "tokens_generated_is_estimate", "timestamp")
    search_fields = ("request_id", "query_text", "error")
    date_hierarchy = "timestamp"
    ordering = ("-timestamp",)

    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.model._meta.fields]

    @admin.display(description="request_id")
    def short_id(self, obj):
        return str(obj.request_id)[:8]

    @admin.display(boolean=True, description="est?")
    def is_estimate(self, obj):
        return obj.tokens_generated_is_estimate

    @admin.display(boolean=True, description="err")
    def errored(self, obj):
        return bool(obj.error)


@admin.register(EvalResult)
class EvalResultAdmin(admin.ModelAdmin):
    list_display = (
        "created_at", "short_id", "matched_ground_truth", "keyword_source",
        "keyword_coverage", "hit", "reciprocal_rank", "target_section",
        "response_chars",
    )
    list_filter = ("matched_ground_truth", "keyword_source", "hit", "created_at")
    search_fields = ("request_id",)
    date_hierarchy = "created_at"
    ordering = ("-created_at",)

    def get_readonly_fields(self, request, obj=None):
        return [f.name for f in self.model._meta.fields]

    @admin.display(description="request_id")
    def short_id(self, obj):
        return str(obj.request_id)[:8]
