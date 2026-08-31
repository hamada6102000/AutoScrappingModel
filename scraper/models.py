"""
Domain models for the Self-Healing Web Scraper.

Each model maps onto a data feature described in the research proposal
(Section 4.2, Table 4.1: Dataset Features). Comments note the mapping.
"""

from django.db import models
from django.utils import timezone


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class Category(models.TextChoices):
    ECOMMERCE = "ecommerce", "E-commerce"
    NEWS = "news", "News / Media"
    FINANCE = "finance", "Finance"
    JOBS = "jobs", "Job Listings"
    ACADEMIC = "academic", "Academic"
    OTHER = "other", "Other"


class FieldType(models.TextChoices):
    STRING = "string", "String"
    NUMBER = "number", "Number"
    PRICE = "price", "Price"
    DATE = "date", "Date"
    URL = "url", "URL"
    MIXED = "mixed", "Mixed"


class RuleSource(models.TextChoices):
    HEURISTIC = "heuristic", "Heuristic"
    LLM = "llm", "LLM"
    HUMAN = "human", "Human"


class RunStatus(models.TextChoices):
    PENDING = "pending", "Pending"
    RUNNING = "running", "Running"
    SUCCESS = "success", "Success"
    HEALED = "healed", "Healed"
    DEGRADED = "degraded", "Degraded"
    FAILED = "failed", "Failed"


class ChangeType(models.TextChoices):
    NATURAL = "natural", "Natural"
    SIMULATED = "simulated", "Simulated"


class Resolution(models.TextChoices):
    AUTO_HEALED = "auto_healed", "Auto-Healed"
    UNRESOLVED = "unresolved", "Unresolved"


class LLMPurpose(models.TextChoices):
    INITIAL = "initial", "Initial Generation"
    HEALING = "healing", "Self-Healing"


# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------
class ScrapeTarget(models.Model):
    """A website to be scraped continuously (proposal Section 4.1)."""

    name = models.CharField(max_length=200)
    url = models.URLField(max_length=1000)
    category = models.CharField(
        max_length=20, choices=Category.choices, default=Category.OTHER
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["name"]

    def __str__(self) -> str:
        return self.name

    @property
    def active_rules(self):
        return XPathRule.objects.filter(field__target=self, is_active=True)

    @property
    def latest_run(self):
        return self.runs.order_by("-started_at").first()

    @property
    def latest_health(self):
        run = self.latest_run
        return run.health_score if run else None


class TargetField(models.Model):
    """
    A single data field to extract for a target
    (e.g. "title", "price"). Feature: target schema definition.
    """

    target = models.ForeignKey(
        ScrapeTarget, on_delete=models.CASCADE, related_name="fields"
    )
    name = models.SlugField(max_length=100)
    description = models.CharField(
        max_length=300,
        blank=True,
        help_text="Natural-language hint given to the LLM to locate this field.",
    )
    field_type = models.CharField(
        max_length=20, choices=FieldType.choices, default=FieldType.STRING
    )
    is_required = models.BooleanField(default=True)
    # Ground-Truth Labels (Table 4.1) — reference value for F1 computation.
    ground_truth = models.CharField(max_length=1000, blank=True)
    order = models.PositiveIntegerField(default=0)

    class Meta:
        ordering = ["order", "id"]
        unique_together = [("target", "name")]

    def __str__(self) -> str:
        return f"{self.target.name}:{self.name}"

    @property
    def active_rule(self):
        return self.rules.filter(is_active=True).order_by("-version").first()


# ---------------------------------------------------------------------------
# Extraction rules (Generated XPaths — Table 4.1)
# ---------------------------------------------------------------------------
class XPathRule(models.Model):
    """A versioned XPath expression for one field."""

    field = models.ForeignKey(
        TargetField, on_delete=models.CASCADE, related_name="rules"
    )
    xpath = models.CharField(max_length=2000)
    version = models.PositiveIntegerField(default=1)
    is_active = models.BooleanField(default=True)
    source = models.CharField(
        max_length=20, choices=RuleSource.choices, default=RuleSource.LLM
    )
    # Structural fingerprint of the DOM the rule was generated against.
    dom_signature = models.CharField(max_length=64, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["field", "-version"]

    def __str__(self) -> str:
        return f"{self.field.name} v{self.version}: {self.xpath}"


# ---------------------------------------------------------------------------
# Runs and results
# ---------------------------------------------------------------------------
class ScrapeRun(models.Model):
    """
    One execution of the pipeline against a target. Aggregates the
    run-level metrics: Health Score, HTTP Status, LLM Token Usage,
    Scrape Timestamp (Table 4.1).
    """

    target = models.ForeignKey(
        ScrapeTarget, on_delete=models.CASCADE, related_name="runs"
    )
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(
        max_length=20, choices=RunStatus.choices, default=RunStatus.PENDING
    )
    http_status = models.IntegerField(null=True, blank=True)

    # Health metrics (proposal RQ1 — structural divergence metrics).
    health_score = models.FloatField(default=0.0)
    xpath_failure_rate = models.FloatField(default=0.0)
    field_count_mismatch = models.IntegerField(default=0)
    structural_distance = models.FloatField(
        default=0.0,
        help_text="Normalised tree-edit / structural distance vs. the last DOM.",
    )

    fields_total = models.IntegerField(default=0)
    fields_extracted = models.IntegerField(default=0)

    # Self-healing bookkeeping.
    heal_attempts = models.IntegerField(default=0)
    healed = models.BooleanField(default=False)

    # Raw HTML Snapshot fingerprints (Table 4.1) — stored as hashes to keep
    # the DB lean; full snapshots can be persisted to disk if required.
    raw_html_signature = models.CharField(max_length=64, blank=True)
    dom_signature = models.CharField(max_length=64, blank=True)
    # Compact tag-only DOM skeleton, used to compute the structural (tree-edit)
    # distance against the previous run during change detection (RQ1).
    dom_skeleton = models.TextField(blank=True)

    # LLM Token Usage (Table 4.1).
    token_usage = models.IntegerField(default=0)

    # Regeneration latency (proposal RQ4 / Table 5.1).
    duration_ms = models.IntegerField(default=0)
    heal_latency_ms = models.IntegerField(default=0)

    log = models.JSONField(default=list, blank=True)

    class Meta:
        ordering = ["-started_at"]

    def __str__(self) -> str:
        return f"Run #{self.pk} {self.target.name} [{self.status}]"

    @property
    def below_threshold(self) -> bool:
        from django.conf import settings

        return self.health_score < settings.SCRAPER["HEALTH_THRESHOLD"]

    @property
    def health_percent(self) -> int:
        return round(self.health_score * 100)


class ExtractedRecord(models.Model):
    """
    Extracted Values (Table 4.1) — one value per field per run, including
    both the raw and normalised representations for F1 comparison.
    """

    run = models.ForeignKey(
        ScrapeRun, on_delete=models.CASCADE, related_name="records"
    )
    field = models.ForeignKey(TargetField, on_delete=models.CASCADE)
    xpath_used = models.ForeignKey(
        XPathRule, on_delete=models.SET_NULL, null=True, blank=True
    )
    raw_value = models.TextField(blank=True)
    normalized_value = models.TextField(blank=True)
    success = models.BooleanField(default=False)
    # Match against ground truth, when available (for F1 tracking).
    matches_ground_truth = models.BooleanField(null=True, blank=True)

    class Meta:
        ordering = ["field__order", "id"]

    def __str__(self) -> str:
        return f"{self.field.name} = {self.normalized_value!r}"


class ChangeEvent(models.Model):
    """
    Layout-Change Events (Table 4.1) — a detected structural change and its
    resolution status. This structured log is the primary input for the
    self-healing evaluation metrics (proposal Section 4.3, step 6).
    """

    target = models.ForeignKey(
        ScrapeTarget, on_delete=models.CASCADE, related_name="change_events"
    )
    run = models.ForeignKey(
        ScrapeRun, on_delete=models.CASCADE, related_name="change_events"
    )
    detected_at = models.DateTimeField(default=timezone.now)
    change_type = models.CharField(
        max_length=20, choices=ChangeType.choices, default=ChangeType.NATURAL
    )
    resolution = models.CharField(
        max_length=20, choices=Resolution.choices, default=Resolution.UNRESOLVED
    )
    affected_fields = models.JSONField(default=list, blank=True)
    structural_distance = models.FloatField(default=0.0)
    time_to_recovery_ms = models.IntegerField(default=0)

    class Meta:
        ordering = ["-detected_at"]

    def __str__(self) -> str:
        return f"Change on {self.target.name} [{self.resolution}]"


class FailureRecord(models.Model):
    """
    XPath Failure Logs (Table 4.1). Recent failures form the "failure-history
    memory" that is fed back to the LLM during regeneration (proposal 1.4 /
    Self-Healing Layer).
    """

    field = models.ForeignKey(
        TargetField, on_delete=models.CASCADE, related_name="failures"
    )
    run = models.ForeignKey(
        ScrapeRun, on_delete=models.CASCADE, related_name="failures"
    )
    failed_xpath = models.CharField(max_length=2000, blank=True)
    reason = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"FAIL {self.field.name}: {self.reason}"


class LLMCallLog(models.Model):
    """
    Record of each LLM invocation for cost / latency analysis
    (LLM Token Usage — Table 4.1; RQ4 latency scaling).
    """

    run = models.ForeignKey(
        ScrapeRun, on_delete=models.CASCADE, related_name="llm_calls"
    )
    provider = models.CharField(max_length=40)
    model = models.CharField(max_length=100, blank=True)
    purpose = models.CharField(
        max_length=20, choices=LLMPurpose.choices, default=LLMPurpose.INITIAL
    )
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    latency_ms = models.IntegerField(default=0)
    success = models.BooleanField(default=True)
    error = models.CharField(max_length=300, blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ["-created_at"]

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __str__(self) -> str:
        return f"{self.provider}/{self.model} [{self.purpose}]"
