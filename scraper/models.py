"""
Domain models for the Self-Healing Web Scraper.

Each model maps onto a data feature described in the research proposal
(Section 4.2, Table 4.1: Dataset Features). Comments note the mapping.

Evaluation-specific models (Experiment, Snapshot, GroundTruth) make every
reported number traceable to a saved page snapshot, a human-verified label,
and an exact configuration (proposal Section 4.3; reproducibility).
"""

import json

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


class Cardinality(models.TextChoices):
    SINGLE = "single", "Single value"
    MANY = "many", "List of values"


class RuleSource(models.TextChoices):
    HEURISTIC = "heuristic", "Heuristic"
    LLM = "llm", "LLM"
    HUMAN = "human", "Human"


class RulePurpose(models.TextChoices):
    INITIAL = "initial", "Initial generation"
    REPAIR = "repair", "Validation repair"
    HEALING = "healing", "Self-healing"
    COPIED = "copied", "Copied (paired baseline)"


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
    NOT_ATTEMPTED = "not_attempted", "Not attempted (healing disabled)"


class LLMPurpose(models.TextChoices):
    INITIAL = "initial", "Initial Generation"
    REPAIR = "repair", "Validation Repair"
    HEALING = "healing", "Self-Healing"


class Condition:
    """Experimental conditions (free-text field; these are the standard ones)."""

    LIVE = ""                      # ordinary (non-experiment) runs
    SELF_HEALING = "self_healing"  # proposed system
    STATIC = "static"              # same LLM rules, healing disabled (RQ3)
    BASELINE_PREFIX = "baseline:"  # imported external baselines (O4)


class StepKind(models.TextChoices):
    LIVE = "live", "Live run"
    BASELINE = "baseline", "Baseline snapshot (t0)"
    CONTROL = "control", "Control (no change)"
    STRUCTURAL = "structural", "Simulated structural change"
    CONTENT = "content", "Simulated content-only change"
    NATURAL = "natural", "Natural snapshot"


class SnapshotKind(models.TextChoices):
    LIVE = "live", "Live fetch"
    BASELINE = "baseline", "Baseline"
    SIMULATED = "simulated", "Simulated change"
    NATURAL = "natural", "Natural (re-fetched later)"


class GTSource(models.TextChoices):
    HUMAN = "human", "Human-verified"
    DERIVED = "derived", "Derived (text preserved by construction)"
    PERTURBED = "perturbed", "Updated by construction (content perturbation)"


class RecordPhase(models.TextChoices):
    PRE_HEAL = "pre_heal", "Before healing"
    FINAL = "final", "Final"


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------
class Experiment(models.Model):
    """One evaluation run of the protocol, with its frozen configuration."""

    name = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(default=timezone.now)
    # Resolved configuration: provider, model, thresholds, seeds, protocol.
    config = models.JSONField(default=dict, blank=True)
    # Environment: Python/package versions, platform, tokenizer, git commit.
    environment = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Target configuration
# ---------------------------------------------------------------------------
class ScrapeTarget(models.Model):
    """A website to be scraped (proposal Section 4.1)."""

    name = models.CharField(max_length=200)
    url = models.URLField(max_length=1000)
    category = models.CharField(
        max_length=20, choices=Category.choices, default=Category.OTHER
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(default=timezone.now)

    # Evaluation bookkeeping (blank for ordinary dashboard targets).
    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, null=True, blank=True,
        related_name="targets",
    )
    site_key = models.CharField(max_length=100, blank=True)
    condition = models.CharField(max_length=60, blank=True, default="")
    repetition = models.PositiveIntegerField(null=True, blank=True)

    # Fetch options.
    render = models.CharField(
        max_length=10, default="static",
        help_text="'static' (HTTP fetch) or 'js' (headless browser render).",
    )
    html_file = models.CharField(
        max_length=500, blank=True,
        help_text="Optional local HTML file used instead of fetching the URL.",
    )
    prune_keep = models.JSONField(
        default=list, blank=True,
        help_text="Tag names to keep that the pruner removes by default.",
    )

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
    """A single data field to extract for a target (e.g. "title", "price")."""

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
    cardinality = models.CharField(
        max_length=10, choices=Cardinality.choices, default=Cardinality.SINGLE
    )
    is_required = models.BooleanField(default=True)
    # Static default ground truth (used only for ordinary live runs; the
    # evaluation uses per-snapshot GroundTruth instead).
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
# Snapshots and ground truth (reproducibility)
# ---------------------------------------------------------------------------
class Snapshot(models.Model):
    """
    A saved copy of a page (Raw HTML Snapshot — Table 4.1). Every evaluation
    run reads its input from a snapshot on disk, so results are reproducible
    and external baselines can be run on identical inputs.
    """

    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, null=True, blank=True,
        related_name="snapshots",
    )
    site_key = models.CharField(max_length=100, blank=True)
    kind = models.CharField(
        max_length=20, choices=SnapshotKind.choices, default=SnapshotKind.LIVE
    )
    label = models.CharField(max_length=50, blank=True)
    repetition = models.PositiveIntegerField(null=True, blank=True)
    step = models.PositiveIntegerField(null=True, blank=True)
    parent = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="children",
    )
    # Mutation spec for simulated changes (operator, seed, severity, focus).
    mutation = models.JSONField(default=dict, blank=True)

    url = models.URLField(max_length=1000, blank=True)
    fetched_at = models.DateTimeField(default=timezone.now)
    http_status = models.IntegerField(null=True, blank=True)
    path = models.CharField(max_length=500, help_text="Relative to SCRAPER['DATA_DIR'].")
    sha256 = models.CharField(max_length=64, blank=True)
    size_bytes = models.IntegerField(default=0)

    # Token accounting for the pruning evaluation (proposal §4.3 step 2).
    tokenizer = models.CharField(max_length=50, blank=True)
    raw_tokens = models.IntegerField(null=True, blank=True)
    clean_tokens = models.IntegerField(null=True, blank=True)
    llm_view_tokens = models.IntegerField(null=True, blank=True)
    llm_view_truncated = models.BooleanField(default=False)

    robots_allowed = models.BooleanField(null=True, blank=True)
    fetch_meta = models.JSONField(default=dict, blank=True)
    # Per field: is the ground-truth value present in the page / LLM view?
    field_presence = models.JSONField(default=dict, blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["site_key", "repetition", "step", "id"]

    def __str__(self) -> str:
        bits = [self.site_key or "snapshot", self.kind]
        if self.repetition is not None:
            bits.append(f"r{self.repetition}")
        if self.step is not None:
            bits.append(f"s{self.step}")
        return ":".join(bits)

    @property
    def token_reduction(self):
        if not self.raw_tokens or self.llm_view_tokens is None:
            return None
        return 1.0 - (self.llm_view_tokens / self.raw_tokens)


class GroundTruth(models.Model):
    """
    Ground-Truth Labels (Table 4.1), per snapshot and field. Baseline and
    natural snapshots are labelled by a human; simulated snapshots inherit the
    labels of their parent because mutations preserve visible text, or have
    them updated by construction for content perturbations.
    """

    snapshot = models.ForeignKey(
        Snapshot, on_delete=models.CASCADE, related_name="ground_truth"
    )
    field_name = models.SlugField(max_length=100)
    cardinality = models.CharField(
        max_length=10, choices=Cardinality.choices, default=Cardinality.SINGLE
    )
    # Single value as text; lists stored as a JSON array string.
    value = models.TextField(blank=True)
    source = models.CharField(
        max_length=20, choices=GTSource.choices, default=GTSource.HUMAN
    )
    verified_by = models.CharField(max_length=100, blank=True)
    notes = models.TextField(blank=True)
    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        unique_together = [("snapshot", "field_name")]
        ordering = ["snapshot", "field_name"]

    def __str__(self) -> str:
        return f"GT {self.snapshot}:{self.field_name}={self.value!r}"

    @property
    def values(self) -> list[str]:
        if self.cardinality == Cardinality.MANY:
            try:
                data = json.loads(self.value or "[]")
                return [str(v) for v in data if str(v).strip()]
            except (ValueError, TypeError):
                return [v.strip() for v in self.value.split("||") if v.strip()]
        alternatives = self.alternatives
        if alternatives is not None:
            return alternatives
        return [self.value] if self.value.strip() else []

    @property
    def alternatives(self) -> list[str] | None:
        """For a single-value field labelled with several acceptable forms
        (e.g. the same instant shown in two time zones): the list of forms,
        stored as {"any": [...]}; otherwise None."""
        if self.cardinality == Cardinality.MANY or not self.value.startswith('{"any"'):
            return None
        try:
            data = json.loads(self.value)
        except (ValueError, TypeError):
            return None
        return [str(v) for v in data.get("any", []) if str(v).strip()]

    @property
    def display_value(self) -> str:
        vals = self.values
        return " || ".join(vals) if (self.cardinality == Cardinality.MANY or self.alternatives) else self.value


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
    purpose = models.CharField(
        max_length=20, choices=RulePurpose.choices, default=RulePurpose.INITIAL
    )
    copied_from = models.ForeignKey(
        "self", on_delete=models.SET_NULL, null=True, blank=True
    )
    created_in_run = models.ForeignKey(
        "ScrapeRun", on_delete=models.SET_NULL, null=True, blank=True,
        related_name="rules_created",
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
    completed = models.BooleanField(default=False)
    error = models.TextField(blank=True)
    http_status = models.IntegerField(null=True, blank=True)

    # --- evaluation design ------------------------------------------------
    experiment = models.ForeignKey(
        Experiment, on_delete=models.CASCADE, null=True, blank=True,
        related_name="runs",
    )
    snapshot = models.ForeignKey(
        Snapshot, on_delete=models.SET_NULL, null=True, blank=True,
        related_name="runs",
    )
    condition = models.CharField(max_length=60, blank=True, default="")
    repetition = models.PositiveIntegerField(null=True, blank=True)
    step = models.PositiveIntegerField(null=True, blank=True)
    step_kind = models.CharField(
        max_length=20, choices=StepKind.choices, default=StepKind.LIVE
    )
    change_type = models.CharField(max_length=20, blank=True)
    # Design/human label: was there a structural change since the last run?
    true_change = models.BooleanField(null=True, blank=True)
    heal_enabled = models.BooleanField(default=True)

    # --- health (final state, after any healing) ---------------------------
    health_score = models.FloatField(default=0.0)
    xpath_failure_rate = models.FloatField(default=0.0)
    field_count_mismatch = models.IntegerField(default=0)
    structural_distance = models.FloatField(
        default=0.0,
        help_text="Weighted Jaccard distance of tag-path profiles vs. previous DOM.",
    )
    markup_distance = models.FloatField(
        default=0.0,
        help_text="Same, but paths include class/id tokens (sensitive to renames).",
    )
    fields_total = models.IntegerField(default=0)
    fields_extracted = models.IntegerField(default=0)

    # --- health before healing (detection signals, RQ1) --------------------
    detected = models.BooleanField(default=False)
    pre_heal_health = models.FloatField(null=True, blank=True)
    pre_heal_failure_rate = models.FloatField(null=True, blank=True)
    pre_heal_field_mismatch = models.IntegerField(null=True, blank=True)
    pre_heal_fields_extracted = models.IntegerField(null=True, blank=True)
    pre_heal_failed = models.JSONField(default=list, blank=True)

    # --- self-healing bookkeeping -----------------------------------------
    heal_attempts = models.IntegerField(default=0)
    healed = models.BooleanField(default=False)

    # --- snapshot fingerprints --------------------------------------------
    raw_html_signature = models.CharField(max_length=64, blank=True)
    dom_signature = models.CharField(max_length=64, blank=True)
    dom_skeleton = models.TextField(blank=True)  # legacy, kept for old rows
    dom_profile = models.JSONField(default=dict, blank=True)

    # --- LLM accounting (LLM Token Usage — Table 4.1) ----------------------
    provider = models.CharField(max_length=40, blank=True)
    model = models.CharField(max_length=100, blank=True)
    tokenizer = models.CharField(max_length=50, blank=True)
    llm_calls_count = models.IntegerField(default=0)
    llm_errors = models.IntegerField(default=0)
    fallback_used = models.BooleanField(default=False)
    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    token_usage = models.IntegerField(default=0)
    cost_usd = models.FloatField(null=True, blank=True)
    raw_tokens = models.IntegerField(null=True, blank=True)
    llm_view_tokens = models.IntegerField(null=True, blank=True)
    llm_input_truncated = models.BooleanField(default=False)

    # --- timing (proposal RQ4 / Table 5.1) ---------------------------------
    duration_ms = models.IntegerField(default=0)
    heal_latency_ms = models.IntegerField(default=0)
    rate_limit_wait_ms = models.IntegerField(default=0)

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
    Extracted Values (Table 4.1) — one row per field per run and phase, with
    raw and normalised values and the ground-truth comparison.
    """

    run = models.ForeignKey(
        ScrapeRun, on_delete=models.CASCADE, related_name="records"
    )
    field = models.ForeignKey(TargetField, on_delete=models.CASCADE)
    phase = models.CharField(
        max_length=20, choices=RecordPhase.choices, default=RecordPhase.FINAL
    )
    xpath_used = models.ForeignKey(
        XPathRule, on_delete=models.SET_NULL, null=True, blank=True
    )
    raw_value = models.TextField(blank=True)
    normalized_value = models.TextField(blank=True)
    values = models.JSONField(default=list, blank=True)  # list fields
    success = models.BooleanField(default=False)
    reason = models.CharField(max_length=60, blank=True)

    # Ground-truth comparison (filled when GT exists for the snapshot).
    gt_available = models.BooleanField(default=False)
    gt_value = models.TextField(blank=True)
    exact_match = models.BooleanField(null=True, blank=True)
    matches_ground_truth = models.BooleanField(null=True, blank=True)  # legacy alias
    token_precision = models.FloatField(null=True, blank=True)
    token_recall = models.FloatField(null=True, blank=True)
    token_f1 = models.FloatField(null=True, blank=True)
    list_tp = models.IntegerField(null=True, blank=True)
    list_fp = models.IntegerField(null=True, blank=True)
    list_fn = models.IntegerField(null=True, blank=True)

    class Meta:
        ordering = ["field__order", "id"]

    def __str__(self) -> str:
        return f"{self.field.name} = {self.normalized_value!r}"


class ChangeEvent(models.Model):
    """
    Layout-Change Events (Table 4.1) — a detected structural change and its
    resolution status. The primary input for self-healing evaluation metrics
    (proposal Section 4.3, step 6).
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
    mutation = models.JSONField(default=dict, blank=True)
    resolution = models.CharField(
        max_length=20, choices=Resolution.choices, default=Resolution.UNRESOLVED
    )
    affected_fields = models.JSONField(default=list, blank=True)
    fields_recovered = models.JSONField(default=list, blank=True)
    fields_unrecovered = models.JSONField(default=list, blank=True)
    structural_distance = models.FloatField(default=0.0)
    markup_distance = models.FloatField(default=0.0)
    attempts = models.IntegerField(default=0)
    failure_history_len = models.IntegerField(default=0)
    time_to_recovery_ms = models.IntegerField(default=0)
    time_to_recovery_excl_wait_ms = models.IntegerField(default=0)

    class Meta:
        ordering = ["-detected_at"]

    def __str__(self) -> str:
        return f"Change on {self.target.name} [{self.resolution}]"


class FailureRecord(models.Model):
    """
    XPath Failure Logs (Table 4.1). Recent failures form the "failure-history
    memory" fed back to the LLM during regeneration.
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
    One LLM (or heuristic) invocation: tokens, latency, rate-limit waits,
    fallback, failure-history length, and the raw response for auditing.
    """

    run = models.ForeignKey(
        ScrapeRun, on_delete=models.CASCADE, related_name="llm_calls"
    )
    provider = models.CharField(max_length=40)
    requested_provider = models.CharField(max_length=40, blank=True)
    fallback_used = models.BooleanField(default=False)
    model = models.CharField(max_length=100, blank=True)
    purpose = models.CharField(
        max_length=20, choices=LLMPurpose.choices, default=LLMPurpose.INITIAL
    )
    attempt = models.IntegerField(default=0)
    failure_history_len = models.IntegerField(default=0)
    fields_requested = models.IntegerField(default=0)
    xpaths_returned = models.IntegerField(default=0)
    xpaths_valid = models.IntegerField(default=0)

    prompt_tokens = models.IntegerField(default=0)
    completion_tokens = models.IntegerField(default=0)
    cost_usd = models.FloatField(null=True, blank=True)
    latency_ms = models.IntegerField(default=0)
    validation_ms = models.IntegerField(default=0)
    rate_limit_wait_ms = models.IntegerField(default=0)
    retries = models.IntegerField(default=0)
    input_chars = models.IntegerField(default=0)
    input_truncated = models.BooleanField(default=False)
    prompt_sha = models.CharField(max_length=64, blank=True)
    prompt_version = models.CharField(max_length=20, blank=True)
    raw_response = models.TextField(blank=True)

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
