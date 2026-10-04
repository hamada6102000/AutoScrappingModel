"""
Django settings for the Self-Healing Web Scraper.

Project: SELF-HEALING WEB SCRAPING VIA LLM-DRIVEN XPATH REGENERATION
A reference implementation of the four-layer, closed-loop self-healing
scraping architecture described in the research proposal.
"""

from pathlib import Path
import json
import os

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from a .env file if present (see .env.example).
load_dotenv(BASE_DIR / ".env")


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Core Django
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv(
    "DJANGO_SECRET_KEY",
    "dev-insecure-key-change-me-in-production-0123456789abcdef",
)
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = os.getenv("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1,testserver").split(",")

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    # Local apps
    "scraper",
    "dashboard",
    "demopages",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        # Override with DJANGO_DB_PATH, e.g. to open the database from an
        # exported results bundle and recompute the report.
        "NAME": os.getenv("DJANGO_DB_PATH") or (BASE_DIR / "db.sqlite3"),
    }
}

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LANGUAGE_CODE = "en-us"
TIME_ZONE = "UTC"
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MESSAGE_STORAGE = "django.contrib.messages.storage.session.SessionStorage"

# ---------------------------------------------------------------------------
# Self-Healing Scraper configuration
# ---------------------------------------------------------------------------
# These map directly onto the thresholds and targets defined in the proposal
# (Section 5.2, Table 5.1: Target Performance Metrics).
SCRAPER = {
    # Which LLM provider drives XPath generation / regeneration.
    # Options: "heuristic" (offline, zero-config default), "groq" (free),
    #          "anthropic", "openai", "ollama".
    "LLM_PROVIDER": os.getenv("SCRAPER_LLM_PROVIDER", "heuristic"),

    # Groq (free, fast LLaMA inference — get a free key at console.groq.com).
    "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    "GROQ_MODEL": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "GROQ_HOST": os.getenv("GROQ_HOST", "https://api.groq.com/openai/v1"),

    # Anthropic / Claude (paid API — separate from claude.ai subscription).
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "ANTHROPIC_MODEL": os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),

    # OpenAI (optional).
    "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", ""),
    "OPENAI_MODEL": os.getenv("OPENAI_MODEL", "gpt-4o"),

    # Ollama / local LLaMA (optional, free).
    "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://localhost:11434"),
    "OLLAMA_MODEL": os.getenv("OLLAMA_MODEL", "llama3.1"),

    # Health-score threshold below which the self-healing loop is triggered
    # (health score is clipped to [0, 1]).
    "HEALTH_THRESHOLD": env_float("SCRAPER_HEALTH_THRESHOLD", 0.80),

    # Maximum self-healing regeneration attempts per run before giving up
    # (keeps the closed loop bounded — proposal Section 3.1).
    "MAX_HEAL_ATTEMPTS": env_int("SCRAPER_MAX_HEAL_ATTEMPTS", 3),

    # Number of most-recent failures fed back to the LLM as failure-history
    # memory during regeneration (proposal Section 3.2: feedback-loop stability).
    "FAILURE_MEMORY_SIZE": env_int("SCRAPER_FAILURE_MEMORY_SIZE", 10),

    # HTTP fetch settings.
    "FETCH_TIMEOUT": env_int("SCRAPER_FETCH_TIMEOUT", 20),
    "USER_AGENT": os.getenv(
        "SCRAPER_USER_AGENT",
        "SelfHealingScraper/1.0 (+research; Fayoum University)",
    ),

    # Where snapshots, ground-truth sheets and evaluation results are written.
    "DATA_DIR": os.getenv("SCRAPER_DATA_DIR", str(BASE_DIR / "eval_data")),

    # Save the raw HTML of every live (non-evaluation) run as a snapshot.
    "SAVE_SNAPSHOTS": env_bool("SCRAPER_SAVE_SNAPSHOTS", True),

    # If the real LLM fails (rate limit, network, bad key), fall back to the
    # offline heuristic for that call. The evaluation turns this OFF so that
    # LLM results are never silently mixed with heuristic ones.
    "ALLOW_FALLBACK": env_bool("SCRAPER_ALLOW_FALLBACK", True),

    # Upper bound on the pruned HTML characters sent to the LLM. Anything cut
    # is flagged (llm_input_truncated) so it can be reported.
    "LLM_MAX_INPUT_CHARS": env_int("SCRAPER_LLM_MAX_INPUT_CHARS", 60000),

    # Rate-limit handling for remote LLM APIs (HTTP 429). Waiting time is
    # recorded separately and excluded from latency metrics.
    "LLM_MAX_RETRIES": env_int("SCRAPER_LLM_MAX_RETRIES", 6),
    "LLM_MAX_WAIT_SECONDS": env_int("SCRAPER_LLM_MAX_WAIT_SECONDS", 900),

    # Optional price table for cost estimates, USD per 1M tokens, e.g.
    # SCRAPER_PRICING='{"llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79}}'
    # Left empty, costs are reported as unknown (only token counts are used).
    "PRICING": json.loads(os.getenv("SCRAPER_PRICING", "{}") or "{}"),

    # Tokenizer used to count tokens of pages that are never sent to the LLM
    # (raw HTML). Falls back to a chars/4 estimate if unavailable, and the
    # choice is recorded with every snapshot.
    "TOKENIZER": os.getenv("SCRAPER_TOKENIZER", "cl100k_base"),
}
