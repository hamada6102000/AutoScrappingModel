"""
Views that serve the self-contained demo product page.

The scraper points at ``/demo/product/``; flipping the active version (via the
dashboard "Simulate layout change" action or the ``simulate_change`` command)
changes the HTML served here, so the next scrape observes a real structural
change and must self-heal.
"""

from __future__ import annotations

from django.http import HttpResponse

from . import state


def product(request):
    """Serve whichever version is currently 'live'."""
    version = state.get_active_version()
    html = state.load_fixture(version)
    return HttpResponse(html)


def product_version(request, version: str):
    """Serve a specific version explicitly (useful for inspection/tests)."""
    try:
        html = state.load_fixture(version)
    except ValueError:
        return HttpResponse("Unknown version", status=404)
    return HttpResponse(html)
