"""Minimal, dependency-free HTML escaping and self-contained-page helpers.

No templating engine, no remote CDN, no network access. Every untrusted
text field (instrument names, warnings, provenance notes, exception
messages, configuration labels) must be passed through :func:`escape`
before being embedded in generated HTML.
"""

from __future__ import annotations

import html as _html


def escape(value: object) -> str:
    """HTML-escape ``value`` (converted to ``str`` first) — the only safe
    way any untrusted text reaches a generated report."""
    return _html.escape(str(value), quote=True)


def wrap_page(title: str, body_html: str, *, style: str = "") -> str:
    """Wrap ``body_html`` in a minimal, self-contained HTML document.

    ``title`` is escaped; ``body_html``/``style`` are assumed already
    safe (built exclusively from :func:`escape`-passed fragments and
    literal markup by the caller). No external stylesheet, script, or
    font reference is ever included.
    """
    return (
        "<!doctype html>\n"
        "<html lang=\"en\">\n<head>\n"
        f"<meta charset=\"utf-8\">\n<title>{escape(title)}</title>\n"
        f"<style>{style}</style>\n</head>\n<body>\n{body_html}\n</body>\n</html>\n"
    )
