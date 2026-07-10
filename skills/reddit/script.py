"""Reddit JSON API accessor for the Reddit skill.

Logs one structured INFO line per ``fetch_json`` call (logger
``reddit_client``) with the domain that ultimately served the request, the
final HTTP status, attempt count, and elapsed wall-clock time. The line
intentionally omits the proxy URL (it embeds the password) — only the host
piece is safe to surface, and that's already in DOMAINS.



Reddit's edge (Fastly + their own bot service) blocks scripted clients on
multiple layers — IP reputation, TLS fingerprint, request shape. Hitting
``www.reddit.com/.json`` from a default Python HTTP client returns 403 with
the new-Reddit HTML shell as the body. This module hides every workaround so
the skill script (``script.py``) can stay free of HTTP concerns and just call
``fetch_json(path, params)``.

The defenses applied here, layered from network up to application, are:

1. **Residential IP** via DataImpulse — datacenter ASNs are blocklisted at
   Reddit's edge before any HTTP byte is parsed. Required, not optional;
   ``build_proxy_url()`` raises if creds are missing.
2. **Chrome TLS fingerprint** via ``curl_cffi`` (``impersonate="chrome120"``)
   — Python's stdlib ``ssl`` produces a JA3 no real browser uses, and Fastly
   keys on it. ``curl_cffi`` ships the exact ClientHello bytes Chrome 120
   sends, including HTTP/2 SETTINGS order.
3. **Browser-realistic headers** — see the ``HEADERS`` block. Sending
   ``Accept: application/json`` or omitting ``Sec-Fetch-*`` flags the request
   even when the IP and TLS look fine.
4. **old.reddit.com first** — ``DOMAINS`` order. ``old.reddit.com`` doesn't
   share ``www.``'s bot-challenge route. ``www.`` is kept as a fallback.
5. **Connection reuse** via a singleton ``Session`` — keeps the same TCP/TLS
   connection across ``fetch_json`` calls, which means the same DataImpulse
   exit IP for the whole process. Reddit treats fresh-IP-per-request as a
   tell.
6. **HTML-at-200 treated as blocked** — Reddit's challenge page sometimes
   ships at status 200. We require ``content-type: application/json`` to
   accept a 200 as success.
"""
import json
import logging
import os
import secrets
import time

# Stdlib logging on purpose — this module is sandbox-bundled (uploaded to
# Daytona alongside script.py) and the sandbox runtime has no `app` package,
# so importing `app.logging_config.log_info` would `ModuleNotFoundError` at
# load time. The structured shape we need (`event` field, JSON payload) is
# emitted by hand below in `_emit`. Backend-side modules should still use
# `log_info` per the project rule in CLAUDE.md.
log = logging.getLogger("reddit_client")


# Anti-bot defense: try old.reddit.com before www.reddit.com.
# www.reddit.com routes through Fastly's bot-challenge layer which serves the
# new-Reddit HTML shell at status 403 (or sometimes 200) on .json endpoints
# when the request looks scripted. old.reddit.com is the legacy frontend and
# doesn't have that route. Live tests showed old.reddit at 8/8 success while
# www. was inconsistent depending on the residential IP slice. We keep www.
# as a fallback so a degraded old.reddit doesn't take the skill down.
DOMAINS = ("https://old.reddit.com", "https://www.reddit.com")


# Anti-bot defense: every header here matches Chrome's profile for an
# in-tab navigation (i.e. typing a URL in the address bar). Reddit's edge
# inspects these BEFORE the TLS fingerprint check fires, so they're cheap
# tells if wrong:
#   - Accept: text/html...        — sending application/json on a navigation
#                                   URL marks the client as scripted.
#   - Accept-Encoding: gzip/...   — real browsers always send this. Also
#                                   beneficial: DataImpulse meters wire
#                                   bytes, so compressed responses cost less.
#   - Cache-Control + Pragma      — sent by Chrome on hard reload; harmless
#                                   on first load and matches a real fp.
#   - Sec-Fetch-*                 — Chrome stamps these on every request
#                                   since 2019. Missing them is a strong
#                                   signal of an HTTP library, not a browser.
#   - Upgrade-Insecure-Requests   — sent on top-level navigation by Chrome.
HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

DEFAULT_TIMEOUT = 20

# 403/429/503 are retried on the same domain before falling back to the next
# one. 403 is in here (rather than non-retryable) because Reddit's bot
# challenge fires probabilistically — a retry on the same connection often
# succeeds even without changing anything.
RETRYABLE_STATUSES = (403, 429, 503)


class RedditError(Exception):
    """Raised when Reddit cannot be reached or returns a blocking response.

    Carries an HTTP-style ``status_code`` so the skill script can map it to
    a user-visible error.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def build_proxy_url() -> str:
    """Build a DataImpulse residential-proxy URL from env vars.

    Required env vars: ``DATAIMPULSE_USER``, ``DATAIMPULSE_PASS``. Optional:
    ``DATAIMPULSE_HOST`` (default ``gw.dataimpulse.com``), ``DATAIMPULSE_PORT``
    (default ``823``), ``DATAIMPULSE_COUNTRY`` (default ``us``),
    ``DATAIMPULSE_SESSION`` (auto-generated if unset).

    Raises ``RedditError`` if user or pass are missing — going direct from a
    datacenter IP gets 403'd, so failing fast with a clear message is friendlier
    than letting the caller chase a generic block.

    The username embeds DataImpulse's modifiers:
      - ``__cr.<country>`` — pin exit country (US has the cleanest reputation
        with Reddit; this is anti-bot defensive).
      - ``__sid.<id>``     — sticky-session ID. On the rotating gateway
        ``gw.dataimpulse.com:823`` this is a no-op (each TCP connection still
        gets a new IP), but the singleton ``curl_cffi.Session`` reuses one
        connection for the whole process and therefore one IP. We include
        ``__sid`` anyway in case the user upgrades to a sticky-tier endpoint
        that does honor it; the cost is just URL noise.
    """
    user = os.environ.get("DATAIMPULSE_USER", "").strip()
    pwd = os.environ.get("DATAIMPULSE_PASS", "").strip()
    if not user or not pwd:
        raise RedditError("DataImpulse proxy not configured: USER or PASS is missing")

    host = os.environ.get("DATAIMPULSE_HOST", "gw.dataimpulse.com").strip()
    port = os.environ.get("DATAIMPULSE_PORT", "823").strip()
    country = os.environ.get("DATAIMPULSE_COUNTRY", "us").strip()
    session_id = os.environ.get("DATAIMPULSE_SESSION", "").strip() or secrets.token_hex(4)

    parts = [user]
    if country:
        parts.append(f"cr.{country}")
    parts.append(f"sid.{session_id}")
    return f"http://{'__'.join(parts)}:{pwd}@{host}:{port}"


_session = None


def _get_session():
    """Return a process-wide ``curl_cffi`` Session, creating it on first use.

    The Session is a singleton on purpose: connection reuse means every
    ``fetch_json`` call goes through the same TCP/TLS connection — and
    therefore the same DataImpulse exit IP. That gives us de-facto sticky-IP
    behavior without depending on the proxy's `__sid` modifier.
    """
    global _session
    if _session is not None:
        return _session

    # Lazy import: unit tests patch _get_session directly without ever
    # needing curl_cffi installed in the test environment.
    from curl_cffi import requests as curl_requests

    proxy = build_proxy_url()
    proxies = {"http": proxy, "https": proxy}
    # Anti-bot defense: impersonate=chrome120 emits the exact TLS ClientHello
    # bytes (cipher suites, extensions, GREASE values) and HTTP/2 SETTINGS
    # frames Chrome 120 sends. Without this, Fastly's JA3/H2 fingerprint
    # check identifies the client as a Python HTTP library and 403s the
    # request before any application-layer header is even read.
    _session = curl_requests.Session(
        impersonate="chrome120",
        proxies=proxies,
        headers=HEADERS,
        timeout=DEFAULT_TIMEOUT,
    )
    return _session


def fetch_json(path: str, params: dict | None = None, max_attempts: int = 2):
    """GET a Reddit JSON endpoint and return the parsed body.

    ``path`` is the path portion only (e.g. ``/r/python/hot.json``). The
    client tries each domain in ``DOMAINS`` order, retrying ``max_attempts``
    times per domain on retryable statuses (see ``RETRYABLE_STATUSES``)
    before falling back to the next domain.

    Raises ``RedditError`` with ``status_code=403`` if every domain/attempt
    combination fails, including the case where Reddit returned status 200
    but with an HTML body (its bot-challenge page).
    """
    s = _get_session()
    started_at = time.monotonic()
    last_error = None
    last_status = None
    attempts = 0  # total attempts across all domains
    final_domain = None

    def _emit(outcome: str):
        # Structured single-line summary. JSON-encoded so it's grep+jq-friendly
        # in tail logs without needing the backend's structured-log helpers.
        log.info(json.dumps({
            "event": "reddit_fetch",
            "outcome": outcome,
            "path": path,
            "domain": final_domain,
            "status": last_status,
            "attempts": attempts,
            "duration_ms": int((time.monotonic() - started_at) * 1000),
            "error": last_error,
        }))

    for domain in DOMAINS:
        final_domain = domain
        url = f"{domain}{path}"
        for attempt in range(max_attempts):
            attempts += 1
            try:
                r = s.get(url, params=params, allow_redirects=True)
            except Exception as e:
                # Any low-level network/TLS failure — retry once with a small
                # back-off, then move on.
                last_error = f"{type(e).__name__}: {e}"
                last_status = None
                time.sleep(1)
                continue

            last_status = r.status_code

            if r.status_code == 200:
                ctype = r.headers.get("content-type", "")
                if "json" in ctype:
                    last_error = None
                    _emit("ok")
                    return r.json()
                # Anti-bot defense: Reddit's edge sometimes serves the
                # bot-challenge HTML at status 200 (not 403) when the JS
                # challenge is meant to redirect a real browser. We refuse
                # to treat that as success — a JSON endpoint must return
                # JSON. Retry, then fall back to the other domain.
                last_error = f"non-JSON 200 from {domain} (likely bot challenge)"
                time.sleep(1.0 + attempt)
                continue

            if r.status_code in RETRYABLE_STATUSES:
                last_error = f"{r.status_code} from {domain}"
                # Linear back-off: 1s, 2s, 3s. Reddit's per-IP rate limits
                # reset on the order of seconds, and a hot retry often slips
                # through the probabilistic bot challenge.
                time.sleep(1.0 + attempt)
                continue

            # 4xx/5xx that won't change on retry (404, 500, etc.) — abandon
            # this domain immediately and try the next one.
            last_error = f"HTTP {r.status_code} from {domain}"
            break

    _emit("failed")
    raise RedditError(f"Reddit request failed: {last_error}", status_code=403)
