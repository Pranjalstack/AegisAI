# backend/url_analyzer.py

import ipaddress
import re
from urllib.parse import urlparse


SUSPICIOUS_PATH_TERMS = {
    "verify",
    "verification",
    "login",
    "signin",
    "sign-in",
    "secure",
    "security",
    "account",
    "update",
    "confirm",
    "recover",
    "unlock",
    "payment",
    "billing",
    "refund",
    "wallet",
    "password",
    "otp",
}


SHORTENER_DOMAINS = {
    "bit.ly",
    "tinyurl.com",
    "t.co",
    "goo.gl",
    "ow.ly",
    "is.gd",
    "buff.ly",
}


def normalize_url(url):
    return str(url or "").strip()


def get_hostname(url):
    parsed = urlparse(
        normalize_url(url)
    )

    hostname = parsed.hostname

    if not hostname:
        return ""

    return hostname.lower().rstrip(".")


def is_ip_address(hostname):
    if not hostname:
        return False

    try:
        ipaddress.ip_address(
            hostname
        )
        return True

    except ValueError:
        return False


def has_punycode(hostname):
    return any(
        label.startswith("xn--")
        for label in hostname.split(".")
        if label
    )


def has_username_component(url):
    parsed = urlparse(
        normalize_url(url)
    )

    return bool(
        parsed.username
    )


def has_unusual_port(url):
    parsed = urlparse(
        normalize_url(url)
    )

    try:
        port = parsed.port
    except ValueError:
        return True

    if port is None:
        return False

    return port not in {
        80,
        443,
    }


def has_many_subdomains(hostname):
    if not hostname:
        return False

    labels = [
        label
        for label in hostname.split(".")
        if label
    ]

    return len(labels) >= 5


def has_suspicious_path(url):
    parsed = urlparse(
        normalize_url(url)
    )

    combined = (
        f"{parsed.path}?"
        f"{parsed.query}"
    ).lower()

    return any(
        re.search(
            rf"(?<![a-z]){re.escape(term)}(?![a-z])",
            combined,
            re.IGNORECASE,
        )
        for term in SUSPICIOUS_PATH_TERMS
    )


def is_shortener(hostname):
    if not hostname:
        return False

    if hostname in SHORTENER_DOMAINS:
        return True

    return any(
        hostname.endswith(
            "." + domain
        )
        for domain in SHORTENER_DOMAINS
    )


def analyze_url(url):
    """
    Analyze a URL using local structural heuristics.

    This function does NOT perform a network request and does NOT
    claim that a domain is malicious based only on its appearance.
    """

    url = normalize_url(
        url
    )

    if not url:
        return {
            "url": "",
            "risk": "LOW",
            "score": 0,
            "reasons": [],
            "hostname": "",
        }

    if not re.match(
        r"^https?://",
        url,
        re.IGNORECASE,
    ):
        return {
            "url": url,
            "risk": "LOW",
            "score": 0,
            "reasons": [
                "URL does not use HTTP or HTTPS"
            ],
            "hostname": "",
        }

    parsed = urlparse(
        url
    )

    hostname = get_hostname(
        url
    )

    reasons = []
    score = 0


    # ---------------------------------------------------------
    # IP address host
    # ---------------------------------------------------------

    if is_ip_address(
        hostname
    ):
        reasons.append(
            "Uses an IP address instead of a domain name"
        )
        score += 2


    # ---------------------------------------------------------
    # Punycode / internationalized hostname
    # ---------------------------------------------------------

    if has_punycode(
        hostname
    ):
        reasons.append(
            "Uses an internationalized/punycode hostname"
        )
        score += 2


    # ---------------------------------------------------------
    # Embedded username
    # Example:
    # https://example.com@evil.example/
    # ---------------------------------------------------------

    if has_username_component(
        url
    ):
        reasons.append(
            "Contains an embedded username component"
        )
        score += 2


    # ---------------------------------------------------------
    # Unusual port
    # ---------------------------------------------------------

    if has_unusual_port(
        url
    ):
        reasons.append(
            "Uses a non-standard HTTP/HTTPS port"
        )
        score += 1


    # ---------------------------------------------------------
    # Excessive subdomains
    # ---------------------------------------------------------

    if has_many_subdomains(
        hostname
    ):
        reasons.append(
            "Uses an unusually deep subdomain structure"
        )
        score += 1


    # ---------------------------------------------------------
    # Suspicious path terminology
    # ---------------------------------------------------------

    if has_suspicious_path(
        url
    ):
        reasons.append(
            "URL path contains an account or security-related term"
        )
        score += 1


    # ---------------------------------------------------------
    # URL shortener
    # ---------------------------------------------------------

    if is_shortener(
        hostname
    ):
        reasons.append(
            "Uses a URL-shortening service"
        )
        score += 1


    # ---------------------------------------------------------
    # Long URL
    # ---------------------------------------------------------

    if len(url) > 180:
        reasons.append(
            "URL is unusually long"
        )
        score += 1


    # ---------------------------------------------------------
    # No hostname
    # ---------------------------------------------------------

    if not hostname:
        reasons.append(
            "URL does not contain a valid hostname"
        )
        score += 2


    # ---------------------------------------------------------
    # Risk classification
    # ---------------------------------------------------------

    if score >= 5:
        risk = "HIGH"

    elif score >= 2:
        risk = "MEDIUM"

    else:
        risk = "LOW"


    # Remove duplicate reasons.
    unique_reasons = []

    for reason in reasons:

        if reason not in unique_reasons:
            unique_reasons.append(
                reason
            )


    return {
        "url": url,
        "risk": risk,
        "score": score,
        "reasons": unique_reasons,
        "hostname": hostname,
    }


if __name__ == "__main__":

    test_urls = [
        "https://www.google.com",
        "https://example.com",
        "https://192.168.1.20/login",
        "https://xn--pple-43d.example/login",
        "https://example.com@evil.example/login",
        "https://example.com:8080/login",
        "https://a.b.c.d.e.example.com/login",
        "https://bit.ly/abc123",
        "https://example.com/account/verify",
    ]


    print(
        "\n===== AEGIS URL INTELLIGENCE TEST ====="
    )


    for url in test_urls:

        result = analyze_url(
            url
        )

        print(
            "\n----------------------------------------"
        )

        print(
            f"URL: {result['url']}"
        )

        print(
            f"Hostname: {result['hostname']}"
        )

        print(
            f"Risk: {result['risk']}"
        )

        print(
            f"Score: {result['score']}"
        )

        print(
            "Reasons:"
        )

        if result["reasons"]:

            for reason in result["reasons"]:
                print(
                    f"• {reason}"
                )

        else:

            print(
                "• None"
            )
            