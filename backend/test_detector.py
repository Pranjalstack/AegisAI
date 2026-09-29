# backend/threat_detector.py

import re
from urllib.parse import urlparse


# ============================================================
# URL detection
# ============================================================

URL_PATTERN = re.compile(
    r"https?://[^\s<>\"]+",
    re.IGNORECASE,
)


# ============================================================
# Text normalization
# ============================================================

def normalize_text(text):
    value = str(
        text or ""
    )

    value = value.replace(
        "\u2018",
        "'",
    )

    value = value.replace(
        "\u2019",
        "'",
    )

    value = value.replace(
        "\u201c",
        '"',
    )

    value = value.replace(
        "\u201d",
        '"',
    )

    value = value.replace(
        "\u2013",
        "-",
    )

    value = value.replace(
        "\u2014",
        "-",
    )

    value = value.replace(
        "\u00a0",
        " ",
    )

    value = re.sub(
        r"\s+",
        " ",
        value,
    )

    return value.strip()


# ============================================================
# Generic regex helper
# ============================================================

def contains_any(text, patterns):
    value = str(
        text or ""
    )

    for pattern in patterns:

        try:

            if re.search(
                pattern,
                value,
                re.IGNORECASE,
            ):
                return True

        except re.error:
            continue

    return False


# ============================================================
# URL extraction
# ============================================================

def extract_urls(text):
    original_text = str(
        text or ""
    )

    urls = []

    matches = URL_PATTERN.findall(
        original_text
    )

    for match in matches:

        cleaned = match.rstrip(
            ".,;:!?)]}>\"'"
        )

        if cleaned and cleaned not in urls:

            urls.append(
                cleaned
            )

    return urls


# ============================================================
# URL analysis
# ============================================================

def analyze_url(url):
    original_url = str(
        url or ""
    ).strip()

    reasons = []
    score = 0

    if not original_url:

        return {
            "risk": "LOW",
            "score": 0,
            "reasons": [],
            "url": "",
        }

    parsed = urlparse(
        original_url
    )

    hostname = (
        parsed.hostname
        or ""
    ).lower()

    path = (
        parsed.path
        or ""
    ).lower()

    query = (
        parsed.query
        or ""
    ).lower()

    # ---------------------------------------------------------
    # Missing hostname
    # ---------------------------------------------------------

    if not hostname:

        reasons.append(
            "URL does not contain a valid hostname"
        )

        score += 3

    # ---------------------------------------------------------
    # Raw IPv4 address
    # ---------------------------------------------------------

    ipv4_pattern = (
        r"^(?:\d{1,3}\.){3}\d{1,3}$"
    )

    if re.fullmatch(
        ipv4_pattern,
        hostname,
    ):

        reasons.append(
            "Uses a raw IP address instead of a normal domain"
        )

        score += 3

    # ---------------------------------------------------------
    # Punycode
    # ---------------------------------------------------------

    if "xn--" in hostname:

        reasons.append(
            "Uses punycode that may indicate a lookalike domain"
        )

        score += 3

    # ---------------------------------------------------------
    # Username in URL
    # ---------------------------------------------------------

    if parsed.username:

        reasons.append(
            "Contains a username before the hostname"
        )

        score += 2

    # ---------------------------------------------------------
    # Unusual port
    # ---------------------------------------------------------

    try:

        port = parsed.port

    except ValueError:

        port = None

        reasons.append(
            "Contains an invalid URL port"
        )

        score += 2

    if port is not None and port not in {
        80,
        443,
    }:

        reasons.append(
            "Uses an unusual explicit network port"
        )

        score += 2

    # ---------------------------------------------------------
    # Excessive subdomains
    # ---------------------------------------------------------

    hostname_parts = [
        part
        for part in hostname.split(".")
        if part
    ]

    if len(hostname_parts) >= 5:

        reasons.append(
            "Uses an unusually deep subdomain structure"
        )

        score += 2

    # ---------------------------------------------------------
    # Suspicious path
    # ---------------------------------------------------------

    suspicious_path_patterns = [
        r"\blogin\b",
        r"\blog[\s_-]*in\b",
        r"\bsign[\s_-]*in\b",
        r"\bverify\b",
        r"\bverification\b",
        r"\baccount\b",
        r"\bsecure\b",
        r"\bsecurity\b",
        r"\bpassword\b",
        r"\botp\b",
        r"\bupdate\b",
        r"\bunlock\b",
        r"\bconfirm\b",
        r"\breactivate\b",
    ]

    if contains_any(
        path,
        suspicious_path_patterns,
    ):

        reasons.append(
            "URL path contains account or security-related keywords"
        )

        score += 1

    # ---------------------------------------------------------
    # URL shorteners
    # ---------------------------------------------------------

    shortener_domains = {
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "is.gd",
        "ow.ly",
        "buff.ly",
        "cutt.ly",
        "rb.gy",
        "shorturl.at",
        "tiny.cc",
        "rebrand.ly",
    }

    if hostname in shortener_domains:

        reasons.append(
            "Uses a URL-shortening service that hides the final destination"
        )

        score += 3

    # ---------------------------------------------------------
    # Long URL
    # ---------------------------------------------------------

    if len(
        original_url
    ) >= 180:

        reasons.append(
            "URL is unusually long and difficult to inspect"
        )

        score += 1

    # ---------------------------------------------------------
    # Suspicious query parameters
    # ---------------------------------------------------------

    suspicious_query_patterns = [
        r"\btoken=",
        r"\bredirect=",
        r"\breturn=",
        r"\bnext=",
        r"\bcontinue=",
        r"\bpassword=",
        r"\botp=",
        r"\bverify=",
        r"\blogin=",
    ]

    if contains_any(
        query,
        suspicious_query_patterns,
    ):

        reasons.append(
            "URL contains security-sensitive or redirect parameters"
        )

        score += 1

    # ---------------------------------------------------------
    # URL risk level
    # ---------------------------------------------------------

    if score >= 5:

        risk = "HIGH"

    elif score >= 2:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    return {
        "risk": risk,
        "score": score,
        "reasons": reasons,
        "url": original_url,
    }


# ============================================================
# Main threat detector
# ============================================================

def analyze_text(text):
    """
    Context-aware Aegis threat detector.

    Detects:
    - urgency
    - credential harvesting
    - payment requests
    - account security threats
    - technical-support scams
    - suspicious links
    - impersonation context
    - URL structural anomalies
    """

    original_text = str(
        text or ""
    )

    lowered = normalize_text(
        original_text
    ).lower()

    urls = extract_urls(
        original_text
    )

    reasons = []
    categories = []
    score = 0

    # ========================================================
    # 1. STRONG URGENCY
    # ========================================================

    strong_urgency_patterns = [

        r"\burgent\b",

        r"\bimmediately\b",

        r"\bact\s+now\b",

        r"\bwithin\s+\d+\s*(?:hours?|minutes?)\b",

        r"\blast\s+warning\b",

        r"\bfinal\s+notice\b",

        r"\bwill\s+be\s+suspended\b",

        r"\bwill\s+be\s+blocked\b",

        r"\bwill\s+be\s+locked\b",

        r"\bwill\s+be\s+closed\b",

        r"\baccount\s+(?:will|is|has\s+been)\s+"
        r"(?:suspended|blocked|locked|restricted)\b",

        r"\baccess\s+(?:will|is)\s+(?:be\s+)?"
        r"(?:suspended|blocked|locked|restricted)\b",
    ]

    strong_urgency_detected = contains_any(
        lowered,
        strong_urgency_patterns,
    )

    if strong_urgency_detected:

        reasons.append(
            "Uses urgent or threatening language"
        )

        categories.append(
            "urgency"
        )

        score += 2

    # ========================================================
    # 2. MILD URGENCY
    # ========================================================

    mild_urgency_patterns = [

        r"\baction\s+required\b",

        r"\bplease\s+verify\b",

        r"\breview\s+your\s+account\b",

        r"\bverification\s+is\s+required\b",

    ]

    mild_urgency_detected = contains_any(
        lowered,
        mild_urgency_patterns,
    )

    # ========================================================
    # 3. CREDENTIAL HARVESTING
    # ========================================================

    credential_action_words = (
        r"(?:"
        r"enter"
        r"|entering"
        r"|provide"
        r"|providing"
        r"|send"
        r"|sending"
        r"|share"
        r"|sharing"
        r"|submit"
        r"|submitting"
        r"|give"
        r"|giving"
        r"|tell"
        r"|telling"
        r"|confirm"
        r"|confirming"
        r")"
    )

    credential_words = (
        r"(?:"
        r"password"
        r"|otp"
        r"|one[-\s]?time\s+password"
        r"|pin"
        r"|cvv"
        r"|security\s+code"
        r"|verification\s+code"
        r")"
    )

    financial_information_words = (
        r"(?:"
        r"card\s+number"
        r"|credit\s+card"
        r"|debit\s+card"
        r"|account\s+number"
        r"|cvv"
        r"|upi"
        r"|ifsc"
        r")"
    )

    credential_patterns = [

        # Action followed by credential.
        rf"\b{credential_action_words}\b"
        rf".{{0,80}}"
        rf"\b{credential_words}\b",

        # Credential followed by action.
        rf"\b{credential_words}\b"
        rf".{{0,80}}"
        rf"\b{credential_action_words}\b",

        # Login/sign-in followed by credential.
        r"\b(?:login|log\s+in|sign[-\s]?in)\b"
        r".{0,80}"
        rf"\b{credential_words}\b",

        # Action followed by financial information.
        rf"\b{credential_action_words}\b"
        rf".{{0,80}}"
        rf"\b{financial_information_words}\b",

        # Very common direct requests.
        r"\b(?:send|sending|share|sharing|give|giving|provide|providing)\b"
        r".{0,80}"
        r"\b(?:your\s+)?(?:password|otp|pin|cvv)\b",

        # "your OTP and password" style phrases.
        r"\b(?:your|the)\s+(?:otp|password|pin|cvv)\b"
        r".{0,30}"
        r"\b(?:and|or)\b"
        r".{0,30}"
        r"\b(?:otp|password|pin|cvv)\b",
    ]

    credential_detected = contains_any(
        lowered,
        credential_patterns,
    )

    if credential_detected:

        reasons.append(
            "Requests sensitive credentials or financial information"
        )

        categories.append(
            "credentials"
        )

        score += 3

    # ========================================================
    # 4. PAYMENT / MONEY REQUEST
    # ========================================================

    payment_patterns = [

        r"\bpayment\s+of\s+"
        r"(?:rs\.?|inr|₹|\$|usd)?\s*\d+",

        r"\bpay\s+"
        r"(?:rs\.?|inr|₹|\$|usd)?\s*\d+",

        r"\b(?:processing|verification|activation|"
        r"service|support)\s+fee\b",

        r"\bfee\s+of\s+"
        r"(?:rs\.?|inr|₹|\$|usd)?\s*\d+",

        r"\b(?:pay|send|sending|transfer|transferring)\b"
        r".{0,80}"
        r"\b(?:money|fee|payment|amount|"
        r"rs\.?|inr|₹|\$|usd)\b",

        r"\b(?:payment|fee|amount)\b"
        r".{0,50}"
        r"\b(?:required|due|needed|to\s+continue)\b",

        r"\b(?:make|complete)\s+(?:a\s+)?payment\b",

    ]

    payment_detected = contains_any(
        lowered,
        payment_patterns,
    )

    if payment_detected:

        reasons.append(
            "Contains a payment or money-transfer request"
        )

        categories.append(
            "payment"
        )

        score += 2

    # ========================================================
    # 5. ACCOUNT SECURITY CONTEXT
    # ========================================================

    account_security_patterns = [

        r"\b(?:bank|banking)\b"
        r".{0,120}"
        r"\b(?:verify|verification|suspended|blocked|"
        r"locked|restricted|restore|reactivate)\b",

        r"\baccount\b"
        r".{0,100}"
        r"\b(?:suspended|blocked|locked|restricted)\b",

        r"\baccount\b"
        r".{0,100}"
        r"\b(?:verify|verification)\b",

    ]

    account_security_detected = contains_any(
        lowered,
        account_security_patterns,
    )

    # ========================================================
    # 6. TECHNICAL SUPPORT SCAM
    # ========================================================

    technical_support_patterns = [

        r"\btechnical\s+support\b"
        r".{0,120}"
        r"\b(?:remote|install|download|access|control|"
        r"password|otp|pay|fee)\b",

        r"\b(?:technician|help\s*desk)\b"
        r".{0,120}"
        r"\b(?:remote\s+access|remote\s+control|"
        r"install|download|anydesk|teamviewer|quick\s*assist)\b",

        r"\b(?:support|technician|help\s*desk)\b"
        r".{0,120}"
        r"\b(?:password|otp|pay|fee|remote\s+access)\b",

    ]

    technical_support_detected = contains_any(
        lowered,
        technical_support_patterns,
    )

    if technical_support_detected:

        reasons.append(
            "Contains potential technical-support scam indicators"
        )

        categories.append(
            "technical_support"
        )

        score += 3

    # ========================================================
    # 7. LINK CONTEXT
    # ========================================================

    link_context_patterns = [

        r"\bclick\s+(?:here|this|the\s+link)\b",

        r"\bopen\s+(?:this|the)\s+link\b",

        r"\bvisit\s+(?:this|the)\s+link\b",

        r"\bverify\s+(?:your|the)\s+account\b",

        r"\benter\b.{0,80}"
        r"\b(?:password|otp|pin|cvv)\b",

    ]

    suspicious_link_context = (
        bool(urls)
        and (
            strong_urgency_detected
            or credential_detected
            or payment_detected
            or account_security_detected
            or technical_support_detected
            or mild_urgency_detected
            or contains_any(
                lowered,
                link_context_patterns,
            )
        )
    )

    if suspicious_link_context:

        reasons.append(
            "Contains a link that should be verified before opening"
        )

        if "link" not in categories:

            categories.append(
                "link"
            )

        score += 1

    # ========================================================
    # 8. IMPERSONATION
    # ========================================================

    impersonation_patterns = [

        r"\b(?:your|the)\s+bank\s+account\b"
        r".{0,120}"
        r"\b(?:verify|verification|suspended|blocked|"
        r"locked|restricted|restore|reactivate)\b",

        r"\b(?:paypal|paytm|phonepe|google\s+pay|"
        r"amazon|microsoft|apple|google|meta|instagram|"
        r"facebook|whatsapp|netflix)\b"
        r".{0,120}"
        r"\b(?:verify|verification|suspended|blocked|"
        r"locked|restricted|restore|reactivate|payment)\b",

    ]

    impersonation_detected = contains_any(
        lowered,
        impersonation_patterns,
    )

    if impersonation_detected:

        reasons.append(
            "References an organization commonly used in impersonation scams"
        )

        categories.append(
            "impersonation"
        )

        score += 2

    # ========================================================
    # 9. URL STRUCTURAL INTELLIGENCE
    # ========================================================

    url_score = 0

    for url in urls:

        url_result = analyze_url(
            url
        )

        url_score += url_result[
            "score"
        ]

        for reason in url_result[
            "reasons"
        ]:

            if reason not in reasons:

                reasons.append(
                    reason
                )

        if url_result[
            "score"
        ] >= 2:

            if "link" not in categories:

                categories.append(
                    "link"
                )

    # Limit URL structural contribution.
    score += min(
        url_score,
        3,
    )

    # ========================================================
    # 10. CONTEXTUAL COMBINATIONS
    # ========================================================

    if mild_urgency_detected and (
        credential_detected
        or payment_detected
        or bool(urls)
    ):

        score += 1

        if "urgency" not in categories:

            categories.append(
                "urgency"
            )

    if payment_detected and (
        strong_urgency_detected
        or credential_detected
        or account_security_detected
        or impersonation_detected
    ):

        score += 1

    if credential_detected and (
        strong_urgency_detected
        or account_security_detected
        or impersonation_detected
    ):

        score += 1

    # Strong bank/account security + credential request
    # receives additional contextual weighting.
    if (
        credential_detected
        and account_security_detected
        and (
            strong_urgency_detected
            or impersonation_detected
        )
    ):

        score += 0

    # ========================================================
    # 11. FINAL RISK
    # ========================================================

    if score >= 8:

        risk = "HIGH"

    elif score >= 4:

        risk = "MEDIUM"

    else:

        risk = "LOW"

    # ========================================================
    # 12. UNIQUE REASONS
    # ========================================================

    unique_reasons = []

    for reason in reasons:

        if reason not in unique_reasons:

            unique_reasons.append(
                reason
            )

    # ========================================================
    # 13. UNIQUE CATEGORIES
    # ========================================================

    unique_categories = []

    for category in categories:

        if category not in unique_categories:

            unique_categories.append(
                category
            )

    # ========================================================
    # RESULT
    # ========================================================

    return {
        "risk": risk,
        "score": score,
        "reasons": unique_reasons,
        "categories": unique_categories,
        "urls": urls,
    }


# ============================================================
# Backward-compatible wrapper
# ============================================================

def detect_threat(text):
    return analyze_text(
        text
    )