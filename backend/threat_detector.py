# backend/threat_detector.py

import re

from backend.url_analyzer import analyze_url


URL_PATTERN = re.compile(
    r"https?://[^\s<>\]\)\"']+",
    re.IGNORECASE,
)


def normalize_text(text):
    text = str(text or "")

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


def extract_urls(text):
    urls = URL_PATTERN.findall(
        str(text or "")
    )

    cleaned = []

    for url in urls:

        url = url.rstrip(
            ".,!?;:"
        )

        if url not in cleaned:
            cleaned.append(
                url
            )

    return cleaned


def contains_any(text, patterns):
    return any(
        re.search(
            pattern,
            text,
            re.IGNORECASE,
        )
        for pattern in patterns
    )


def analyze_text(text):
    """
    Context-aware Aegis threat detector.

    Text signals and URL structural signals are analyzed
    separately and then combined into one result.
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
    score = 0


    # ---------------------------------------------------------
    # 1. Strong urgency / threat
    # ---------------------------------------------------------

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
        r"\baccount\s+(?:will|is|has\s+been)\s+(?:suspended|blocked|locked|restricted)\b",
        r"\baccess\s+(?:will|is)\s+(?:be\s+)?(?:suspended|blocked|locked|restricted)\b",
    ]

    strong_urgency_detected = contains_any(
        lowered,
        strong_urgency_patterns,
    )

    if strong_urgency_detected:

        reasons.append(
            "Uses urgent or threatening language"
        )

        score += 2


    # ---------------------------------------------------------
    # 2. Mild urgency
    # ---------------------------------------------------------

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


    # ---------------------------------------------------------
    # 3. Credential harvesting
    # ---------------------------------------------------------

    credential_patterns = [

        r"\b(?:enter|provide|send|share|submit|give|tell|confirm)\b"
        r".{0,60}"
        r"\b(?:password|otp|one[-\s]?time\s+password|pin|cvv|security\s+code|verification\s+code)\b",

        r"\b(?:password|otp|one[-\s]?time\s+password|pin|cvv|security\s+code|verification\s+code)\b"
        r".{0,60}"
        r"\b(?:enter|provide|send|share|submit|give|tell|confirm)\b",

        r"\b(?:login|sign[-\s]?in)\b"
        r".{0,60}"
        r"\b(?:password|otp|pin|verification\s+code)\b",

        r"\b(?:send|share|give|provide)\b"
        r".{0,60}"
        r"\b(?:card\s+number|credit\s+card|debit\s+card|account\s+number|cvv|upi|ifsc)\b",
    ]

    credential_detected = contains_any(
        lowered,
        credential_patterns,
    )

    if credential_detected:

        reasons.append(
            "Requests sensitive credentials or financial information"
        )

        score += 3


    # ---------------------------------------------------------
    # 4. Payment / money request
    # ---------------------------------------------------------

    payment_patterns = [

        r"\bpayment\s+of\s+(?:rs\.?|inr|₹|\$|usd)?\s*\d+",

        r"\bpay\s+(?:rs\.?|inr|₹|\$|usd)?\s*\d+",

        r"\b(?:processing|verification|activation|service|support)\s+fee\b",

        r"\bfee\s+of\s+(?:rs\.?|inr|₹|\$|usd)?\s*\d+",

        r"\b(?:pay|send|transfer)\b"
        r".{0,60}"
        r"\b(?:money|fee|payment|amount|rs\.?|inr|₹|\$|usd)\b",

        r"\b(?:payment|fee|amount)\b"
        r".{0,40}"
        r"\b(?:required|due|needed|to\s+continue)\b",
    ]

    payment_detected = contains_any(
        lowered,
        payment_patterns,
    )

    if payment_detected:

        reasons.append(
            "Contains a payment or money-transfer request"
        )

        score += 2


    # ---------------------------------------------------------
    # 5. Account security context
    # ---------------------------------------------------------

    account_security_patterns = [

        r"\b(?:bank|banking)\b"
        r".{0,100}"
        r"\b(?:verify|verification|suspended|blocked|locked|restore|reactivate)\b",

        r"\baccount\b"
        r".{0,80}"
        r"\b(?:suspended|blocked|locked|restricted)\b",

        r"\baccount\b"
        r".{0,80}"
        r"\b(?:verify|verification)\b",
    ]

    account_security_detected = contains_any(
        lowered,
        account_security_patterns,
    )


    # ---------------------------------------------------------
    # 6. Technical support scam
    # ---------------------------------------------------------

    technical_support_patterns = [

        r"\btechnical\s+support\b"
        r".{0,100}"
        r"\b(?:remote|install|download|access|control|password|otp|pay|fee)\b",

        r"\b(?:technician|help\s*desk)\b"
        r".{0,100}"
        r"\b(?:remote\s+access|remote\s+control|install|download|anydesk|teamviewer|quick\s*assist)\b",

        r"\b(?:support|technician|help\s*desk)\b"
        r".{0,100}"
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

        score += 3


    # ---------------------------------------------------------
    # 7. Link context
    # ---------------------------------------------------------

    link_context_patterns = [
        r"\bclick\s+(?:here|this|the\s+link)\b",
        r"\bopen\s+(?:this|the)\s+link\b",
        r"\bvisit\s+(?:this|the)\s+link\b",
        r"\bverify\s+(?:your|the)\s+account\b",
        r"\benter\b.{0,60}\b(?:password|otp|pin)\b",
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

        score += 1


    # ---------------------------------------------------------
    # 8. Impersonation context
    # ---------------------------------------------------------

    impersonation_patterns = [

        r"\b(?:your|the)\s+bank\s+account\b"
        r".{0,100}"
        r"\b(?:verify|verification|suspended|blocked|locked|restore|reactivate)\b",

        r"\b(?:paypal|paytm|phonepe|google\s+pay|amazon|microsoft|apple|google|meta|instagram|facebook|whatsapp|netflix)\b"
        r".{0,100}"
        r"\b(?:verify|verification|suspended|blocked|locked|restore|reactivate|payment)\b",
    ]

    impersonation_detected = contains_any(
        lowered,
        impersonation_patterns,
    )

    if impersonation_detected:

        reasons.append(
            "References an organization commonly used in impersonation scams"
        )

        score += 2


    # ---------------------------------------------------------
    # 9. URL structural intelligence
    # ---------------------------------------------------------

    url_score = 0

    for url in urls:

        url_result = analyze_url(
            url
        )

        url_score += url_result["score"]

        for reason in url_result["reasons"]:

            if reason not in reasons:

                reasons.append(
                    reason
                )


    # Prevent URL structure alone from overwhelming the
    # message context. At most 3 points are contributed
    # to the overall text risk score.
    score += min(
        url_score,
        3
    )


    # ---------------------------------------------------------
    # 10. Contextual combinations
    # ---------------------------------------------------------

    if mild_urgency_detected and (
        credential_detected
        or payment_detected
        or urls
    ):
        score += 1


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


    # ---------------------------------------------------------
    # Final risk level
    # ---------------------------------------------------------

    if score >= 8:

        risk = "HIGH"

    elif score >= 4:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    unique_reasons = []

    for reason in reasons:

        if reason not in unique_reasons:

            unique_reasons.append(
                reason
            )


    return {
        "risk": risk,
        "score": score,
        "reasons": unique_reasons,
        "urls": urls,
    }


if __name__ == "__main__":

    examples = [

        (
            "SAFE",
            "Hey, are we still meeting at the gym at 6 PM?"
        ),

        (
            "SAFE_WEBSITE",
            """
            Your order has been shipped.
            You can check the delivery status from the
            shopping website you normally use.
            """
        ),

        (
            "SAFE_ACCOUNT",
            """
            You can review your account information from
            the official mobile app whenever convenient.
            """
        ),

        (
            "LOW_SECURITY_NOTICE",
            """
            Your password expires next month.
            Please update it when you have time.
            """
        ),

        (
            "MEDIUM_URGENCY",
            """
            Action required: please verify your account
            information before the end of the week.
            """
        ),

        (
            "MEDIUM_LINK",
            """
            Your account requires verification.
            Please click here to review your account:
            https://example.com/account
            """
        ),

        (
            "MEDIUM_CREDENTIAL",
            """
            Security verification is required.
            Please enter your OTP to continue.
            """
        ),

        (
            "MEDIUM_PAYMENT",
            """
            Your subscription requires a payment of Rs. 299
            to continue service.
            """
        ),

        (
            "HIGH_BANK_SCAM",
            """
            URGENT: Your bank account will be suspended within
            24 hours.

            Click the link below and enter your password and OTP:
            https://example.com/verify

            A processing fee of Rs. 499 is required to restore access.
            """
        ),

        (
            "HIGH_TECH_SUPPORT",
            """
            URGENT: Your computer has a serious security problem.

            Call technical support immediately.
            Install remote access software and provide your password
            so the technician can fix the issue.

            A support fee of Rs. 999 is required today.
            """
        ),

        (
            "URL_IP",
            "Please verify your account here: https://192.168.1.20/login"
        ),

        (
            "URL_SHORTENER",
            "Here is the shortened link: https://bit.ly/abc123"
        ),
    ]


    print(
        "\n===== AEGIS THREAT DETECTOR ====="
    )


    for name, example in examples:

        result = analyze_text(
            example
        )

        print(
            "\n----------------------------------------"
        )

        print(
            f"Test: {name}"
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


        print(
            "URLs:"
        )

        if result["urls"]:

            for url in result["urls"]:

                print(
                    f"• {url}"
                )

        else:

            print(
                "• None"
            )