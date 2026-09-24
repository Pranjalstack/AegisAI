# backend/test_detector.py

from backend.threat_detector import analyze_text


TEST_CASES = [
    (
        "LEGIT_GOOGLE",
        """
        Visit the Google website:
        https://www.google.com
        """
    ),

    (
        "LEGIT_LOGIN_PATH",
        """
        You can sign in to your account here:
        https://example.com/login
        """
    ),

    (
        "LEGIT_ACCOUNT_PATH",
        """
        Open your account dashboard:
        https://example.com/account
        """
    ),

    (
        "LEGIT_HTTPS",
        """
        Our website is available at:
        https://example.com
        """
    ),

    (
        "SHORTENER_ONLY",
        """
        Here is the link:
        https://bit.ly/abc123
        """
    ),

    (
        "IP_URL_ONLY",
        """
        Internal service:
        https://192.168.1.20
        """
    ),

    (
        "PUNYCODE_ONLY",
        """
        Visit:
        https://xn--pple-43d.example
        """
    ),

    (
        "URL_PLUS_CREDENTIAL_REQUEST",
        """
        Please enter your password here:
        https://example.com/login
        """
    ),

    (
        "URL_PLUS_URGENT_BANK_REQUEST",
        """
        URGENT: Your bank account will be suspended today.

        Verify your account immediately:
        https://example.com/account/verify
        """
    ),

    (
        "URL_PLUS_PAYMENT",
        """
        Your subscription requires payment of Rs. 299.

        Complete payment here:
        https://example.com/payment
        """
    ),
]


def run_tests():

    print(
        "\n===== AEGIS LEGITIMATE URL BENCHMARK ====="
    )

    for name, text in TEST_CASES:

        result = analyze_text(text)

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


if __name__ == "__main__":
    run_tests()