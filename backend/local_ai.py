# backend/local_ai.py

from PIL import Image

from transformers import (
    AutoProcessor,
    AutoModelForMultimodalLM,
)


MODEL_NAME = "Qwen/Qwen3.5-2B"


print(
    "Loading Aegis local AI model...",
    flush=True,
)


processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
)


print(
    "Aegis local AI model loaded.",
    flush=True,
)


def generate_text(
    messages,
    max_new_tokens=180,
):
    """
    Generate a response from the local Qwen model.
    """

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
    )

    generated_tokens = outputs[
        0
    ][
        inputs["input_ids"].shape[-1]:
    ]

    answer = processor.decode(
        generated_tokens,
        skip_special_tokens=True,
    )

    return answer.strip()


def explain_threat(
    text,
    detection_result,
):
    """
    Generate an evidence-constrained explanation.

    The deterministic detector controls the risk level.
    Qwen explains the supplied evidence and does not
    independently classify or invent facts.
    """

    risk = str(
        detection_result.get(
            "risk",
            "LOW",
        )
    ).upper()

    score = int(
        detection_result.get(
            "score",
            0,
        )
    )

    reasons = detection_result.get(
        "reasons",
        [],
    )

    reasons_text = "\n".join(
        f"- {reason}"
        for reason in reasons
    )

    # ---------------------------------------------------------
    # Hard safety gate for a clean result.
    # ---------------------------------------------------------

    if (
        risk == "LOW"
        and score == 0
        and not reasons
    ):
        return (
            "No concrete security indicators were detected "
            "in the supplied content. The available evidence "
            "does not indicate a known phishing, credential, "
            "payment, or scam pattern."
        )


    # ---------------------------------------------------------
    # Evidence-constrained prompt.
    # ---------------------------------------------------------

    prompt = f"""
You are Aegis AI, a digital safety assistant.

Your task is to explain the security evidence already
detected by Aegis.

The risk level and score are produced by a deterministic
security detector. You must NOT change them.

Aegis risk level:
{risk}

Aegis risk score:
{score}

Aegis detected indicators:
{reasons_text}

Observed text:
{text[:5000]}

STRICT EVIDENCE RULES:

1. Use only information explicitly present in the
   observed text or Aegis detected indicators.

2. Do not invent facts about:
   - websites
   - domains
   - companies
   - banks
   - organizations
   - people
   - devices
   - files
   - applications
   - locations
   - network activity
   - malware
   - cameras
   - microphones
   - background processes

3. Do not claim that a URL is malicious, fraudulent,
   a phishing site, unsafe, compromised, or owned by a
   particular organization unless that exact fact is
   explicitly present in the supplied evidence.

4. Do not claim that a person or organization is trying
   to steal information, money, or credentials unless
   the supplied evidence explicitly establishes that.

5. Do not infer motives, intentions, ownership, identity,
   or hidden actions.

6. Do not treat suspicious-looking words, OCR errors,
   random characters, filenames, timestamps, or formatting
   as security evidence unless the detector explicitly
   identified them as an indicator.

7. Do not compare dates or times from the content with
   the current date or time.

8. Do not use stronger language than the evidence supports.

9. When discussing a URL, describe it as a visible or
   detected link. Structural URL warnings may be described
   only using the exact detector indicator supplied above.

10. If the evidence is insufficient to establish a claim,
    explicitly say that the evidence is insufficient.

Return exactly these three sections:

WHY THIS MAY BE RISKY
Explain the concrete evidence that caused Aegis to flag
the content. Do not add unsupported interpretation.

WARNING SIGNS
List the supplied Aegis indicators in concise language.
Do not create new indicators.

SAFE ACTION
Give practical defensive advice based only on the evidence.
Do not claim that an external organization has confirmed
the threat.

Keep the response concise and factual.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                }
            ],
        }
    ]

    return generate_text(
        messages,
        max_new_tokens=220,
    )


def analyze_image(
    image_path,
):
    """
    Analyze a screenshot visually.

    The model reports only directly visible information
    and does not infer hidden system activity.
    """

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )

    prompt = """
You are the visual analysis component of Aegis AI.

Inspect the screenshot carefully.

Report ONLY information that is directly visible.

You may identify:
- visible warning messages
- visible urgent language
- visible credential requests
- visible payment requests
- visible URLs
- visible suspicious popups
- clearly visible security-related controls
- visible account or verification requests

STRICT RULES:

1. Do not guess what software, file, process, device,
   website owner, person, company, browser, network
   connection, webcam, microphone, or background service
   is involved.

2. Do not infer hidden actions.

3. Do not invent:
   - filenames
   - URLs
   - organizations
   - identities
   - malware
   - attacks
   - ownership
   - motives

4. Do not compare screenshot dates or times with the
   current date or time.

5. Do not call a date or timestamp suspicious merely
   because of its value.

6. Do not treat OCR-like garbled text as proof of a
   security threat.

7. Do not treat generic words such as:
   "Security Result"
   "Scan Screen"
   "Scan File"
   "Scan Message"
   "Scan Audio"
   "suspicious"
   as evidence of an actual threat when they are part
   of the Aegis interface.

8. Do not claim a visible URL is malicious merely because
   it looks unfamiliar.

9. Do not claim an interface belongs to a particular
   organization unless that identity is clearly visible
   and directly established.

10. If no concrete security evidence is visible, write:
    "None directly visible."

Return exactly:

VISIBLE CONTENT
Briefly describe only the important visible elements.

SECURITY EVIDENCE
List only directly visible security-relevant evidence.
If none is visible, write:
"None directly visible."

CONFIDENCE LIMITATION
State what cannot be determined from the screenshot alone.

Keep the response concise and evidence-based.
"""

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        }
    ]

    return generate_text(
        messages,
        max_new_tokens=220,
    )