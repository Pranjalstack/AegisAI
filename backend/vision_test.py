from PIL import Image
from transformers import AutoProcessor, AutoModelForMultimodalLM


MODEL_NAME = "Qwen/Qwen3.5-2B"

IMAGE_PATH = r"screenshots\screen_20260919_145235.png"

print("Loading vision model...")

processor = AutoProcessor.from_pretrained(MODEL_NAME)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto"
)

print("Vision model loaded.")

image = Image.open(IMAGE_PATH).convert("RGB")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": (
                    "Look at this screenshot carefully. "
                    "Describe what is visibly shown on the screen. "
                    "Mention any security warning, suspicious message, "
                    "login request, payment request, or suspicious link "
                    "you can actually see. Do not guess."
                )
            }
        ]
    }
]

inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt"
).to(model.device)

outputs = model.generate(
    **inputs,
    max_new_tokens=150
)

generated_tokens = outputs[
    0
][inputs["input_ids"].shape[-1]:]

answer = processor.decode(
    generated_tokens,
    skip_special_tokens=True
)

print("\n===== AEGIS VISION TEST =====")
print(answer)