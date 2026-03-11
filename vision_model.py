import base64
import json
import re
import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_image_with_vision(image_path):

    base64_image = encode_image(image_path)

    prompt = """
You are a forensic document analysis system.

Analyze the ID document image for possible tampering.

Focus ONLY on visual inconsistencies such as:
- text alignment issues
- font mismatch
- photo replacement
- layout inconsistencies
- editing artifacts

Return STRICT JSON ONLY:

{
 "tampering_signs": [],
 "layout_consistency": "normal | suspicious",
 "photo_region": "normal | suspicious",
 "risk_level": "low | medium | high",
 "explanation": ""
}

Do not add any text outside the JSON.
"""

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                ],
            }
        ],
    )

    output = response.choices[0].message.content

    try:
        json_match = re.search(r"\{.*\}", output, re.S)
        parsed = json.loads(json_match.group())
    except:
        parsed = {"raw_output": output}

    return parsed