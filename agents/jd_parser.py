import json
import os
from openai import OpenAI

_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "placeholder"),
    base_url="https://api.deepseek.com"
)


def parse_jd(jd_text: str) -> dict:
    """Parse JD text into structured dict.

    Returns:
        {"company": str, "role": str, "role_type": str, "key_skills": list[str]}
        role_type is one of: commercial, ai_product, growth, platform, other
    """
    response = _client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=512,
        messages=[
            {
                "role": "system",
                "content": "You extract structured information from job descriptions. Return only valid JSON, no explanation."
            },
            {
                "role": "user",
                "content": f"""Extract from this job description:
- company: company name (string)
- role: exact job title (string)
- role_type: one of "commercial", "ai_product", "growth", "platform", "other"
- key_skills: list of 3-5 core required skills (strings, in Chinese if the JD is Chinese)

JD:
{jd_text}"""
            }
        ]
    )
    return json.loads(response.choices[0].message.content)
