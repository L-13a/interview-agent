import json
import anthropic

_client = anthropic.Anthropic()


def parse_jd(jd_text: str) -> dict:
    """Parse JD text into structured dict.

    Returns:
        {"company": str, "role": str, "role_type": str, "key_skills": list[str]}
        role_type is one of: commercial, ai_product, growth, platform, other
    """
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        system="You extract structured information from job descriptions. Return only valid JSON, no explanation.",
        messages=[{
            "role": "user",
            "content": f"""Extract from this job description:
- company: company name (string)
- role: exact job title (string)
- role_type: one of "commercial", "ai_product", "growth", "platform", "other"
- key_skills: list of 3-5 core required skills (strings, in Chinese if the JD is Chinese)

JD:
{jd_text}"""
        }]
    )
    return json.loads(response.content[0].text)
