import os
from openai import OpenAI
from pathlib import Path

_client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY", "placeholder"),
    base_url="https://api.deepseek.com"
)
PROFILE_PATH = Path("./knowledge/self_profile.md")


def run(jd_info: dict, company_research: str) -> str:
    """Generate interview question bank with answer framework hints."""
    profile = PROFILE_PATH.read_text(encoding="utf-8") if PROFILE_PATH.exists() else "（无个人简介）"

    response = _client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=2048,
        messages=[{
            "role": "user",
            "content": f"""为以下候选人生成面试准备题库。

目标公司：{jd_info['company']}
目标岗位：{jd_info['role']}
核心技能要求：{', '.join(jd_info['key_skills'])}

公司研究资料：
{company_research}

候选人背景：
{profile}

请生成以下四类题目（用中文 Markdown 格式）：

## 行为题（3条）
结合候选人背景，STAR 格式。每题附一行**回答框架**提示。

## 产品题（3条）
结合该公司具体产品，例如"如何提升XX的次日留存"。每题附一行**回答框架**提示。

## 技术认知题（2条）
AI PM 专项：模型评估、AI功能成功指标定义等。每题附一行**回答框架**提示。

## 反问题（2条）
建议候选人向面试官提问的高质量问题（体现产品思维和对公司的了解）。"""
        }]
    )
    return response.choices[0].message.content
