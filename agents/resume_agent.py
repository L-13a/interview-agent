import anthropic

_client = anthropic.Anthropic()


def run(jd_info: dict, kb) -> str:
    """Generate tailored resume bullets with a reflection loop."""
    chunks = kb.query(" ".join(jd_info["key_skills"]) + " " + jd_info["role_type"], n_results=3)

    if not chunks:
        return "⚠️ 未找到相关经历，请在 knowledge/ 目录下补充你的实习和项目经历文档。"

    context = "\n\n---\n\n".join(chunks)
    draft = _generate_bullets(jd_info, context)
    critique = _critique_bullets(jd_info, draft)
    return _revise_bullets(jd_info, draft, critique)


def _generate_bullets(jd_info: dict, context: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""你是专业简历顾问。根据候选人经历，为以下岗位撰写定制简历 bullets。

目标岗位：{jd_info['role']}（{jd_info['company']}）
岗位类型：{jd_info['role_type']}
核心要求：{', '.join(jd_info['key_skills'])}

候选人相关经历：
{context}

每段经历写 2-3 条 bullet，要求：
- 动词开头
- 优先包含量化数据
- 突出与目标岗位相关的能力

用中文输出。"""
        }]
    )
    return response.content[0].text


def _critique_bullets(jd_info: dict, draft: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": f"""审查以下简历 bullets，目标岗位要求：{', '.join(jd_info['key_skills'])}。

{draft}

对每条 bullet 检查：
1. 是否体现了岗位核心要求中的某项能力？
2. 是否有量化数据？
3. 是否动词开头？

列出需要改进的具体建议，简短。"""
        }]
    )
    return response.content[0].text


def _revise_bullets(jd_info: dict, draft: str, critique: str) -> str:
    response = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""根据以下审查意见，修改简历 bullets。只输出修改后的 bullets，不需要解释。

原始 bullets：
{draft}

审查意见：
{critique}

目标岗位：{jd_info['role']}（{jd_info['company']}）
用中文输出。"""
        }]
    )
    return response.content[0].text
