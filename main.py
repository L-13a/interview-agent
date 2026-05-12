import asyncio
import os
import sys
from datetime import date
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from rag.knowledge_base import KnowledgeBase
from memory.manager import load_history, save_application, find_similar
from agents.jd_parser import parse_jd
from agents.resume_agent import run as resume_run
from agents.research_agent import run as research_run
from agents.interview_agent import run as interview_run

load_dotenv()
console = Console()


def _read_jd() -> str:
    console.print("\n[bold cyan]请粘贴 JD 内容，粘贴完成后按两次 Enter：[/bold cyan]")
    lines = []
    while True:
        line = input()
        if line == "" and lines and lines[-1] == "":
            break
        lines.append(line)
    return "\n".join(lines[:-1]).strip()


async def _run_phase1(jd_info: dict, kb: KnowledgeBase) -> tuple[str, str]:
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=2) as pool:
        resume_future = loop.run_in_executor(pool, resume_run, jd_info, kb)
        research_future = loop.run_in_executor(pool, research_run, jd_info)
        resume_result, research_result = await asyncio.gather(resume_future, research_future)
    return resume_result, research_result


def main():
    # ── Setup ──────────────────────────────────────────────
    console.print(Panel("[bold]Job Interview Prep Agent[/bold]", border_style="cyan"))

    console.print("\n[dim]正在初始化知识库...[/dim]")
    kb = KnowledgeBase()
    count = kb.index_documents()
    console.print(f"[green]✓ 知识库就绪，共 {count} 个文本块[/green]")

    # ── JD Input & Parse ───────────────────────────────────
    jd_text = _read_jd()
    if not jd_text:
        console.print("[red]JD 内容为空，退出。[/red]")
        sys.exit(1)

    console.print("\n[dim]解析 JD 中...[/dim]")
    jd_info = parse_jd(jd_text)
    console.print(
        f"[green]✓ 公司：{jd_info['company']} | 岗位：{jd_info['role']} | 类型：{jd_info['role_type']}[/green]"
    )

    # ── Memory Check ───────────────────────────────────────
    similar = find_similar(jd_info["company"], jd_info["role_type"])
    if similar:
        names = "、".join(f"{a['company']} {a['role']}" for a in similar[:3])
        console.print(f"[yellow]💡 发现历史投递记录：{names}[/yellow]")

    # ── Phase 1: Parallel ──────────────────────────────────
    console.print("\n[bold]Phase 1：简历改写 & 公司研究（并行执行中）...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
        transient=True,
    ) as progress:
        t1 = progress.add_task("Resume Agent  — RAG 召回 + 生成 + Reflection", total=None)
        t2 = progress.add_task("Research Agent — Tavily 搜索 + 汇总", total=None)
        resume_result, research_result = asyncio.run(_run_phase1(jd_info, kb))
        progress.update(t1, completed=True)
        progress.update(t2, completed=True)

    console.print("[green]✓ Phase 1 完成[/green]")

    # ── Human-in-the-Loop ──────────────────────────────────
    console.print("\n[bold cyan]⏸ 公司研究结果如下，请确认：[/bold cyan]")
    console.print(Markdown(research_result))
    console.print("\n[yellow]如有补充或纠正请输入，直接回车跳过：[/yellow]", end="")
    correction = input().strip()
    if correction:
        research_result += f"\n\n**补充信息（用户提供）：** {correction}"

    # ── Phase 2 ────────────────────────────────────────────
    console.print("\n[bold]Phase 2：生成面试题库...[/bold]")
    interview_result = interview_run(jd_info, research_result)
    console.print("[green]✓ Phase 2 完成[/green]")

    # ── Assemble Report ────────────────────────────────────
    today = date.today().strftime("%Y%m%d")
    output_path = f"outputs/{jd_info['company']}_{today}.md"
    Path("outputs").mkdir(exist_ok=True)

    report = f"""# 求职准备报告 — {jd_info['company']} {jd_info['role']}

**生成日期：** {date.today()}

---

## 一、定制简历

{resume_result}

---

## 二、公司研究

{research_result}

---

## 三、面试题库

{interview_result}
"""
    Path(output_path).write_text(report, encoding="utf-8")

    # ── Display & Save ─────────────────────────────────────
    console.print("\n")
    console.print(Panel(Markdown(report), title=f"✅ 报告完成 → {output_path}", border_style="green"))

    save_application(
        company=jd_info["company"],
        role=jd_info["role"],
        role_type=jd_info["role_type"],
        jd_keywords=jd_info["key_skills"],
        output_path=output_path,
    )
    console.print(f"\n[dim]已记录到 memory/history.json[/dim]")


if __name__ == "__main__":
    main()
