import json
from datetime import datetime


def export_markdown_report(result: dict) -> str:
    """Generate a clean, structured Markdown report from meeting insights."""
    title = result.get("title", "Meeting Report")
    summary = result.get("summary", "")
    action_items = result.get("action_items", "")
    key_decisions = result.get("key_decisions", "")
    open_questions = result.get("open_questions", "")
    transcript = result.get("transcript", "")

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = f"""# {title}

> **Generated on:** {timestamp}  
> **Author / Engine:** AI Video Assistant (Shubham Sura)

---

## 📋 Executive Summary
{summary}

---

## ✅ Action Items
{action_items}

---

## 🔑 Key Decisions
{key_decisions}

---

## ❓ Open Questions & Follow-ups
{open_questions}

---

## 📜 Full Transcript
```text
{transcript}
```
"""
    return md_content


def export_json_report(result: dict) -> str:
    """Generate JSON report string containing structured meeting analysis."""
    export_data = {
        "title": result.get("title", ""),
        "timestamp": datetime.now().isoformat(),
        "generated_by": "AI Video Assistant - Shubham Sura",
        "summary": result.get("summary", ""),
        "action_items": result.get("action_items", ""),
        "key_decisions": result.get("key_decisions", ""),
        "open_questions": result.get("open_questions", ""),
        "transcript": result.get("transcript", ""),
    }
    return json.dumps(export_data, indent=2)
