"""Turn the claude.ai data export (conversations.json) into one Markdown file per chat (M6).

Usage:  python scripts/export_chats.py private/conversations-000.zip "Speech recognition coding challenge prep=01-session-M1-M4" "M5 project handover=02-session-M5-M6"

Everything in each chat is kept, in order: user and assistant text, the assistant's reasoning
summaries, every tool call (name + input) and every tool result (in a collapsible block), system
notes, and references to images/files (the export holds no image data). The account field (e-mail)
is not written. Output: private/chats/<name>.md (git-ignored; the chats are
shared with the reviewers by e-mail, not on GitHub)
"""
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def fence(text: str, lang: str = "") -> str:
    """Code block whose fence is longer than any backtick run inside the text."""
    longest, run = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    ticks = "`" * max(3, longest + 1)
    return f"{ticks}{lang}\n{text}\n{ticks}"


def result_text(content) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for item in content or []:
        if item.get("type") == "text":
            parts.append(item.get("text", ""))
        else:
            parts.append(f"[{item.get('type')} content]")
    return "\n".join(parts)


def render_block(block) -> str:
    kind = block.get("type")
    if kind == "text":
        return block.get("text", "").strip()
    if kind == "thinking":
        summaries = [s.get("summary", "") for s in block.get("summaries") or []]
        text = block.get("thinking") or " · ".join(s for s in summaries if s)
        return f"> *Thinking: {text.strip()}*" if text.strip() else ""
    if kind == "tool_use":
        body = json.dumps(block.get("input"), ensure_ascii=False, indent=2)
        return f"**Tool call — `{block.get('name')}`**\n\n{fence(body, 'json')}"
    if kind == "tool_result":
        error = " (error)" if block.get("is_error") else ""
        return (f"<details><summary>Tool result — {block.get('name')}{error}</summary>\n\n"
                f"{fence(result_text(block.get('content')))}\n\n</details>")
    if kind == "injected_prompt_block":
        note = block.get("prompt", "").strip()
        if "\n" not in note:
            return f"> *System note: {note}*"
        return f"<details><summary>System note</summary>\n\n{fence(note)}\n\n</details>"
    if kind in ("image", "document"):
        name = block.get("file_name") or block.get("title") or block.get("file_uuid") or ""
        return f"*[{kind} attached: {name}]*"
    raise ValueError(f"Unknown block type {kind!r} — update export_chats.py")  # fail loudly


def render_chat(chat) -> str:
    lines = [f"# {chat['name']}", "",
             f"Exported from claude.ai. Created {chat['created_at']}, last updated {chat['updated_at']}, "
             f"{len(chat['chat_messages'])} messages. Rendered in full by `scripts/export_chats.py`.", ""]
    for message in chat["chat_messages"]:
        who = "User" if message["sender"] == "human" else "Claude"
        lines += ["---", "", f"## {who} — {message['created_at'][:19].replace('T', ' ')} UTC", ""]
        blocks = message.get("content") or [{"type": "text", "text": message.get("text", "")}]
        for block in blocks:
            text = render_block(block)
            if text:
                lines += [text, ""]
        for f in (message.get("files") or []) + (message.get("attachments") or []):
            lines += [f"*[file: {f.get('file_name') or f.get('file_uuid', '')}]*", ""]
    return "\n".join(lines)


def main():
    zip_path, wanted = sys.argv[1], dict(arg.split("=", 1) for arg in sys.argv[2:])
    with zipfile.ZipFile(zip_path) as z:
        chats = json.loads(z.read("conversations.json"))
    found = {c["name"]: c for c in chats}
    missing = set(wanted) - set(found)
    if missing:
        raise SystemExit(f"Chats not in the export: {sorted(missing)}. Present: {sorted(found)}")
    out_dir = ROOT / "private" / "chats"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, out in wanted.items():
        target = out_dir / f"{out}.md"
        target.write_text(render_chat(found[name]), encoding="utf-8")
        print(f"{target.relative_to(ROOT)}: {len(found[name]['chat_messages'])} messages, "
              f"{target.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
