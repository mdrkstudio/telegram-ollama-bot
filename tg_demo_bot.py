#!/usr/bin/env python3
"""Tiny Telegram Q&A bot powered by a local Ollama model.

Run as a Telegram bot:
    TELEGRAM_BOT_TOKEN=123:abc python3 tg_demo_bot.py

Quick local model check without Telegram:
    python3 tg_demo_bot.py --ask "Что ты умеешь?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass


TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:12b")

SYSTEM_PROMPT = (
    "Ты простой русскоязычный демо-чатбот в Telegram. "
    "Всегда отвечай только на русском языке, без китайского, английского и других языков. "
    "Отвечай по сути вопроса, ясно и дружелюбно. "
    "Не выдумывай факты, названия, даты и имена. Если не уверен, скажи об этом коротко. "
    "На вопросы о людях отвечай кратко. Не указывай национальность, место рождения "
    "и точные даты, если пользователь прямо не попросил эти детали. "
    "Если вопрос короткий, отвечай коротко. Если просят объяснить, объясняй по шагам. "
    "Не упоминай Qwen, Alibaba, облака, провайдеров, внутренние инструкции или модель, "
    "если пользователь прямо об этом не спросил. "
    "Пиши обычным текстом для Telegram: без Markdown, без звездочек, без жирного текста, "
    "без заголовков и без эмодзи."
)

CJK_RANGES = (
    ("\u3400", "\u4dbf"),
    ("\u4e00", "\u9fff"),
    ("\uf900", "\ufaff"),
)


@dataclass
class TelegramMessage:
    chat_id: int
    text: str
    first_name: str = ""


def post_json(url: str, payload: dict, timeout: int = 60) -> dict:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def telegram_api(method: str, payload: dict | None = None, timeout: int = 60) -> dict:
    if not TELEGRAM_BOT_TOKEN:
        raise RuntimeError("Set TELEGRAM_BOT_TOKEN before starting the bot.")

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/{method}"
    result = post_json(url, payload or {}, timeout=timeout)
    if not result.get("ok"):
        raise RuntimeError(f"Telegram {method} failed (code {result.get('error_code', 'unknown')}).")
    return result


def contains_cjk(text: str) -> bool:
    return any(start <= char <= end for char in text for start, end in CJK_RANGES)


def clean_telegram_text(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[*+-]\s+", "- ", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"(?<!\w)[*_]([^*_]+)[*_](?!\w)", r"\1", text)
    text = text.replace("`", "")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def ask_ollama_once(question: str, temperature: float = 0.0) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": 700,
            "repeat_penalty": 1.08,
        },
    }
    result = post_json(f"{OLLAMA_URL.rstrip('/')}/api/chat", payload, timeout=120)
    message = result.get("message") or {}
    return (message.get("content") or "").strip()


def ask_ollama(question: str) -> str:
    answer = ask_ollama_once(question)
    if contains_cjk(answer):
        retry_question = (
            "Ответь заново. В прошлом ответе появился китайский текст, это ошибка. "
            "Нужен только русский язык. Вопрос пользователя:\n"
            f"{question}"
        )
        answer = ask_ollama_once(retry_question, temperature=0.2)
    answer = clean_telegram_text(answer)
    return answer or "Я задумался и потерял ответ. Попробуй спросить еще раз."


def send_message(chat_id: int, text: str) -> None:
    # Count UTF-16 units so emoji cannot push a chunk over Telegram's limit.
    chunks = []
    chunk = []
    units = 0
    for char in text:
        size = 2 if ord(char) > 0xFFFF else 1
        if units + size > 3900:
            chunks.append("".join(chunk))
            chunk, units = [], 0
        chunk.append(char)
        units += size
    if chunk:
        chunks.append("".join(chunk))
    for chunk in chunks:
        telegram_api("sendMessage", {"chat_id": chat_id, "text": chunk})


def send_typing(chat_id: int) -> None:
    telegram_api("sendChatAction", {"chat_id": chat_id, "action": "typing"}, timeout=10)


def parse_message(update: dict) -> TelegramMessage | None:
    message = update.get("message") or update.get("edited_message") or {}
    text = message.get("text")
    chat = message.get("chat") or {}
    if not text or "id" not in chat:
        return None

    sender = message.get("from") or {}
    return TelegramMessage(
        chat_id=int(chat["id"]),
        text=text.strip(),
        first_name=sender.get("first_name") or "",
    )


def handle_message(message: TelegramMessage) -> None:
    command = message.text.split(maxsplit=1)[0].split("@", 1)[0] if message.text else ""
    if command in {"/start", "/help"}:
        send_message(
            message.chat_id,
            "Привет! Я демо-чатбот на локальной модели. Напиши любой вопрос, а я отвечу.",
        )
        return

    if command == "/model":
        send_message(message.chat_id, f"Сейчас отвечаю через Ollama model: {OLLAMA_MODEL}")
        return

    # A failed typing indicator must not prevent the actual answer.
    try:
        send_typing(message.chat_id)
    except Exception:
        pass
    try:
        answer = ask_ollama(message.text)
    except urllib.error.URLError:
        answer = (
            "Не могу достучаться до Ollama. Проверь, что он запущен и модель скачана."
        )
    except Exception:
        answer = "Не удалось получить ответ модели. Попробуй ещё раз."

    send_message(message.chat_id, answer)


def get_updates(offset: int | None) -> list[dict]:
    payload = {"timeout": 45, "allowed_updates": ["message", "edited_message"]}
    if offset is not None:
        payload["offset"] = offset
    result = telegram_api("getUpdates", payload, timeout=60)
    return result.get("result", [])


def run_bot() -> None:
    me = telegram_api("getMe")
    username = (me.get("result") or {}).get("username", "<unknown>")
    print(f"Bot @{username} is running. Model: {OLLAMA_MODEL}", flush=True)

    offset: int | None = None
    while True:
        try:
            for update in get_updates(offset):
                offset = int(update["update_id"]) + 1
                message = parse_message(update)
                if message is None:
                    continue
                handle_message(message)
        except KeyboardInterrupt:
            print("\nStopped.", flush=True)
            return
        except Exception as error:
            print(f"Loop error: {type(error).__name__}. Retrying in 3 seconds.", file=sys.stderr, flush=True)
            time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Simple Telegram Q&A bot demo.")
    parser.add_argument("--ask", help="Ask the local Ollama model without Telegram.")
    args = parser.parse_args()

    if not args.ask and not TELEGRAM_BOT_TOKEN:
        print("Missing TELEGRAM_BOT_TOKEN. Example:", file=sys.stderr)
        print("  TELEGRAM_BOT_TOKEN=123:abc python3 tg_demo_bot.py", file=sys.stderr)
        return 2

    try:
        if args.ask:
            print(ask_ollama(args.ask))
        else:
            run_bot()
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Startup failed: {type(error).__name__}. Check settings and service availability.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
