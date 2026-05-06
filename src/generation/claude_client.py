"""
Generation client — provider priority (first key found wins):
  1. Groq    (GROQ_API_KEY)    — free 100k TPD, fast
  2. Gemini  (GEMINI_API_KEY)  — free tier, use when Groq hits daily limit
  3. Ollama  (OLLAMA_MODEL)    — fully local, zero limits, no key needed
  4. Anthropic (ANTHROPIC_API_KEY) — paid, best quality + prompt caching

Ollama setup (one-time):
  1. Install from https://ollama.com
  2. Run: ollama pull llama3.2
  3. Set in .env: OLLAMA_MODEL=llama3.2
"""

from __future__ import annotations

import os
import json
import urllib.request

_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_GROQ_MODEL      = "llama-3.3-70b-versatile"
_GEMINI_MODEL    = "gemini-2.0-flash"
_OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

_SYSTEM_PROMPT = """You are a customer support specialist assistant. You have access to a knowledge base of historical support tickets and their resolutions.

When given a new customer ticket, you will:
1. Analyze the customer's problem carefully
2. Review the provided similar historical tickets and their resolutions
3. Synthesize a clear, actionable resolution for the current ticket
4. Cite which historical tickets informed your recommendation (by ticket ID)

Guidelines:
- Be specific and step-by-step in your resolution
- If historical tickets have conflicting resolutions, note this and recommend the most reliable approach
- If the context doesn't contain enough information, say so and suggest escalation paths
- Match the language/tone of a professional support agent
- Keep responses concise but complete"""


def _format_context(hits: list[dict]) -> str:
    parts = []
    for i, hit in enumerate(hits, start=1):
        ticket_id = hit.get("ticket_id", "unknown")
        category  = hit.get("metadata", {}).get("category", "")
        lang      = hit.get("metadata", {}).get("language", "")
        score     = hit.get("rerank_score", hit.get("rrf_score", 0))

        header = f"[Ticket {i} | ID: {ticket_id}"
        if category:
            header += f" | Category: {category}"
        if lang:
            header += f" | Language: {lang}"
        header += f" | Relevance: {score:.3f}]"
        parts.append(f"{header}\n{hit['text']}")

    return "\n\n---\n\n".join(parts)


def _generate_groq(system: str, context_text: str, new_ticket: str) -> str:
    from groq import Groq
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=_GROQ_MODEL,
        max_tokens=1024,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Here are the most relevant historical tickets from our knowledge base:\n\n{context_text}"
                f"\n\nNew customer ticket to resolve:\n\n{new_ticket}"
            )},
        ],
    )
    return response.choices[0].message.content


def _generate_gemini(system: str, context_text: str, new_ticket: str) -> str:
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    response = client.models.generate_content(
        model=_GEMINI_MODEL,
        contents=(
            f"Here are the most relevant historical tickets from our knowledge base:\n\n{context_text}"
            f"\n\nNew customer ticket to resolve:\n\n{new_ticket}"
        ),
        config=types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=1024,
        ),
    )
    return response.text


def _generate_ollama(system: str, context_text: str, new_ticket: str) -> str:
    model = os.environ["OLLAMA_MODEL"]
    payload = json.dumps({
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Here are the most relevant historical tickets from our knowledge base:\n\n{context_text}"
                f"\n\nNew customer ticket to resolve:\n\n{new_ticket}"
            )},
        ],
    }).encode()

    req = urllib.request.Request(
        f"{_OLLAMA_BASE_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def _generate_anthropic(system: str, context_text: str, new_ticket: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Here are the most relevant historical tickets from our knowledge base:\n\n{context_text}",
                    "cache_control": {"type": "ephemeral"},
                },
                {
                    "type": "text",
                    "text": f"\n\nNew customer ticket to resolve:\n\n{new_ticket}",
                },
            ],
        }],
    )
    return response.content[0].text


def generate_response(new_ticket: str, hits: list[dict]) -> str:
    """Route to whichever provider has a key configured."""
    context_text = _format_context(hits)

    if os.getenv("GROQ_API_KEY"):
        return _generate_groq(_SYSTEM_PROMPT, context_text, new_ticket)
    elif os.getenv("GEMINI_API_KEY"):
        return _generate_gemini(_SYSTEM_PROMPT, context_text, new_ticket)
    elif os.getenv("OLLAMA_MODEL"):
        return _generate_ollama(_SYSTEM_PROMPT, context_text, new_ticket)
    else:
        return _generate_anthropic(_SYSTEM_PROMPT, context_text, new_ticket)
