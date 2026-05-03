"""
Generation client — uses Groq (llama-3.3-70b) if GROQ_API_KEY is set,
otherwise falls back to Anthropic Claude (claude-sonnet-4-6).

Groq: fast, free tier, OpenAI-compatible API.
Anthropic: prompt caching support (lower cost at scale).
"""

from __future__ import annotations

import os

_ANTHROPIC_MODEL = "claude-sonnet-4-6"
_GROQ_MODEL = "llama-3.3-70b-versatile"

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
        subject = hit.get("metadata", {}).get("subject", "")
        category = hit.get("metadata", {}).get("category", "")
        lang = hit.get("metadata", {}).get("language", "")
        score = hit.get("rerank_score", hit.get("rrf_score", 0))

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


def _generate_anthropic(system: str, context_text: str, new_ticket: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model=_ANTHROPIC_MODEL,
        max_tokens=1024,
        system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        messages=[
            {
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
            }
        ],
    )
    return response.content[0].text


def generate_response(new_ticket: str, hits: list[dict]) -> str:
    """
    Generate a support resolution for new_ticket given the retrieved hits.
    Uses Groq if GROQ_API_KEY is set, otherwise Anthropic.
    """
    context_text = _format_context(hits)

    if os.getenv("GROQ_API_KEY"):
        return _generate_groq(_SYSTEM_PROMPT, context_text, new_ticket)
    else:
        return _generate_anthropic(_SYSTEM_PROMPT, context_text, new_ticket)
