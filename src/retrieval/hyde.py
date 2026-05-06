"""
HyDE — Hypothetical Document Embeddings

Problem it solves:
  A query ticket is a *problem description*.
  The index contains *problem + resolution* text.
  These live in different embedding spaces, causing retrieval drift at ranks 2-5.

How it works:
  1. Send the query ticket to the LLM: "What would the resolution look like?"
  2. Embed that hypothetical resolution (not the raw query)
  3. Use that embedding for dense search — now we're searching like-for-like
  4. BM25 still uses the original query (keywords work fine as-is)
  5. RRF fuses both — HyDE improves dense, BM25 stays unchanged

Result: dense search now matches against resolutions using a resolution-shaped vector,
which is a much closer match to what's actually in the index.
"""

from __future__ import annotations

import os


_HYDE_PROMPT = """You are a customer support specialist. Given the following support ticket, write a brief but realistic resolution that an experienced support agent would provide.

Be concise (2-4 sentences). Focus on the likely fix, not the diagnosis.

Support ticket:
{ticket}

Resolution:"""


def generate_hypothetical_resolution(ticket_text: str) -> str:
    """
    Call the configured LLM to produce a hypothetical resolution for a query ticket.
    Uses the same provider priority as the main generation client:
    Groq → Gemini → Ollama → Anthropic
    """
    prompt = _HYDE_PROMPT.format(ticket=ticket_text[:1000])  # cap to avoid token waste

    if os.getenv("GROQ_API_KEY"):
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content

    elif os.getenv("GEMINI_API_KEY"):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config=types.GenerateContentConfig(max_output_tokens=150),
        )
        return resp.text

    elif os.getenv("OLLAMA_MODEL"):
        import json, urllib.request
        payload = json.dumps({
            "model": os.environ["OLLAMA_MODEL"],
            "stream": False,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            f"{os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read())["message"]["content"]

    else:
        import anthropic
        client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=150,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


def hyde_query_embedding(ticket_text: str, embedder) -> tuple[any, str]:
    """
    Generate a hypothetical resolution and embed it.
    Returns (embedding, hypothetical_text) — the text is useful for debugging.
    """
    hypothetical = generate_hypothetical_resolution(ticket_text)
    # Embed as a passage (what we're searching for matches passage-style text in index)
    embedding = embedder.embed_passages([hypothetical])[0]
    return embedding, hypothetical
