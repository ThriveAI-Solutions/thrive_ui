import os
import sys

import pytest

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss")


def _has_ollama():
    try:
        import httpx

        with httpx.Client(base_url=OLLAMA_URL, timeout=2.0) as client:
            r = client.get("/api/tags")
            return r.status_code == 200
    except Exception:
        return False


@pytest.mark.skipif(not _has_ollama(), reason="Ollama not available on localhost:11434")
def test_streams_include_thinking_tokens_if_present():
    """Integration: stream one reply from Ollama and capture raw chunks to see CoT tokens."""
    import httpx

    # Minimal chat request; adjust model via env if needed
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": "Solve 17*19 step by step and show your reasoning."},
        ],
        "stream": True,
    }

    chunks = []
    with httpx.Client(base_url=OLLAMA_URL, timeout=None) as client:
        with client.stream("POST", "/api/chat", json=payload) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                # Each line is a JSON object per chunk
                try:
                    import json

                    obj = json.loads(line)
                except Exception:
                    # Collect raw in case of partials
                    chunks.append(line.decode("utf-8", errors="ignore") if isinstance(line, bytes) else str(line))
                    continue
                content = obj.get("message", {}).get("content")
                if content:
                    chunks.append(content)

    full = "".join(chunks)

    # Print to stdout for developer visibility when running locally
    print("\n--- RAW STREAM CONCAT ---\n", full[:2000], "\n--- END ---\n")

    # This assertion is lenient: we just verify we received any streamed tokens
    assert any(chunks), "No streamed content received from Ollama. Is the model loaded?"

    # Heuristic check: thinking models often emit <think>...</think> or the words 'Thinking...' in output
    # We don't fail if absent, but we note it so developers can see the printout above.
    # If you expect strict CoT visibility, switch to assert and tune model/template.
    _has_think_markup = ("<think>" in full and "</think>" in full) or ("Thinking" in full)
    # Soft assertion: ensure test doesn't fail on non-thinking templates
    if not _has_think_markup:
        sys.stdout.write("Note: No explicit CoT markers (<think> or 'Thinking') detected in stream.\n")
