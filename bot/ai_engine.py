import os
import time
import requests

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))


class AIEngine:
    """Talks to OpenRouter with automatic model fallback on rate limits/errors."""

    def __init__(self):
        self.api_key = os.environ.get("OPENROUTER_API_KEY", "")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.model_name = os.environ.get(
            "OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free"
        )
        raw = os.environ.get("OPENROUTER_FALLBACK_MODELS", "")
        self.fallback_models = [m.strip() for m in raw.split(",") if m.strip()]
        # Last verified against the live free catalog on 2026-10-06.
        # Free tier: 20 req/min, 50 req/day on accounts without credits.

    # ------------------------------------------------------------------ #
    # Low-level call
    # ------------------------------------------------------------------ #
    def _call_openrouter(self, system_prompt: str, user_prompt: str,
                         max_tokens: int = 2048, temperature: float = 0.3) -> tuple[str, str]:
        """Returns (text, model_used). Raises RuntimeError with a friendly message."""
        if not self.api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not configured. Add it to bot/.env and restart the server."
            )

        models = [self.model_name] + [
            m for m in self.fallback_models if m != self.model_name
        ]
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "Summary AI Bot",
            "Content-Type": "application/json",
        }
        payload = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        last_error = "Unknown error"
        for model in models:
            payload["model"] = model
            for attempt in range(2):  # one retry per model
                try:
                    response = requests.post(
                        self.api_url, headers=headers, json=payload, timeout=120
                    )
                    if response.status_code == 429:
                        # Rate limited: wait briefly and retry, then next model.
                        retry_after = float(response.headers.get("retry-after", 4))
                        time.sleep(min(retry_after, 8))
                        last_error = f"Rate limited ({model})"
                        continue
                    response.raise_for_status()
                    data = response.json()
                    content = (data["choices"][0]["message"]["content"] or "").strip()
                    if not content:
                        last_error = f"Empty response from {model}"
                        continue
                    return content, model
                except requests.HTTPError as e:
                    status = e.response.status_code if e.response is not None else 0
                    last_error = f"HTTP {status} from {model}"
                    break  # try the next model
                except (requests.ConnectionError, requests.Timeout) as e:
                    last_error = f"Connection problem with {model}: {e.__class__.__name__}"
                    break
                except (KeyError, IndexError, ValueError) as e:
                    last_error = f"Unexpected response format from {model}"
                    break
            else:
                pass
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    # ------------------------------------------------------------------ #
    # Operations
    # ------------------------------------------------------------------ #
    def generate_summary(self, text: str, length: str = "short") -> tuple[str, str]:
        """Summarise the document. Returns (summary, model)."""
        length_prompts = {
            "short": (
                "Summarize this document in 4-6 crisp bullet points that capture the "
                "core message. Keep each bullet under 25 words. End with one line "
                "starting '**Bottom line:**' stating the single most important takeaway."
            ),
            "medium": (
                "Summarize this document in 2 short sections: '**Overview**' (2-3 "
                "sentences) and '**Key Points**' (5-8 bullets). Mention concrete facts, "
                "numbers, and names where present."
            ),
            "detailed": (
                "Produce a structured summary with markdown headings ('##') for each "
                "major topic in order, detailed bullets under each, and a final "
                "'## Conclusion' section. Preserve specific figures, terms and examples."
            ),
            "findings": (
                "Extract the top key findings, metrics, and notable data points from this document. "
                "Format as markdown with '**Key Insights & Findings**', bullet points with bold keywords, "
                "and any statistical or measurable data highlighted."
            ),
            "faq": (
                "Generate 5 essential Questions and Answers (Q&A / FAQ) that explain the core contents "
                "of this document in detail. Format each item as '### Q: ...' followed by '**A:** ...'."
            ),
        }
        system_prompt = (
            "You are an expert document analyst. Summarize accurately using ONLY the "
            "document text provided. Never invent facts. Use markdown formatting. "
            "Do not add commentary about yourself."
        )
        return self._call_openrouter(
            system_prompt,
            f"{length_prompts.get(length, length_prompts['short'])}\n\n---\n\nDOCUMENT:\n{text}",
            max_tokens=2500 if length == "detailed" else 1400,
            temperature=0.2,
        )

    def answer_question(self, context: str, question: str) -> tuple[str, str]:
        """Answer strictly from retrieved chunks. Returns (answer, model)."""
        system_prompt = (
            "You are a precise document assistant. Answer the user's question using "
            "ONLY the provided document excerpts. Cite the page number you used in "
            "the format (p. N) after each claim when page markers are present in the "
            "excerpts. If the excerpts do not contain the answer, reply exactly: "
            "'I could not find that information in the document.' Never guess."
        )
        return self._call_openrouter(
            system_prompt,
            f"DOCUMENT EXCERPTS:\n{context}\n\nQUESTION: {question}",
            max_tokens=1200,
            temperature=0.2,
        )

    def suggest_questions(self, text: str) -> tuple[str, str]:
        """Suggest 3-4 questions the user could ask about the document."""
        system_prompt = (
            "You generate thoughtful reading-comprehension questions about a document. "
            "Output ONLY 3 questions, one per line, numbered 1., 2., 3. No other text. "
            "Each question must be answerable from the document excerpt given."
        )
        return self._call_openrouter(
            system_prompt,
            f"DOCUMENT EXCERPT:\n{text}",
            max_tokens=300,
            temperature=0.5,
        )
