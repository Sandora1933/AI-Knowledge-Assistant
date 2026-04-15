# prompts.py

def build_rag_prompt(message, history, knowledge):
    return f"""
You are an assistant which answers questions based on knowledge which is provided to you.

Rules:
- Answer only from the provided knowledge in the "The knowledge" section.
- Do NOT use any prior knowledge.
- If the answer is not explicitly stated in the knowledge, say: "No relevant information found in documents".
- If the answer is not contained in the knowledge, say that the information is not available.
- Do not mention anything about the provided knowledge.

The question:
{message}

Conversation history:
{history}

The knowledge:
{knowledge}
"""

def build_judge_prompt(answer, docs):
    chunks_text = []

    for i, doc in enumerate(docs, start=1):
        chunk_text = _truncate_chunk(doc.page_content, max_len=2000)
        chunks_text.append(f"[{i}] {chunk_text}")

    joined_chunks = "\n\n".join(chunks_text)

    prompt = f"""
You are an evaluation model for a retrieval-augmented generation system.

Your task is to assess how well the assistant answer is supported by the retrieved document chunks.

Rules:
1. Evaluate only whether the answer is supported by the retrieved chunks.
2. Do not evaluate writing style.
3. If the answer is exactly "No relevant information found in documents", return confidence_score as null.
4. Confidence score must be an integer from 0 to 100.
5. High score means the answer is strongly supported by the retrieved chunks.
6. Low score means the answer is weakly supported, partially supported, or unsupported.
7. Return ONLY valid JSON with this schema:

{{
  "confidence_score": <integer 0-100 or null>,
  "reason": "<short explanation>"
}}

Assistant answer:
{answer}

Retrieved chunks:
{joined_chunks}
"""
    return prompt.strip()


def _normalize_whitespace(text):
    return " ".join((text or "").split())

def _truncate_chunk(text, max_len=1200):
    text = _normalize_whitespace(text)
    return text if len(text) <= max_len else text[:max_len] + " ..."