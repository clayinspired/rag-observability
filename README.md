# RAG with Real-time Observability & Monitoring

Build a Retrieval-Augmented Generation system for Japanese Payment Services Act documents that demonstrates observability at each pipeline stage (load → split → embed → retrieve → generate).

## Project Goal

Show production-level observability that detects failures automatically rather than just building a working system. The goal is to identify problems through metrics, not guessing.

## Architecture

PDF Document → Chunking (Japanese-aware) → Embeddings (all-minilm) → Vector Store (Pinecone) → RAG Pipeline → Observability Logging (JSON)

## What Gets Logged

- Query (input question)
- Config (top_k, temperature)
- Retrieval Scores (min/max/avg similarity)
- Response Language (English detection %)
- Response Length (characters)
- Response Text (full LLM output)

## Key Technical Decisions

### 1. Direct Pinecone API Over LangChain Abstraction

LangChain's `similarity_search()` has a hidden score threshold that silently filters results. Switched to direct `index.query()` for transparency.

### 2. Language Detection (Not Control)

Initial approach: prompt engineering to force English. Better approach: accept that local models don't follow language constraints reliably. Instead, detect and log it.

### 3. Skip Configuration Optimization

Testing chunk sizes requires re-embedding 243 vectors (~24 hours). Instead, focus on observable metrics (top_k variations, language detection).

## Key Findings

### Retrieval Score Issue: 0.36 avg (Consistent)

**Root Cause:**
- English queries against Japanese-only document
- Chunk splitting fragments semantic meaning
- 36-page document → 243 granular chunks

**What This Means:**
- LLM translates rather than copies from context
- Low scores indicate semantic mismatch, not poor retrieval

**What Would Fix This:** Bilingual embedding model (out of scope for this project).

### Language Consistency: 0% → 99% English

**Cause:** Local llama3.2 doesn't reliably follow system prompts.

**Production Solution:** Use Azure OpenAI (gpt-4o) for better instruction following.

## Results Table

| top_k | Temp | Scores Avg | Language | Response Length |
|-------|------|-----------|----------|-----------------|
| 1 | 0.1 | 0.36 | 99% English | 885 |
| 3 | 0.1 | 0.36 | 93% English | 534 |
| 5 | 0.1 | 0.35 | 98% English | 773 |
| 10 | 0.1 | 0.32 | 99% English | 1838 |

**Pattern:** Higher top_k = longer responses, lower retrieval quality.

## Setup

pip install -r requirements.txt
export PINECONE_API_KEY="your-key"
python main.py

Requires: Ollama (locally) with all-minilm and llama3.2:3b models.

## Lessons Learned

1. **Observability > Optimization** — Detecting problems beats optimizing broken systems
2. **Know When to Stop** — Not all problems need solutions within scope
3. **Log What Matters** — Language detection caught what prompt engineering couldn't
4. **Direct APIs > Abstractions** — Go lower-level when black boxes hide behavior
5. **Document Limitations Honestly** — Shows professional maturity

## What's Next (Production)

1. Use Azure OpenAI (gpt-4o) for language control
2. Implement bilingual embeddings (multilingual-e5)
3. Add LangSmith tracing for full observability
4. Create alert thresholds for language/retrieval failures

## Why This Matters

This demonstrates:
- **Production thinking:** Observability-first design
- **Problem-solving:** Identified LangChain bug, debugged retrieval
- **Pragmatism:** Chose not to optimize what's out of scope
- **Communication:** Documented limitations honestly
- **Architecture:** Provider-agnostic code (easy Ollama → Azure swap)

You're showing you understand when to measure, accept, and move forward.
