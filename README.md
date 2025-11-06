# RAG with Real-time Observability & Monitoring

## Why This Project Matters (For Non-Technical Leaders)

**The Problem:** Organizations deploy AI tools before understanding what success looks like. They measure adoption by "did we launch it?" not "does it actually work?" 

**What This Project Shows:**
- **Observability-first thinking:** Build measurement into systems *before* optimization—detect failures automatically instead of discovering them through incidents
- **Pragmatism:** Know when NOT to optimize (scope discipline) rather than chasing diminishing returns
- **Honest communication:** Document limitations and tradeoffs instead of hiding them behind abstractions
- **Production readiness:** Think like teams that scale AI adoption, not teams that only build features

**Why This Matters for Your Organization:**

If you're scaling AI across teams (like Hitachi's Lumada strategy or Rakuten's AI-nization), this demonstrates how production teams *think*—not about shipping fast, but about shipping systems that *tell you when they're broken*. That mindset is how adoption scales from pilot to enterprise.

---

## The Technical Challenge (For Engineers)

Build a Retrieval-Augmented Generation system for [translate:Japanese Payment Services Act] documents that demonstrates observability at each pipeline stage (load → split → embed → retrieve → generate).

### Project Goal

Show production-level observability that detects failures automatically rather than just building a working system. The goal is to identify problems through metrics, not guessing.

### Architecture

PDF Document → Chunking (Japanese-aware) → Embeddings (all-minilm) → Vector Store (Pinecone) → RAG Pipeline → Observability Logging (JSON)

### What Gets Logged

- Query (input question)
- Config (top_k, temperature)
- Retrieval Scores (min/max/avg similarity)
- Response Language (English detection %)
- Response Length (characters)
- Response Text (full LLM output)

---

## Key Technical Decisions

### 1. Direct Pinecone API Over LangChain Abstraction

**The Problem:** LangChain's `similarity_search()` has a hidden score threshold that silently filters results. You don't know when data is being dropped.

**The Solution:** Switched to direct `index.query()` for full transparency. If you can't see what's happening, you can't measure it.

**Lesson for Teams:** When frameworks hide behavior, they hide problems. Go lower-level when adoption requires trust.

### 2. Language Detection (Not Control)

**The Problem:** Initial approach was prompt engineering to force English responses. But local models don't reliably follow language constraints.

**The Solution:** Instead of trying to control the model, detect what it actually produces and log it. This catches failure modes that prompt engineering would hide.

**Lesson for Teams:** Sometimes "accepting and measuring" beats "forcing and hoping."

### 3. Skip Configuration Optimization

**The Problem:** Testing chunk sizes requires re-embedding 243 vectors (~24 hours). That's outside scope for this project.

**The Solution:** Focus on observable metrics (top_k variations, language detection) instead. Not everything broken needs fixing *now*.

**Lesson for Teams:** Scope discipline separates production teams from teams that optimize forever.

---

## Key Findings

### Finding 1: Retrieval Score Issue (0.36 avg—Consistent)

**Root Cause:**
- English queries against [translate:Japanese]-only document
- Chunk splitting fragments semantic meaning
- 36-page document → 243 granular chunks

**What This Means:**
- LLM translates rather than copies from context
- Low scores indicate semantic mismatch, not poor retrieval strategy
- This is expected behavior, not a bug

**What Would Fix This:** Bilingual embedding model (out of scope for this project).

**Production Insight:** If you're scaling RAG across Japanese enterprises, this is your baseline problem. You *need* bilingual embeddings.

---

### Finding 2: Language Consistency (0% → 99% English)

**Cause:** Local llama3.2 doesn't reliably follow system prompts—a known limitation of smaller models.

**Production Solution:** Use Azure OpenAI (gpt-4o) for better instruction following in production environments.

**Organizational Insight:** If your teams are choosing between local LLMs and proprietary models, this shows the tradeoff: local = cost savings, proprietary = reliability. Both are valid choices depending on your adoption strategy.

---

## Experiments

### Experiment 1: Retrieval Depth (Varying top_k)

**Question:** Does retrieving more context improve response quality?

**Setup:** Fixed temperature (0.1), varying top_k (1, 3, 5, 10)

| top_k | Avg Score | Response Length | Language |
|-------|-----------|-----------------|----------|
| 1 | 0.36 | ~885 | 99% English |
| 3 | 0.36 | ~534 | 93% English |
| 5 | 0.35 | ~773 | 98% English |
| 10 | 0.32 | ~1838 | 99% English |

**Finding:** Higher top_k increases response length but *degrades* retrieval quality (lower avg scores). More context doesn't necessarily improve relevance.

**For Non-Technical Leaders:** This is a classic scaling problem. Giving teams access to more data doesn't automatically make them more productive—sometimes it creates noise. The same applies to AI.

---

### Experiment 2: Generation Randomness (Varying Temperature)

**Question:** Does temperature affect response quality and consistency?

**Setup:** Fixed top_k (3), varying temperature (0.0, 0.1, 0.3, 0.5, 1.0)

| Temperature | Avg Score | Response Length | Language |
|-------------|-----------|-----------------|----------|
| 0.0 | 0.36 | ~772 | 94% English |
| 0.1 | 0.36 | ~467 | 97% English |
| 0.3 | 0.36 | ~380 | 93% English |
| 0.5 | 0.36 | ~520 | 93% English |
| 1.0 | 0.36 | ~1282 | 97% English |

**Finding:** 

Temperature doesn't affect retrieval scores (as expected—temperature only affects generation, not retrieval). However:

- **Response length varies dramatically** (380 chars at 0.3 → 1282 chars at 1.0)
- **Language consistency stays high** (93-97% English across all temperatures)
- **Higher temperatures = longer, more exploratory responses** (1.0 produces 3x longer output than 0.3)

**What This Means:** Temperature controls verbosity, not reliability. If your organization needs consistent, concise answers, use temperature 0.3-0.5. If you need exploratory responses, use 1.0. Language stays English either way with this model.

**For Non-Technical Leaders:** Think of temperature like a writer's style control. Low temperature = concise, predictable. High temperature = creative, verbose. Both produce English—the difference is confidence level and length.

