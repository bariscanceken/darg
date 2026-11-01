# DARG – notes for Claude Code and contributors

DARG is a Turkish earthquake information assistant: a hybrid RAG system over AFAD (Disaster and Emergency Management
Authority) documents, with a FastAPI server, a Flutter mobile app and a small web site. TÜBİTAK 2209-A student
research project, Karadeniz Technical University. See README.md for the overview.

## Layout
- `src/preprocessing/` PDF extraction (structure-aware, OCR cache), chunking, gazetteer, province cards
- `src/retrieval/` E5-small FAISS index, BM25, cross-encoder reranking
- `src/rag/` answer pipelines; **v3 is the default** (`pipeline_v3.py`, orchestrated as a LangGraph graph in `graph_v3.py`)
- `src/api/` FastAPI: `/soru`, `/geri-bildirim`, `/istatistik`, `/kutuphane`, organisation accounts and announcements
- `src/evaluation/` measurement scripts; results in `data/evaluation/`
- `mobile/` Flutter app · `website/` static site · `deploy/` Docker, Caddy, setup guide (`deploy/KURULUM.md`)

## Pipeline v3 in one line
question understanding (gpt-4o-mini) → scope check → place detection → wide retrieval (3 queries × dense + BM25,
province plan, source-type pools, cross-encoder, top 8) + province card lines → answer (gpt-4.1-mini): every sentence
either quotes an AFAD passage (quote verified in code) or is marked "(genel bilgi)" → hybrid decision: official web
search (gov.tr, Kızılay; gpt-4o) only for current information or when no sentence is AFAD-backed → answer again.
Safety rules: no guarantee about a specific building, no earthquake prediction, emergency questions start with 112.

## Rules
- **Measure before adopting.** Development sets and independent sets are kept apart; an independent set is written
  before the system sees it, run once, and thresholds are not changed after seeing the result.
- **Closed test sets:** Final Test v1 (`data/evaluation/final_test_v1*`, `src/evaluation/run_final_test_v1.py`) is never
  read, changed or run. Final Test v2 and `model_check_v1` were run once; their runners refuse to run again.
- Library defaults keep the Final Test v1 path (`generate_answer()`); new behaviour is opt-in (CLI flags, v3).
- Never put API keys or secrets in code; they live in `.env` (git-ignored, template `.env.example`).
- OpenAI calls cost money: tests use fake clients; ask before running anything that calls the API.
- Prefer general fixes over question-specific patches.
- Commits: lowercase English subject ending with a period, bullet-list body.

## Commands
```
python -m unittest tests.test_pipeline_v3 tests.test_graph_v3 tests.test_api tests.test_library tests.test_community
python -m src.rag.generate_answer --chat                        # ask questions in the terminal (needs OPENAI_API_KEY)
uvicorn src.api.app:app --port 8000                              # API, docs at /docs
cd mobile && flutter test                                       # mobile app tests
docker compose up -d --build                                    # server (see deploy/KURULUM.md)
```
Data pipeline (documents are not in git, see docs/VERI.md): `extract_structured` → `ocr_pages --cache-only` →
`build_document_titles` → `chunk_structured_v5` → `build_index_v2` (index `data/indexes/mobile_v5`).

## Project owner's local notes (not in git; loaded when present)
@CLAUDE.local.md
