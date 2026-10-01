# AGENTS.md — graphproof-qa

Second portfolio project for Raihan Sikder. Target roles: Noeon Research (Senior ML Engineer LLMOps, Senior Backend Automated Reasoning), VISASQ, Treasure AI, LegalOn — all Tokyo/Japan.

Sibling project: `../flipgate/` (FlipGate release gate — own venv, don't touch it).
Prep context: `../../noeon-prep/` (CLAUDE.md has background; `content/library/04-neuro-symbolic-and-noeon.md` is the domain guide for this project).
Prior art: `../../ILMA/` (his C compiler: lexer → recursive-descent parser → 22-node AST → codegen. The DSL/parser/executor skill transfers directly.)

---

## Active project: GraphProof-QA

Fine-tune Qwen2.5-1.5B-Instruct to translate MetaQA movie questions into a tiny typed path-query DSL instead of answering directly. Grammar-constrain decoding to the KG schema + entity trie, execute over the graph, and compare compile-then-execute vs direct answering — including on edited graphs.

**Publishes:** executable-query rate + 3-hop Hits@1, constrained vs direct model (same base, same data), on unedited KB and renamed-entity condition.

### Honesty rules (load-bearing)

- Executor following an edited graph is **true by construction** — demonstration, not discovery. Say so in every write-up.
- The **real results**: (a) parse+execute rate, (b) failure breakdown, (c) renamed-entity condition, (d) deleted-triple abstain rate.
- MetaQA is template-generated — add 100 hand-written paraphrases as robustness check and say so.
- Only real data slices; no synthetic test records presented as real.

### Key design decisions

- **DSL first.** Path-query language: start entity, relation hops with direction, optional filter. Lark grammar. Executor = plain dict adjacency index over MetaQA KB (~135k triples, fits RAM).
- **Gold queries from qtype files.** Executing gold queries must reproduce gold answers on ≥99% of training questions; rest listed + explained.
- **Three systems, same base/data:** (A) LoRA direct-answer, (B) LoRA writes DSL unconstrained, (C) B + xgrammar schema/entity-trie constraint. The A-vs-C gap is the paper.
- **CF conditions as HF dataset** (`MetaQA-CF`): 500 swapped triples, 200 renamed entities (unseen in training), 300 deleted triples (correct answer: "not in graph").
- **Failure taxonomy** (100 hand-labelled): entity ambiguity, relation direction, hop count, parse error.

### Hardware

One RTX 3060 12GB. Model: Qwen/Qwen2.5-1.5B-Instruct. LoRA on ~30k Q/query pairs ≈ 1–2h/run. Eval on fixed 2,000-question sample/hop level takes minutes. Optional ~$3 A40 rental for 7B scaling point.

### Stack

Python, PyTorch, HF Transformers + PEFT (LoRA), xgrammar (schema grammar + ~43k entity trie), Lark (DSL parser), dict adjacency executor.

### Data (all pinned with checksums)

- MetaQA official: `kb.txt`, 1/2/3-hop QA + qtype files — Google Drive via `gdown`, checksums in `configs/metaqa_checksums.json`. NOT the 10GB HF mirror with subgraphs.
- `data/cf/` — swapped/renamed/deleted conditions, released as HF dataset.

### Milestones (20 days)

1. **DSL + executor** (4d) — ✅ COMPLETE (29 Sep 2026). Lark grammar + EXCEPT clause, dict executor with proof traces, gold queries for full train (1-hop 90,980/96,106; 2-hop 118,948/118,980; 3-hop 114,196/114,196; 324,124 total, 100% verified reproduction). Remainder = ambiguous-title 1-hop questions, listed in `data/remainder_report.json`. 26 tests green.
2. **Three systems** (6d) — ✅ COMPLETE. LoRA A/B/C on fixed 2,000-question test samples per hop:
   - System A (direct-answer LoRA, loss 1.28→0.57): hop1 14.2%, hop2 22.2%, hop3 66.3%, overall 34.2%. Hallucinates plausible entities.
   - System B (DSL LoRA unconstrained, loss 0.69→0.17): hop1 97.2%, hop2 89.7%, hop3 93.0%, overall 93.3%, 0 parse fails, 279 exec-empty.
   - System C (B + xgrammar schema/shape constraint): hop1 98.3%, hop2 97.2%, hop3 94.9%, overall 96.8%, 0 parse fails, 127 exec-empty.
   - A→B gap: +59.1 pts. B→C gap: +3.5 pts (constraint helps most on hop2: +7.5).
   - Trie compile: skeleton 0.5s; full 43k-entity trie ~15s (compiled, not enforced — enforcing it distorts greedy entity selection, documented open issue).
3. **Edited/incomplete graphs** (5d) — ✅ COMPLETE. MetaQA-CF conditions (seed 123):
   - Swapped (500 triple swaps): B follows edit to new answers 95.6% (demonstration, true by construction).
   - Renamed (200 novel entities, 531 test questions): B 81.4% vs A 5.8% (**+75.6 pts** — grounding vs memorisation, the co-headline).
   - Deleted (300 triples → 99 true-disconnect cases): B abstains 8.1%, A 0.0%. Neither abstains; B's wrong queries fail silently (important negative result).
   - Failure taxonomy (100 B misses): relation 83, no-gold-path 17, parse/entity/direction/hop 0.
4. **Failure analysis + release** (5d) — IN PROGRESS.
   - Failure taxonomy (100 B misses): relation 83, no-gold-path 17, parse/entity/direction/hop 0.
   - Released: adapters `raihan-js/graphproof-dsl-1.5b` + `raihan-js/graphproof-direct-1.5b`; dataset `raihan-js/MetaQA-CF` (swaps, renames, deletions + eval sets + manifest).
   - TODO: write-up (dev.to); 100 hand-written paraphrase robustness check (**needs Raihan**); GitHub push retries (network failing, commits safe locally).

### Risks to keep honest

- Edited-graph following is by construction — never sell as reasoning.
- Template-generated questions flatter scores — 100 hand paraphrases required.
- Entity-trie compile can be slow — cache it, report the time.
- batch_invariant / vLLM lessons from flipgate apply: measure, don't assume.

---

## Conventions

- Python 3.10+, pytest for DSL parser, executor, scorers.
- Own venv (`.venv/`); torch cu121 matching system CUDA 12.1.
- Per-item JSONL results, append-only; every claim cites run ID.
- No LLM-as-judge. Rule-based checks + graph execution only.
- LoRA adapters versioned; base model hash pinned in manifest.
