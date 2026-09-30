# GraphProof-QA

Small-model question answering that compiles to an executable graph query, with a proof trace for every answer.

Fine-tunes Qwen2.5-1.5B-Instruct to translate MetaQA movie questions into a tiny typed path-query DSL instead of answering directly. Decoding is grammar-constrained to the KG schema + entity trie, the query executes over the graph, and every answer ships with its executed path as proof.

## Why

Aggregate QA accuracy can't tell memorisation from grounding. Compile-then-execute separates them:
- **Executable-query rate**: how often the model writes a query that parses and runs
- **Failure breakdown**: entity ambiguity, relation direction, hop count, parse errors
- **Hard conditions**: renamed entities (never seen in training), deleted triples (correct answer: "not in graph")

On an edited graph the executor follows the edit **by construction** — a demonstration, not a discovery. The renamed-entity and deleted-triple results are the real measurements.

## Status

Milestone 1 (DSL + executor) in progress. See `AGENTS.md`.

## License

MIT
