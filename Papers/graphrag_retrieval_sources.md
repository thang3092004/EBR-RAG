# GraphRAG Retrieval Sources

These papers support the Unified Multimodal Graph V2 retrieval design.

| Paper | Link | Decision used in this project |
|---|---|---|
| From Local to Global: A Graph RAG Approach to Query-Focused Summarization | https://arxiv.org/abs/2404.16130 | Community retrieval is useful for global questions, but all-community map-reduce is excluded from the default runtime path. |
| LightRAG: Simple and Fast Retrieval-Augmented Generation | https://arxiv.org/abs/2410.05779 | Retrieve both entity-level and relation-level evidence, then return linked source text. |
| HippoRAG | https://arxiv.org/abs/2405.14831 | Personalized PageRank provides controlled multi-hop score propagation. |
| HippoRAG 2 | https://arxiv.org/abs/2502.14802 | Passage provenance should remain attached to graph nodes and paths. |
| G-Retriever | https://arxiv.org/abs/2402.07630 | Return a compact connected subgraph instead of a large list of independent nodes. |
| PathRAG | https://arxiv.org/abs/2502.14902 | Prune noisy graph context and expose short relational paths. |
| KG2RAG | https://arxiv.org/abs/2502.06864 | Start from semantic seeds, expand structurally, and organize source chunks by graph relations. |
| GraphReader | https://arxiv.org/abs/2406.14550 | Agentic traversal is related work but excluded because it requires too many LLM calls. |
| GRAG | https://arxiv.org/abs/2405.16506 | Learned graph retrieval is excluded from the train-free core. |
| When to Use Graphs in RAG | https://arxiv.org/abs/2506.05690 | More hops can reduce accuracy; V2 defaults to two edges and permits three only as fallback. |

Implementation mapping:

- HippoRAG motivates internal Personalized PageRank with restart `0.15`.
- G-Retriever and PathRAG motivate compact evidence packets.
- LightRAG and KG2RAG motivate entity seeds plus provenance segments.
- No paper is used to justify dumping the propagated node set into the prompt.
  V2 returns at most four packets and enforces a graph token budget.

