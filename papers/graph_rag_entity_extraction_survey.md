# Graph RAG Entity Extraction — Literature Survey

## Kết luận chính

**Tất cả các hệ thống GraphRAG hàng đầu đều dùng LLM (GPT-4/GPT-4o-mini) để trích xuất entities và relationships, KHÔNG dùng spaCy hay NER truyền thống.** Lý do: LLM hiểu context, xử lý mô tả generic ("a large male lion"), infer relationships từ ngữ nghĩa — NER chỉ detect named entities cứng.

---

## 1. Microsoft GraphRAG (2024) — Paper gốc, 20K+ GitHub stars

**Paper:** "From Local to Global: A GraphRAG Approach to Query-Focused Summarization"
**ArXiv:** https://arxiv.org/abs/2404.16130
**GitHub:** https://github.com/microsoft/graphrag
**Prompts:** https://github.com/microsoft/graphrag/blob/main/graphrag/prompt_tune/prompt/entity_relationship.py

**Entity extraction method:**
- **GPT-4 with domain-specific prompts + few-shot examples**
- Extract entities + relationships **cùng 1 call** (workflow `extract_graph`)
- Few-shot examples auto-tuned cho từng domain

**Chi tiết pipeline indexing:**

```
Phase 1: Text Chunking
  - Split documents thành text units (600-1200 tokens/chunk)
  - Smaller chunks = more entities extracted but lose coreference context
  - Recommended: 600 tokens for quality, 1200 for cost savings

Phase 2-3: Entity & Relationship Extraction (CORE)
  - Prompt LLM per chunk, extract IN ONE CALL:
    Entities:
      - entity_name: Capitalized, consistent naming
      - entity_type: One of predefined types (person, organization, location, event, ...)
      - entity_description: "Comprehensive description of attributes and activities"
    Relationships:
      - source_entity: name
      - target_entity: name  
      - relationship_description: why they're related
      - relationship_strength: 1-10 integer score

Phase 4: Graph Augmentation
  - Entity resolution: merge duplicates by name similarity
  - Community detection: Leiden algorithm for hierarchical clustering

Phase 5: Community Summarization
  - LLM generates summary report per community
  - Used for global queries

Phase 6: Retrieval
  - Local search: entity-based
  - Global search: community-summary-based
```

**Prompt format (simplified from source):**
```
-Goal-
Given a text document, identify all entities and relationships.

-Steps-
1. Identify all entities. For each entity:
   - entity_name: Name of the entity, capitalized
   - entity_type: One of [person, organization, location, event, concept]
   - entity_description: Comprehensive description

2. From entities identified, identify all relationships. For each:
   - source_entity, target_entity
   - relationship_description  
   - relationship_strength (1-10)

3. Return output as list of (entity/relationship) tuples.

-Real Data-
Entity types: {entity_types}
Text: {input_text}
Output:
```

**Key insight:** Extraction + relationship extraction in ONE call. No separate NER step.

---

## 2. LightRAG (2024) — EMNLP 2025, lightweight alternative

**Paper:** "LightRAG: Simple and Fast Retrieval-Augmented Generation"
**GitHub:** https://github.com/HKUDS/LightRAG
**ArXiv:** https://arxiv.org/abs/2410.05779
**Prompts:** https://github.com/HKUDS/LightRAG/blob/main/lightrag/prompt.py
**Neo4j deep-dive:** https://neo4j.com/blog/developer/under-the-covers-with-lightrag-extraction/

**Chi tiết pipeline:**

```
Phase 1: Document Chunking
  - Split text into chunks
  - Clean multimodal markup (strip_internal_multimodal_markup_for_extraction)

Phase 2: LLM Entity + Relation Extraction (per chunk)
  - System prompt: "You are a Knowledge Graph Specialist"
  - Extract entities AND relationships in ONE call (giống GraphRAG)
  - Output format: tuples delimited by special markers
  
  Entity tuple format:
    ("entity"{delimiter}<entity_name>{delimiter}<entity_type>{delimiter}<entity_description>)
  
  Relationship tuple format:
    ("relationship"{delimiter}<source>{delimiter}<target>{delimiter}<description>{delimiter}<strength>)

Phase 3: Gleaning (iterative refinement)
  - After first extraction, prompt LLM again: "Did you miss anything?"
  - More aggressive prompt forces coverage of ALL entities
  - Configurable: max_gleaning_rounds (default 1-2)
  - Merges new findings with existing ones

Phase 4: Deduplication
  - Merge identical entities within same chunk or across gleaning rounds
  - Entity name truncation (max length)
  - Store into graph + vector DB

Phase 5: Dual-Level Retrieval
  - Low-level: entity-specific retrieval
  - High-level: topic/theme retrieval
  - Combines both for query answering
```

**Prompt template (from source code):**
```
---Goal---
Given a text document that is potentially relevant to this activity and a list of 
entity types, identify all entities of those types from the text and all 
relationships among the identified entities.

---Steps---
1. Identify all entities. For each identified entity, extract:
- entity_name: Name of the entity, capitalized
- entity_type: One of [{entity_types}]  
- entity_description: Comprehensive description of entity's attributes and activities

2. From entities identified in step 1, identify all pairs of (source_entity, target_entity) 
that are *clearly related* to each other.
For each pair, extract:
- source_entity: name of source entity
- target_entity: name of target entity  
- relationship_description: explanation of relationship
- relationship_strength: integer 1-10
- relationship_keywords: one or more high-level key words that summarize the relationship

3. Return output in specified format.

######################
-Examples-
######################
{examples}

######################
-Real Data-
######################
Entity_types: [{entity_types}]
Text:
{input_text}
######################
Output:
```

**Key differences from GraphRAG:**
- Adds `relationship_keywords` (high-level themes for retrieval)
- Gleaning mechanism (retry to catch missed entities)
- Simpler post-processing (no community detection)
- Dual-level retrieval (entity + topic)

---

## 3. KGGen (2025) — NeurIPS 2025, state-of-the-art

**Paper:** "KGGen: Extracting Knowledge Graphs from Plain Text with Language Models"
**ArXiv:** https://arxiv.org/abs/2502.09956
**GitHub:** https://github.com/stair-lab/kg-gen
**Package:** `pip install kg-gen`

**Entity extraction method:**
- **Fine-tuned LLM** specifically for KG extraction
- Novel entity resolution: cluster related entities to reduce sparsity
- Combines triplet extraction + coreference resolution
- No predefined schema required

**Key innovation:** Addresses the sparsity problem (nhiều nodes isolated) bằng cách cluster related entities — exactly our problem!

**Benchmark:** MINE (Measure of Information in Nodes and Edges) — first benchmark for text-to-KG quality.

---

## 4. Neo4j LLM Knowledge Graph Builder (2024)

**URL:** https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/
**Blog:** https://neo4j.com/blog/developer/llm-knowledge-graph-builder-release/

**Entity extraction method:**
- **LLM-based** (OpenAI, Gemini, Llama3, Claude, Qwen)
- Three-step: extract nodes + relationships → entity disambiguation → import
- Optional: user-provided schema to guide extraction
- Supports multimodal: PDFs, images, YouTube transcripts

---

## 5. Video-specific: From Videos to Indexed Knowledge Graphs (2025)

**Paper:** "From Videos to Indexed Knowledge Graphs"
**ArXiv:** https://arxiv.org/abs/2510.01513

**Method:** Framework marries methods for multimodal content analysis — LLM processes visual descriptions + transcripts to build KG.

---

## 6. VHAKG (2024) — Video Knowledge Graph

**Paper:** "VHAKG: A Multi-modal Knowledge Graph Based on Synchronized Multi-view Videos"
**ArXiv:** https://arxiv.org/abs/2408.14895

**Method:** Encodes multi-view, event-centric video knowledge graphs down to temporal (frame) and spatial (bounding box) level.

---

## 7. KET-RAG (2025) — Cost-efficient GraphRAG

**Paper:** "KET-RAG: A Cost-Efficient Multi-Granular Indexing Framework for Graph-RAG"
**ArXiv:** https://arxiv.org/abs/2502.09304

**Method:** Multi-granular indexing combining keyword extraction with entity-level graph construction.

---

## So sánh approaches

| Approach | Tool | Pros | Cons |
|---|---|---|---|
| **LLM extraction (GPT-4)** | GraphRAG, LightRAG, KGGen | Hiểu context, xử lý generic descriptions, infer relationships | Tốn API cost, chậm |
| **spaCy NER** | Traditional NLP | Nhanh, free, deterministic | Chỉ detect named entities, miss generic descriptions, không extract relationships |
| **YOLO + tracking** | Custom CV pipelines | Detect visual objects | Hallucinate (banana, pizza), không hiểu semantics |
| **VLM caption + LLM extract** | Hybrid (EBR-RAG proposed) | Best of both: visual perception + semantic extraction | 2 model calls |

---

## Recommendation cho EBR-RAG

Dựa trên literature, approach đúng cho entity extraction từ captions là:

**Dùng GPT-4o-mini extract entities từ captions (giống GraphRAG/LightRAG), KHÔNG dùng spaCy.**

Lý do:
1. Captions mô tả entities generic ("a large male lion walking from the left") — spaCy NER không catch
2. spaCy noun chunks extract quá nhiều noise ("the scene", "a sense", "the atmosphere")
3. GPT-4o-mini hiểu ngữ cảnh video, phân biệt entity thật vs meta-description
4. Tất cả papers top-cited đều dùng LLM, không ai dùng NER truyền thống cho task này
5. Cost: GPT-4o-mini rất rẻ (~$0.15/1M input tokens), 585 captions × ~300 tokens = ~$0.03 total

**Suggested prompt pattern (from GraphRAG):**
```
Given the following video segment caption, extract all entities and relationships.

For each entity, provide:
- entity_name: descriptive name
- entity_type: person, animal, object, location
- entity_description: appearance, position, distinguishing features

For each relationship, provide:
- source_entity, target_entity, relationship_description

Caption: {caption_text}
```

---

## Sources

- [Microsoft GraphRAG Paper](https://arxiv.org/abs/2404.16130)
- [Microsoft GraphRAG GitHub](https://microsoft.github.io/graphrag/)
- [LightRAG GitHub (EMNLP 2025)](https://github.com/HKUDS/LightRAG)
- [KGGen Paper (NeurIPS 2025)](https://arxiv.org/abs/2502.09956)
- [KGGen GitHub](https://github.com/stair-lab/kg-gen)
- [Neo4j LLM Knowledge Graph Builder](https://neo4j.com/labs/genai-ecosystem/llm-graph-builder/)
- [GraphRAG Auto-tuning Blog](https://www.microsoft.com/en-us/research/blog/graphrag-auto-tuning-provides-rapid-adaptation-to-new-domains/)
- [From Videos to Indexed Knowledge Graphs](https://arxiv.org/abs/2510.01513)
- [VHAKG Paper](https://arxiv.org/abs/2408.14895)
- [KET-RAG Paper](https://arxiv.org/abs/2502.09304)
- [Neo4j Under the Covers with LightRAG](https://neo4j.com/blog/developer/under-the-covers-with-lightrag-extraction/)
