import asyncio
import os
import sys
import json
from dataclasses import asdict
from pathlib import Path

# Add project root to sys.path
root_path = str(Path(__file__).resolve().parent.parent)
if root_path not in sys.path:
    sys.path.append(root_path)

from videorag.videorag import VideoRAG, QueryParam
from videorag._llm import openai_4o_mini_config

# Use the same LLM config as the project
llm_config = openai_4o_mini_config

async def reindex_entities(col_id, sub_category):
    work_dir = Path(root_path) / "longervideos/videorag-workdir" / sub_category
    if not work_dir.exists():
        print(f"Skipping {sub_category}, workdir not found.")
        return

    print(f"\n--- [1/3] Deleting corrupted data for {sub_category} ---")
    
    # Delete the corrupted entities VDB file BEFORE initializing VideoRAG
    entities_vdb_path = work_dir / "vdb_entities.json"
    if entities_vdb_path.exists():
        entities_vdb_path.unlink()
        print(f"Deleted corrupted {entities_vdb_path}")

    # Initialize VideoRAG
    vrag = VideoRAG(llm=llm_config, working_dir=str(work_dir))

    # Load chunks
    chunks_path = work_dir / "kv_store_text_chunks.json"
    if not chunks_path.exists():
        print(f"Error: Chunks not found for {sub_category}")
        return

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    
    # Initialize entities storage
    from videorag._storage.vdb_nanovectordb import NanoVectorDBStorage
    entities_storage = NanoVectorDBStorage(
        namespace="entities",
        global_config=asdict(vrag),
        embedding_func=vrag.embedding_func,
        meta_fields=["entity_type", "description", "source_id"]
    )
    
    from videorag._op import extract_entities
    
    print(f"--- [2/3] Re-extracting entities from {len(chunks_data)} chunks ---")
    
    # Mirroring VideoRAG.ainsert logic 100%
    maybe_new_kg, _, _ = await extract_entities(
        chunks_data,
        knowledge_graph_inst=vrag.chunk_entity_relation_graph,
        entity_vdb=entities_storage,
        global_config=asdict(vrag)
    )
    if maybe_new_kg is None:
        raise RuntimeError("Entity extraction returned None.")
    
    # Update the graph instance
    vrag.chunk_entity_relation_graph = maybe_new_kg
    
    # Save storages
    await vrag.chunk_entity_relation_graph.index_done_callback()
    await entities_storage.index_done_callback()
    

    
    print(f"✅ SUCCESSFULLY restored {sub_category}")

async def main():
    # Collections to fix
    collections = [
        ("6", "6-daubechies-wavelet-lecture"),
        ("11", "11-primetime-emmy-awards"),
        ("19", "19-jeff-bezos")
    ]
    
    for cid, subcat in collections:
        try:
            await reindex_entities(cid, subcat)
        except Exception as e:
            import traceback
            print(f"❌ Failed to fix {subcat}: {str(e)}")
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
