import asyncio

from videorag._storage.gdb_unified_networkx import UnifiedNetworkXStorage
from videorag._unified_graph.retrieval import retrieve_graph_packets


class FakeEntityVDB:
    async def query(self, query, top_k):
        return [
            {"entity_name": "PERSON_001", "similarity": 0.95},
            {"entity_name": "OBJECT_001", "similarity": 0.80},
        ][:top_k]


def test_multidigraph_preserves_relation_occurrences(tmp_path):
    async def run():
        storage = UnifiedNetworkXStorage(
            namespace="test",
            global_config={"working_dir": str(tmp_path)},
        )
        for node in ("PERSON_001", "OBJECT_001", "LOCATION_001"):
            await storage.upsert_node(
                node,
                {
                    "entity_type": node.split("_")[0].lower(),
                    "description": node,
                },
            )
        await storage.upsert_edge(
            "PERSON_001",
            "OBJECT_001",
            {
                "edge_id": "EDGE_1",
                "predicate": "holds",
                "confidence": 0.9,
                "start": 1.0,
                "end": 2.0,
                "segment_id": "SEG_1",
                "storage_id": "video_0",
                "provenance": [{"source": "visual"}],
            },
        )
        await storage.upsert_edge(
            "PERSON_001",
            "OBJECT_001",
            {
                "edge_id": "EDGE_2",
                "predicate": "drops",
                "confidence": 0.8,
                "start": 4.0,
                "end": 5.0,
                "segment_id": "SEG_2",
                "storage_id": "video_1",
                "provenance": [{"source": "visual"}],
            },
        )
        await storage.upsert_edge(
            "OBJECT_001",
            "LOCATION_001",
            {
                "edge_id": "EDGE_3",
                "predicate": "located_in",
                "confidence": 0.9,
                "start": 5.0,
                "end": 6.0,
                "segment_id": "SEG_2",
                "storage_id": "video_1",
                "provenance": [{"source": "visual"}],
            },
        )
        assert storage._graph.number_of_edges("PERSON_001", "OBJECT_001") == 2
        packets = await retrieve_graph_packets(
            "What object does the person hold and where is it?",
            FakeEntityVDB(),
            storage,
            top_k=4,
        )
        assert len(packets) <= 4
        assert any("LOCATION_001" in packet["nodes"] for packet in packets)
        assert all(len(packet["nodes"]) <= 4 for packet in packets)

    asyncio.run(run())

