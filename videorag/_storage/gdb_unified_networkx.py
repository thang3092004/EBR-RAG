from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Union

import networkx as nx
import numpy as np

from .._utils import logger
from ..base import BaseGraphStorage, SingleCommunitySchema


JSON_FIELDS = {
    "aliases",
    "sources",
    "attributes",
    "provenance",
    "modalities",
}


def _encode_attributes(data: dict[str, Any]) -> dict[str, Any]:
    encoded = {}
    for key, value in data.items():
        if value is None:
            encoded[key] = ""
        elif isinstance(value, (dict, list, tuple, set)):
            encoded[key] = json.dumps(value, ensure_ascii=False)
        elif isinstance(value, (str, int, float, bool)):
            encoded[key] = value
        else:
            encoded[key] = json.dumps(value, ensure_ascii=False, default=str)
    return encoded


def _decode_attributes(data: dict[str, Any]) -> dict[str, Any]:
    decoded = dict(data)
    for key in JSON_FIELDS:
        value = decoded.get(key)
        if isinstance(value, str) and value:
            try:
                decoded[key] = json.loads(value)
            except json.JSONDecodeError:
                pass
    return decoded


@dataclass
class UnifiedNetworkXStorage(BaseGraphStorage):
    """Directed multigraph storage for entity and temporal relation occurrences."""

    def __post_init__(self):
        self._graphml_xml_file = os.path.join(
            self.global_config["working_dir"],
            f"graph_{self.namespace}.graphml",
        )
        if os.path.exists(self._graphml_xml_file):
            loaded = nx.read_graphml(
                self._graphml_xml_file,
                node_type=str,
                force_multigraph=True,
            )
            if not loaded.is_directed():
                loaded = loaded.to_directed()
            self._graph = nx.MultiDiGraph(loaded)
            logger.info(
                "Loaded unified graph %s with %d nodes and %d edge occurrences",
                self._graphml_xml_file,
                self._graph.number_of_nodes(),
                self._graph.number_of_edges(),
            )
        else:
            self._graph = nx.MultiDiGraph()

    async def index_done_callback(self):
        os.makedirs(os.path.dirname(self._graphml_xml_file), exist_ok=True)
        nx.write_graphml(self._graph, self._graphml_xml_file)
        logger.info(
            "Wrote unified graph with %d nodes and %d edge occurrences",
            self._graph.number_of_nodes(),
            self._graph.number_of_edges(),
        )

    async def has_node(self, node_id: str) -> bool:
        return self._graph.has_node(node_id)

    async def has_edge(self, source_node_id: str, target_node_id: str) -> bool:
        return self._graph.has_edge(source_node_id, target_node_id)

    async def node_degree(self, node_id: str) -> int:
        return int(self._graph.degree(node_id)) if self._graph.has_node(node_id) else 0

    async def edge_degree(self, src_id: str, tgt_id: str) -> int:
        return await self.node_degree(src_id) + await self.node_degree(tgt_id)

    async def get_node(self, node_id: str) -> Union[dict, None]:
        data = self._graph.nodes.get(node_id)
        return _decode_attributes(data) if data is not None else None

    async def get_edge(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> Union[dict, None]:
        occurrences = self._graph.get_edge_data(source_node_id, target_node_id)
        if not occurrences:
            return None
        decoded = [_decode_attributes(data) for data in occurrences.values()]
        best = max(decoded, key=lambda item: float(item.get("confidence", 0.0)))
        return {**best, "occurrences": decoded}

    async def get_edge_occurrences(
        self,
        source_node_id: str,
        target_node_id: str,
    ) -> list[dict[str, Any]]:
        occurrences = self._graph.get_edge_data(source_node_id, target_node_id) or {}
        return [
            {"edge_id": str(key), **_decode_attributes(data)}
            for key, data in occurrences.items()
        ]

    async def get_node_edges(self, source_node_id: str):
        if not self._graph.has_node(source_node_id):
            return None
        return [
            (source, target)
            for source, target, _ in self._graph.out_edges(
                source_node_id,
                keys=True,
            )
        ]

    async def get_node_edge_occurrences(
        self,
        node_id: str,
        direction: str = "both",
    ) -> list[dict[str, Any]]:
        if not self._graph.has_node(node_id):
            return []
        records = []
        if direction in {"out", "both"}:
            for source, target, key, data in self._graph.out_edges(
                node_id,
                keys=True,
                data=True,
            ):
                records.append(
                    {
                        "edge_id": str(key),
                        **_decode_attributes(data),
                        "source_id": source,
                        "target_id": target,
                    }
                )
        if direction in {"in", "both"}:
            for source, target, key, data in self._graph.in_edges(
                node_id,
                keys=True,
                data=True,
            ):
                records.append(
                    {
                        "edge_id": str(key),
                        **_decode_attributes(data),
                        "source_id": source,
                        "target_id": target,
                    }
                )
        return records

    async def upsert_node(self, node_id: str, node_data: dict[str, Any]):
        existing = dict(self._graph.nodes.get(node_id, {}))
        existing.update(_encode_attributes(node_data))
        self._graph.add_node(node_id, **existing)

    async def upsert_edge(
        self,
        source_node_id: str,
        target_node_id: str,
        edge_data: dict[str, Any],
    ):
        edge_id = str(
            edge_data.get("edge_id")
            or f"EDGE_{self._graph.number_of_edges() + 1:07d}"
        )
        payload = dict(edge_data)
        payload.pop("edge_id", None)
        self._graph.add_edge(
            source_node_id,
            target_node_id,
            key=edge_id,
            **_encode_attributes(payload),
        )

    async def clustering(self, algorithm: str):
        raise NotImplementedError(
            "Unified graph retrieval uses local paths and PageRank, not clustering."
        )

    async def community_schema(self) -> dict[str, SingleCommunitySchema]:
        return {}

    async def embed_nodes(self, algorithm: str) -> tuple[np.ndarray, list[str]]:
        raise NotImplementedError("Unified graph nodes are retrieved through entity VDB.")

    async def clear(self) -> None:
        self._graph.clear()

    async def remove_video(self, video_id: str) -> dict[str, int]:
        edge_keys = []
        for source, target, key, data in self._graph.edges(keys=True, data=True):
            storage_id = str(data.get("storage_id", ""))
            provenance = str(data.get("provenance", ""))
            if storage_id.startswith(f"{video_id}_") or f'"video_id": "{video_id}"' in provenance:
                edge_keys.append((source, target, key))
        self._graph.remove_edges_from(edge_keys)

        removable_nodes = []
        for node_id, data in self._graph.nodes(data=True):
            if self._graph.degree(node_id) != 0:
                continue
            provenance = str(data.get("provenance", ""))
            aliases = str(data.get("aliases", ""))
            if (
                f'"video_id": "{video_id}"' in provenance
                or f"{video_id}::" in aliases
            ):
                removable_nodes.append(node_id)
        self._graph.remove_nodes_from(removable_nodes)
        return {
            "removed_edges": len(edge_keys),
            "removed_nodes": len(removable_nodes),
        }

    def statistics(self) -> dict[str, int]:
        return {
            "nodes": self._graph.number_of_nodes(),
            "edges": self._graph.number_of_edges(),
        }
