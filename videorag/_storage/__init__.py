"""Lazy storage exports to avoid loading ImageBind for graph-only operations."""

_EXPORTS = {
    "NetworkXStorage": (".gdb_networkx", "NetworkXStorage"),
    "UnifiedNetworkXStorage": (
        ".gdb_unified_networkx",
        "UnifiedNetworkXStorage",
    ),
    "Neo4jStorage": (".gdb_neo4j", "Neo4jStorage"),
    "HNSWVectorStorage": (".vdb_hnswlib", "HNSWVectorStorage"),
    "NanoVectorDBStorage": (".vdb_nanovectordb", "NanoVectorDBStorage"),
    "NanoVectorDBVideoSegmentStorage": (
        ".vdb_nanovectordb",
        "NanoVectorDBVideoSegmentStorage",
    ),
    "JsonKVStorage": (".kv_json", "JsonKVStorage"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, attribute = _EXPORTS[name]
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value

