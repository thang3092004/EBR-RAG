__all__ = ["VideoRAG", "QueryParam"]


def __getattr__(name):
    if name in {"VideoRAG", "QueryParam"}:
        from .videorag import QueryParam, VideoRAG

        return {"VideoRAG": VideoRAG, "QueryParam": QueryParam}[name]
    raise AttributeError(name)
