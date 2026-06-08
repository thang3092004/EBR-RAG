import asyncio
from types import SimpleNamespace

import numpy as np

from videorag._storage.vdb_nanovectordb import NanoVectorDBStorage


class _Embedding:
    embedding_dim = 2

    async def __call__(self, texts):
        return np.ones((len(texts), 2), dtype=np.float32)


class _Client:
    def upsert(self, datas):
        return datas

    def query(self, **kwargs):
        return [
            {
                "__id__": "item",
                "__metrics__": 0.9,
            }
        ]


def test_text_vector_storage_does_not_reference_imagebind_embedder():
    storage = object.__new__(NanoVectorDBStorage)
    storage.namespace = "test"
    storage._max_batch_size = 4
    storage.embedding_func = _Embedding()
    storage.meta_fields = set()
    storage._client = _Client()
    storage.cosine_better_than_threshold = 0.2

    async def run():
        inserted = await storage.upsert({"item": {"content": "hello"}})
        queried = await storage.query("hello", top_k=1)
        assert inserted
        assert queried[0]["id"] == "item"
        assert queried[0]["similarity"] == 0.9

    asyncio.run(run())
