import shutil
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.rag_service import KnowledgeBase


class TestIndex(unittest.TestCase):
    def test_single_document_update_and_failed_rebuild(self):
        class Encoder:
            def passage_embed(self, passages):
                return [np.array([1, 1], dtype=np.float32) for _ in passages]

            def query_embed(self, _question):
                return iter([[1, 1]])

        with TemporaryDirectory() as folder, patch.object(KnowledgeBase, "encoder", Encoder()):
            root = Path(folder)
            docs = root / "docs"
            docs.mkdir()
            (docs / "client.md").write_text("# 客户端\n\n令牌过期请重新登录。", encoding="utf-8")
            (docs / "server.md").write_text("# 服务端\n\n令牌最长有效期为 60 分钟。", encoding="utf-8")
            data = root / "data"
            with TestClient(create_app(data, docs)) as client:
                initial = client.get("/api/status").json()
                self.assertEqual(initial["encoded_passages"], 2)
                (docs / "server.md").write_text("# 服务端\n\n令牌最长有效期为 90 分钟。", encoding="utf-8")
                status = client.post("/api/knowledge/reindex").json()
                self.assertEqual((status["encoded_passages"], status["reused_embeddings"]), (1, 1))
                self.assertNotEqual(status["revision"], initial["revision"])
                for mode in ("bm25", "vector", "hybrid"):
                    result = client.post("/api/knowledge/ask", json={
                        "question": "令牌最长有效期", "document": "server.md", "mode": mode,
                    }).json()
                    self.assertIn("90 分钟", result["answer"])
                    self.assertEqual({source["document"] for source in result["sources"]}, {"server.md"})
                    self.assertEqual(result["index_revision"], status["revision"])
                shutil.rmtree(docs)
                self.assertEqual(client.post("/api/knowledge/reindex").status_code, 400)
                self.assertEqual(client.get("/api/status").json()["revision"], status["revision"])

    def test_restarts_from_vector_cache(self):
        class Encoder:
            calls = 0

            def passage_embed(self, passages):
                self.calls += len(passages)
                return [[1, 1] for _ in passages]

            def query_embed(self, _question):
                return iter([[1, 1]])

        encoder = Encoder()
        with TemporaryDirectory() as folder, patch.object(KnowledgeBase, "encoder", encoder):
            root = Path(folder)
            docs = root / "docs"
            docs.mkdir()
            (docs / "guide.txt").write_text("恢复账号需要验证邮箱。", encoding="utf-8")
            for _ in range(2):
                with TestClient(create_app(root / "data", docs)) as client:
                    result = client.post("/api/knowledge/ask", json={"question": "恢复账号", "mode": "vector"})
                    self.assertEqual(result.status_code, 200)
                    self.assertTrue(result.json()["citations"])
            self.assertEqual(encoder.calls, 1)
