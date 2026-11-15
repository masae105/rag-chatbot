import json
import unittest
from unittest.mock import patch

from app.generation.generate_answer import generate_answer


class SourceCitationTests(unittest.TestCase):
    def test_only_sources_with_valid_quotes_are_returned(self):
        chunks = [
            {
                "document_id": "doc-1",
                "page_number": 2,
                "chunk_number": 1,
                "content": "DX取組割合は従業員規模が小さいほど低い。",
            },
            {
                "document_id": "doc-2",
                "page_number": 4,
                "chunk_number": 2,
                "content": "データ利活用の実施割合は上昇した。",
            },
        ]
        stage1_response = json.dumps(
            {
                "facts": [
                    {
                        "topic": "DX取組割合",
                        "reference": "資料1",
                        "quote": chunks[0]["content"],
                        "fact": "DX取組割合は従業員規模が小さいほど低い。",
                    },
                    {
                        "topic": "データ利活用",
                        "reference": "資料2",
                        "quote": chunks[1]["content"],
                        "fact": "データ利活用の実施割合は上昇した。",
                    },
                ]
            },
            ensure_ascii=False,
        )
        response = json.dumps(
            {
                "claims": [
                    {
                        "text": "DX取組割合は従業員規模が小さいほど低いです。",
                        "citations": [
                            {
                                "reference": "資料1",
                                "quote": chunks[0]["content"],
                            },
                            {
                                "reference": "資料2",
                                "quote": "本文にない引用です。",
                            },
                        ],
                    },
                    {
                        "text": "無関係な主張です。",
                        "citations": [
                            {
                                "reference": "資料2",
                                "quote": chunks[1]["content"],
                            }
                        ],
                    },
                ]
            },
            ensure_ascii=False,
        )

        with patch(
            "app.generation.generate_answer._call_ollama",
            side_effect=[stage1_response, response],
        ):
            result = generate_answer("質問", chunks)

        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 2,
                    "chunk_number": 1,
                },
                {
                    "reference": "資料2",
                    "document_id": "doc-2",
                    "page_number": 4,
                    "chunk_number": 2,
                },
            ],
        )
        self.assertNotIn("資料9", result["answer"])


if __name__ == "__main__":
    unittest.main()
