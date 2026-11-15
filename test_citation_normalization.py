import json
import unittest
from unittest.mock import patch

from app.generation.generate_answer import generate_answer


class CitationNormalizationTests(unittest.TestCase):
    def test_nfkc_and_whitespace_differences_are_accepted(self):
        chunk = {
            "document_id": "doc-1",
            "page_number": 1,
            "content": "ＡＩ　活用\nが、成果を上げた。",
        }
        stage1_response = json.dumps(
            {
                "facts": [
                    {
                        "topic": "AI活用",
                        "reference": "資料1",
                        "quote": "AI 活用が、成果を上げた。",
                        "fact": "AI活用で成果が出ました。",
                    }
                ]
            },
            ensure_ascii=False,
        )
        response = json.dumps(
            {
                "claims": [
                    {
                        "text": "AI活用で成果が出ました。",
                        "citations": [
                            {
                                "reference": "資料1",
                                "quote": "AI 活用が、成果を上げた。",
                            }
                        ],
                    }
                ]
            },
            ensure_ascii=False,
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            side_effect=[stage1_response, response],
        ):
            result = generate_answer("質問", [chunk])

        self.assertIn("AI活用で成果が出ました[資料1]。", result["answer"])
        self.assertEqual(result["sources"][0]["document_id"], "doc-1")

    def test_semantically_similar_but_nonmatching_quote_is_rejected(self):
        chunk = {
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_number": 1,
            "content": "DXによって業務効率が向上した。",
        }
        stage1_response = json.dumps(
            {
                "facts": [
                    {
                        "topic": "業務効率",
                        "reference": "資料1",
                        "quote": "仕事の効率が良くなった。",
                        "fact": "業務の生産性が改善しました。",
                    }
                ]
            },
            ensure_ascii=False,
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=stage1_response,
        ):
            result = generate_answer("質問", [chunk])

        self.assertNotIn("業務の生産性が改善しました。", result["answer"])
        self.assertIn(chunk["content"], result["answer"])
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 1,
                    "chunk_number": 1,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
