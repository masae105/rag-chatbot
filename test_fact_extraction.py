import io
import json
import unittest
from unittest.mock import patch

import app.generation.generate_answer as generation


CHUNK = {
    "document_id": "doc-1",
    "page_number": 1,
    "chunk_number": 1,
    "content": "テスト用の資料本文です。",
}


class FactExtractionTests(unittest.TestCase):
    def test_request_includes_json_schema(self):
        requests = []

        def fake_urlopen(request, timeout):
            requests.append((request, timeout))
            return io.BytesIO(
                json.dumps({"response": '{"facts": ["事実A"]}'}).encode("utf-8")
            )

        with patch.object(generation, "urlopen", side_effect=fake_urlopen):
            result = generation._extract_facts_from_chunk(CHUNK, "資料1")

        self.assertEqual(result["facts"], ["事実A"])
        self.assertEqual(len(requests), 1)
        request, timeout = requests[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(
            payload["format"],
            {
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["facts"],
            },
        )
        self.assertEqual(timeout, 120)

    def test_valid_json_fact_arrays(self):
        cases = (
            ('{"facts": ["事実A"]}', ["事実A"]),
            ('{"facts": ["事実A", "事実B"]}', ["事実A", "事実B"]),
            ('{"facts": []}', []),
        )
        for response, expected in cases:
            with self.subTest(response=response):
                with patch.object(generation, "_call_ollama", return_value=response):
                    result = generation._extract_facts_from_chunk(CHUNK, "資料1")
                self.assertEqual(result["facts"], expected)

    def test_malformed_extraction_responses_are_rejected(self):
        responses = (
            '{"facts": [""]}',
            '{"facts": ["[]"]}',
            """{"facts": ["['事実A']"]}""",
            """{"facts": ["{'事実': '従業員規模が小さいほど、DXに取組んでいる企業の割合が低くなる'}"]}""",
            '{"facts": [["事実A"]]}',
            '{"facts": [{"fact": "事実A"}]}',
            '["事実A"]',
            '{"事実": "事実A"}',
            "これはJSONではありません",
        )
        for response in responses:
            with self.subTest(response=response):
                with patch.object(generation, "_call_ollama", return_value=response):
                    with self.assertRaises(generation._FactExtractionError):
                        generation._extract_facts_from_chunk(CHUNK, "資料1")

    def test_extraction_failure_returns_fallback_without_second_stage(self):
        with (
            patch.object(
                generation,
                "_call_ollama",
                return_value='{"facts": ["[]"]}',
            ),
            patch.object(generation, "_generate_answer_from_facts") as generate,
        ):
            result = generation.generate_answer("質問", [CHUNK])

        self.assertEqual(
            result,
            {"answer": generation.NO_SOURCE_ANSWER, "sources": []},
        )
        generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
