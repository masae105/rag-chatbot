import json
import unittest
from unittest.mock import patch

from app.generation.generate_answer import (
    ANSWER_RESPONSE_SCHEMA,
    FACTS_RESPONSE_SCHEMA,
    NO_SOURCE_ANSWER,
    _extract_verified_facts,
    _generate_answer_from_facts,
    _has_unsupported_numbers,
    _is_valid_citation,
    generate_answer,
)


CHUNKS = [
    {
        "document_id": "doc-1",
        "page_number": 3,
        "chunk_number": 1,
        "content": "従業員100人以下の企業では、DX取組割合が低い。",
    },
    {
        "document_id": "doc-2",
        "page_number": 5,
        "chunk_number": 2,
        "content": "従業員101人以上の企業では、DX取組割合が高い。",
    },
]


def response_json(claims):
    return json.dumps({"claims": claims}, ensure_ascii=False)


def claim(text, reference, quote):
    return {
        "text": text,
        "citations": [{"reference": reference, "quote": quote}],
    }


class DirectAnswerTests(unittest.TestCase):
    def test_two_stage_ollama_calls_answer_from_chunks_and_map_sources(self):
        stage1_response = json.dumps(
            {
                "facts": [
                    {
                        "topic": "100人以下のDX取組状況",
                        "reference": "資料1",
                        "quote": CHUNKS[0]["content"],
                        "fact": "従業員100人以下の企業ではDX取組割合が低い。",
                    },
                    {
                        "topic": "101人以上のDX取組状況",
                        "reference": "資料2",
                        "quote": CHUNKS[1]["content"],
                        "fact": "従業員101人以上の企業ではDX取組割合が高い。",
                    },
                ]
            },
            ensure_ascii=False,
        )
        stage2_response = response_json(
            [
                claim(
                    "従業員100人以下の企業では、DX取組割合が低いです。",
                    "資料1",
                    CHUNKS[0]["content"],
                ),
                claim(
                    "従業員101人以上の企業では、DX取組割合が高いです。",
                    "資料2",
                    CHUNKS[1]["content"],
                ),
            ]
        )

        with patch(
            "app.generation.generate_answer._call_ollama",
            side_effect=[stage1_response, stage2_response],
        ) as call_ollama:
            result = generate_answer("DX取組状況は？", CHUNKS)

        self.assertEqual(call_ollama.call_count, 2)
        stage1_prompt, stage1_schema = call_ollama.call_args_list[0].args
        stage2_prompt, stage2_schema = call_ollama.call_args_list[1].args
        self.assertIn("DX取組状況は？", stage1_prompt)
        self.assertIn("[資料1]\n" + CHUNKS[0]["content"], stage1_prompt)
        self.assertIn("[資料2]\n" + CHUNKS[1]["content"], stage1_prompt)
        self.assertEqual(stage1_schema, FACTS_RESPONSE_SCHEMA)
        self.assertIn("DX取組状況は？", stage2_prompt)
        self.assertEqual(stage2_schema, ANSWER_RESPONSE_SCHEMA)
        self.assertEqual(
            result["answer"],
            "従業員100人以下の企業では、DX取組割合が低いです[資料1]。\n"
            "従業員101人以上の企業では、DX取組割合が高いです[資料2]。",
        )
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 3,
                    "chunk_number": 1,
                },
                {
                    "reference": "資料2",
                    "document_id": "doc-2",
                    "page_number": 5,
                    "chunk_number": 2,
                },
            ],
        )

    def test_claim_with_missing_or_unknown_evidence_is_discarded(self):
        stage1_response = json.dumps(
            {
                "facts": [
                    {
                        "topic": "DX取組状況",
                        "reference": "資料1",
                        "quote": "この文章は本文にありません。",
                        "fact": "資料にない主張です。",
                    },
                    {
                        "topic": "別資料",
                        "reference": "資料9",
                        "quote": CHUNKS[0]["content"],
                        "fact": "別の資料の主張です。",
                    },
                ]
            },
            ensure_ascii=False,
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=stage1_response,
        ):
            result = generate_answer("質問", CHUNKS)

        self.assertNotIn("資料にない主張です。", result["answer"])
        self.assertNotIn("別の資料の主張です。", result["answer"])
        self.assertEqual(
            result["answer"],
            "確認できた関連資料の原文を表示します。\n\n"
            f"[資料1] 文書ID: doc-1 / ページ: 3 / チャンク: 1\n{CHUNKS[0]['content']}\n\n"
            f"[資料2] 文書ID: doc-2 / ページ: 5 / チャンク: 2\n{CHUNKS[1]['content']}",
        )
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 3,
                    "chunk_number": 1,
                },
                {
                    "reference": "資料2",
                    "document_id": "doc-2",
                    "page_number": 5,
                    "chunk_number": 2,
                },
            ],
        )

    def test_stage2_with_no_valid_claims_falls_back_to_verbatim_chunks(self):
        stage1_response = json.dumps(
            {
                "facts": [
                    {
                        "topic": "100人以下のDX取組状況",
                        "reference": "資料1",
                        "quote": CHUNKS[0]["content"],
                        "fact": "従業員100人以下の企業ではDX取組割合が低い。",
                    }
                ]
            },
            ensure_ascii=False,
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            side_effect=[stage1_response, '{"claims": []}'],
        ):
            result = generate_answer("質問", CHUNKS)

        self.assertEqual(
            result["answer"],
            "確認できた関連資料の原文を表示します。\n\n"
            f"[資料1] 文書ID: doc-1 / ページ: 3 / チャンク: 1\n{CHUNKS[0]['content']}\n\n"
            f"[資料2] 文書ID: doc-2 / ページ: 5 / チャンク: 2\n{CHUNKS[1]['content']}",
        )
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 3,
                    "chunk_number": 1,
                },
                {
                    "reference": "資料2",
                    "document_id": "doc-2",
                    "page_number": 5,
                    "chunk_number": 2,
                },
            ],
        )

    def test_empty_claims_returns_no_source_answer(self):
        result = generate_answer("質問", [])

        self.assertEqual(
            result,
            {"answer": NO_SOURCE_ANSWER, "sources": []},
        )

    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_answer("", CHUNKS)
        with self.assertRaises(ValueError):
            generate_answer("質問", {})


class Stage1FactExtractionTests(unittest.TestCase):
    def setUp(self):
        self.source_by_reference = {
            "資料1": {
                "reference": "資料1",
                "document_id": "doc-1",
                "page_number": 3,
                "content": "従業員100人以下の企業では、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
            },
            "資料2": {
                "reference": "資料2",
                "document_id": "doc-2",
                "page_number": 5,
                "content": "推進体制とDXの成果の関係は顕著ではない。",
            },
        }
        self.context_parts = [
            f"[資料1]\n{self.source_by_reference['資料1']['content']}",
            f"[資料2]\n{self.source_by_reference['資料2']['content']}",
        ]

    def test_is_valid_citation_checks_reference_and_quote(self):
        self.assertTrue(
            _is_valid_citation(
                "資料1",
                "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後",
                self.source_by_reference,
            )
        )
        self.assertFalse(
            _is_valid_citation(
                "資料9",
                "DX成果なしの企業で3割5分",
                self.source_by_reference,
            )
        )
        self.assertFalse(
            _is_valid_citation(
                "資料1",
                "原文に存在しない文章です。",
                self.source_by_reference,
            )
        )

    def test_has_unsupported_numbers_detects_unmatched_numbers(self):
        quote = "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後"
        self.assertFalse(
            _has_unsupported_numbers(
                "DX成果なしは3割5分、DX成果ありは6割前後である。",
                quote,
            )
        )
        self.assertTrue(
            _has_unsupported_numbers(
                "DX成果なしは3割5分、DX成果ありは8割である。",
                quote,
            )
        )
        self.assertTrue(
            _has_unsupported_numbers(
                "DX成果ありは60%である。",
                quote,
            )
        )

    def test_extract_verified_facts_keeps_valid_and_discards_unsupported_fact_numbers(self):
        raw_stage1 = json.dumps(
            {
                "facts": [
                    {
                        "topic": "組織連携",
                        "reference": "資料1",
                        "quote": "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
                        "fact": "DX成果なしは3割5分、DX成果ありは6割前後である。",
                    },
                    {
                        "topic": "推進体制",
                        "reference": "資料2",
                        "quote": "推進体制とDXの成果の関係は顕著ではない。",
                        "fact": "推進体制は6割前後であり関係は顕著ではない。",
                    },
                    {
                        "topic": "無効な引用",
                        "reference": "資料1",
                        "quote": "原文に存在しない引用文",
                        "fact": "無効",
                    },
                    {
                        "topic": "存在しない資料",
                        "reference": "資料9",
                        "quote": "推進体制とDXの成果の関係は顕著ではない。",
                        "fact": "無効",
                    },
                ]
            },
            ensure_ascii=False,
        )

        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=raw_stage1,
        ) as call_ollama:
            verified = _extract_verified_facts(
                "組織連携と推進体制の違いは？",
                self.context_parts,
                self.source_by_reference,
            )

        call_ollama.assert_called_once()
        _, schema = call_ollama.call_args.args
        self.assertEqual(schema, FACTS_RESPONSE_SCHEMA)
        self.assertEqual(
            verified,
            [
                {
                    "topic": "組織連携",
                    "reference": "資料1",
                    "quote": "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
                    "fact": "DX成果なしは3割5分、DX成果ありは6割前後である。",
                },
                {
                    "topic": "推進体制",
                    "reference": "資料2",
                    "quote": "推進体制とDXの成果の関係は顕著ではない。",
                    "fact": None,
                },
            ],
        )


class Stage2AnswerGenerationTests(unittest.TestCase):
    def setUp(self):
        self.source_by_reference = {
            "資料1": {
                "reference": "資料1",
                "document_id": "doc-1",
                "page_number": 3,
                "chunk_number": 1,
                "content": "従業員100人以下の企業では、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
            },
            "資料2": {
                "reference": "資料2",
                "document_id": "doc-2",
                "page_number": 5,
                "chunk_number": 2,
                "content": "推進体制とDXの成果の関係は顕著ではない。また、別の一文も記載されている。",
            },
        }
        self.verified_facts = [
            {
                "topic": "組織連携",
                "reference": "資料1",
                "quote": "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
                "fact": "DX成果なしは3割5分、DX成果ありは6割前後である。",
            },
            {
                "topic": "推進体制",
                "reference": "資料2",
                "quote": "推進体制とDXの成果の関係は顕著ではない。",
                "fact": "推進体制とDXの成果の関係は顕著ではない。",
            },
        ]

    def test_stage2_prompt_and_valid_answer_formatting(self):
        raw_stage2 = response_json(
            [
                claim(
                    "組織連携の実施割合は、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られます。",
                    "資料1",
                    "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
                ),
                claim(
                    "推進体制とDXの成果の関係は顕著ではありません。",
                    "資料2",
                    "推進体制とDXの成果の関係は顕著ではない。",
                ),
            ]
        )

        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=raw_stage2,
        ) as call_ollama:
            result = _generate_answer_from_facts(
                "従業員100人以下の企業では、組織連携や推進体制にどのような違いがありますか？",
                self.verified_facts,
                self.source_by_reference,
            )

        call_ollama.assert_called_once()
        prompt, schema = call_ollama.call_args.args
        self.assertEqual(schema, ANSWER_RESPONSE_SCHEMA)
        self.assertIn(
            "[根拠1]\n"
            "観点(topic): 組織連携\n"
            "資料番号(reference): 資料1\n"
            "原文引用(quote): DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。\n"
            "事実の要約(fact): DX成果なしは3割5分、DX成果ありは6割前後である。",
            prompt,
        )
        self.assertIn(
            "[根拠2]\n"
            "観点(topic): 推進体制\n"
            "資料番号(reference): 資料2\n"
            "原文引用(quote): 推進体制とDXの成果の関係は顕著ではない。\n"
            "事実の要約(fact): 推進体制とDXの成果の関係は顕著ではない。",
            prompt,
        )
        self.assertEqual(
            result["answer"],
            "組織連携の実施割合は、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られます[資料1]。\n"
            "推進体制とDXの成果の関係は顕著ではありません[資料2]。",
        )
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 3,
                    "chunk_number": 1,
                },
                {
                    "reference": "資料2",
                    "document_id": "doc-2",
                    "page_number": 5,
                    "chunk_number": 2,
                },
            ],
        )

    def test_stage2_does_not_generate_claim_from_null_fact(self):
        facts_with_null = [
            self.verified_facts[0],
            {
                "topic": "推進体制",
                "reference": "資料2",
                "quote": "推進体制とDXの成果の関係は顕著ではない。",
                "fact": None,
            },
        ]
        raw_stage2 = response_json(
            [
                claim(
                    "組織連携の実施割合は、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後です。",
                    "資料1",
                    "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
                ),
                claim(
                    "推進体制とDXの成果の関係は顕著ではありません。",
                    "資料2",
                    "推進体制とDXの成果の関係は顕著ではない。",
                ),
            ]
        )

        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=raw_stage2,
        ) as call_ollama:
            result = _generate_answer_from_facts(
                "質問",
                facts_with_null,
                self.source_by_reference,
            )

        call_ollama.assert_called_once()
        prompt, _ = call_ollama.call_args.args
        self.assertNotIn("推進体制とDXの成果の関係は顕著ではない。", prompt)
        self.assertEqual(
            result["answer"],
            "組織連携の実施割合は、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後です[資料1]。",
        )
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 3,
                    "chunk_number": 1,
                }
            ],
        )

        with patch(
            "app.generation.generate_answer._call_ollama"
        ) as call_ollama_all_null:
            fallback_result = _generate_answer_from_facts(
                "質問",
                [facts_with_null[1]],
                self.source_by_reference,
            )
        call_ollama_all_null.assert_not_called()
        self.assertIn("確認できた関連資料の原文を表示します。", fallback_result["answer"])
        self.assertIn(self.source_by_reference["資料1"]["content"], fallback_result["answer"])
        self.assertIn(self.source_by_reference["資料2"]["content"], fallback_result["answer"])

    def test_stage2_discards_citation_not_present_in_stage1(self):
        raw_stage2 = response_json(
            [
                claim(
                    "資料2の別の一文に基づく主張です。",
                    "資料2",
                    "また、別の一文も記載されている。",
                )
            ]
        )

        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=raw_stage2,
        ):
            result = _generate_answer_from_facts(
                "質問",
                self.verified_facts,
                self.source_by_reference,
            )

        self.assertNotIn("資料2の別の一文に基づく主張です。", result["answer"])
        self.assertIn("確認できた関連資料の原文を表示します。", result["answer"])
        self.assertIn(self.source_by_reference["資料1"]["content"], result["answer"])
        self.assertIn(self.source_by_reference["資料2"]["content"], result["answer"])

    def test_organization_collaboration_numbers_are_not_reused_for_promotion_structure(self):
        raw_stage2 = response_json(
            [
                claim(
                    "組織連携の実施割合は、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後です。",
                    "資料1",
                    "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。",
                ),
                claim(
                    "推進体制はDX成果なしで3割5分、DX成果ありで6割前後であり、関係は顕著ではありません。",
                    "資料2",
                    "推進体制とDXの成果の関係は顕著ではない。",
                ),
                claim(
                    "従業員100人以下の企業では、推進体制とDXの成果の関係は顕著ではありません。",
                    "資料2",
                    "推進体制とDXの成果の関係は顕著ではない。",
                ),
            ]
        )

        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=raw_stage2,
        ):
            result = _generate_answer_from_facts(
                "従業員100人以下の企業では、組織連携や推進体制にどのような違いがありますか？",
                self.verified_facts,
                self.source_by_reference,
            )

        self.assertEqual(
            result["answer"],
            "組織連携の実施割合は、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後です[資料1]。",
        )
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料1",
                    "document_id": "doc-1",
                    "page_number": 3,
                    "chunk_number": 1,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
