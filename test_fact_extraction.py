import json
import unittest
from unittest.mock import patch

from app.generation.generate_answer import (
    ANSWER_RESPONSE_SCHEMA,
    FACTS_RESPONSE_SCHEMA,
    NO_SOURCE_ANSWER,
    _build_skeleton_claim,
    _clean_quote_for_baseline,
    _extract_fact_markers,
    _extract_verified_facts,
    _generate_answer_from_facts,
    _group_complementary_facts,
    _has_unsupported_numbers,
    _is_heading_only_quote,
    _is_valid_citation,
    _is_verified_llm_claim,
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


class Step1HelperFunctionsTests(unittest.TestCase):
    def test_clean_quote_for_baseline_normalizes_whitespace_and_removes_heading_lines_without_rewriting(self):
        raw_quote = (
            "第2章 中小企業における DX の取組および成果の状況\n\n"
            "従業員 100 人以下の企業について、「DX 取組なし」「DX 成果なし」「DX 成果あり」の 3 区分に分けて比較した結果を図表- 3 に示す。\n"
            "図表- 3 各取組状況における比較\n"
            "（従業員 100 人以下）"
        )
        cleaned = _clean_quote_for_baseline(raw_quote)
        self.assertEqual(
            cleaned,
            "従業員100人以下の企業について、「DX取組なし」「DX成果なし」「DX成果あり」の3区分に分けて比較した結果を図表-3に示す。",
        )

    def test_is_heading_only_quote_detects_headings_and_keeps_body_sentences(self):
        self.assertTrue(
            _is_heading_only_quote("図表2-5 DXの成果が出ている企業の割合（従業員規模別）")
        )
        self.assertTrue(
            _is_heading_only_quote("第2章 中小企業における DX の取組および成果の状況")
        )
        self.assertFalse(
            _is_heading_only_quote("推進体制とDXの成果の関係は顕著ではない。")
        )

    def test_extract_fact_markers_extracts_unambiguous_sets_and_semantic_markers(self):
        saas_quote = (
            "SaaS を活用している企業の達成度は活用していない企業の約 2 倍、"
            "パブリッククラウドでは約 3 倍であり、両技術が迅速かつ拡張性の高い IT システム導入に寄与したと推察される。"
        )
        markers = _extract_fact_markers(saas_quote)
        self.assertFalse(markers["is_ambiguous"])
        self.assertEqual(
            markers["condition_number_sets"],
            [
                {
                    "subject": "SaaSを活用している企業",
                    "comparison": "活用していない企業",
                    "value": "2倍",
                },
                {
                    "subject": "パブリッククラウド",
                    "comparison": "活用していない企業",
                    "value": "3倍",
                },
            ],
        )
        self.assertIn("推察される", markers["modality"])
        self.assertIn("寄与した", markers["causality"])

    def test_extract_fact_markers_marks_ambiguous_mapping_as_true(self):
        ambiguous_quote = "A社とB社ではそれぞれ2割、5割となっている。"
        markers = _extract_fact_markers(ambiguous_quote)
        self.assertTrue(markers["is_ambiguous"])

        pronoun_quote = "これについては3割となっている。"
        pronoun_markers = _extract_fact_markers(pronoun_quote)
        self.assertTrue(pronoun_markers["is_ambiguous"])


class Step2GroupingAndSkeletonTests(unittest.TestCase):
    def test_q05_groups_three_category_scope_with_org_collaboration_and_separates_promotion_structure(self):
        facts = [
            {
                "topic": "分析対象と区分",
                "reference": "資料1",
                "quote": "従業員 100 人以下の企業について、DX に関係する取組内容について、それぞれ「DX 取組なし」「DX 成果なし」「DX 成果あり」の 3 区分に分けて比較した結果を図表- 3 に示す。",
                "fact": "100人以下企業を3区分で比較",
            },
            {
                "topic": "別名のトピック（組織連携）",
                "reference": "資料2",
                "quote": "DX 取組なしの企業ではともに「実施している」割合が 2 割前後、DX 成果なしの企業では 3 割 5 分、DX 成果ありの企業では6 割前後と、それぞれ大きな差が見られる。",
                "fact": "組織連携の割合",
            },
            {
                "topic": "推進体制",
                "reference": "資料3",
                "quote": "「IT 分野に見識がある役員」の有無、「DX 推進やデジタルビジネス強化の部署」といった推進体制については DX の成果との関係は顕著ではなかった。",
                "fact": "推進体制とDX成果の関係は顕著ではない",
            },
        ]
        groups = _group_complementary_facts(
            facts,
            "従業員100人以下の企業では、DXの成果の有無によって組織連携の実施割合や推進体制にどのような違いがありますか？",
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual([f["reference"] for f in groups[0]], ["資料1", "資料2"])
        self.assertEqual([f["reference"] for f in groups[1]], ["資料3"])

        skeleton_0 = _build_skeleton_claim(groups[0])
        self.assertEqual(skeleton_0["references"], ["資料1", "資料2"])
        self.assertEqual(
            skeleton_0["formatted_claim"],
            "従業員100人以下の企業について、DXに関係する取組内容について、それぞれ「DX取組なし」「DX成果なし」「DX成果あり」の3区分に分けて比較した結果を図表-3に示す。"
            "DX取組なしの企業ではともに「実施している」割合が2割前後、DX成果なしの企業では3割5分、DX成果ありの企業では6割前後と、それぞれ大きな差が見られる[資料1][資料2]。",
        )

        skeleton_1 = _build_skeleton_claim(groups[1])
        self.assertEqual(skeleton_1["references"], ["資料3"])
        self.assertEqual(
            skeleton_1["formatted_claim"],
            "「IT分野に見識がある役員」の有無、「DX推進やデジタルビジネス強化の部署」といった推進体制についてはDXの成果との関係は顕著ではなかった[資料3]。",
        )

    def test_q01_combines_scale_trend_quotes_and_q02_keeps_unconfirmed_quotes_separate(self):
        q01_facts = [
            {
                "topic": "全体傾向",
                "reference": "資料1",
                "quote": "日本の企業では従業員規模が小さくなるほど DX の取組も少なくなっていることを示した。",
                "fact": "規模が小さいほど取組が少ない",
            },
            {
                "topic": "規模別内訳",
                "reference": "資料2",
                "quote": "DX に取組んでいる企業の割合は従業員規模が小さいほど低くなっている。また、従業員100 人以下の中小企業では、「DX に取組む予定はない」が 3 割強であり、従業員 101 人以上の企業の回答と比べて著しく高い。",
                "fact": "100人以下では予定なしが3割強",
            },
            {
                "topic": "成果の有無",
                "reference": "資料3",
                "quote": "従業員規模が小さいほど DX の取組の割合は低いものの、100 人以下の企業でも DX に取組んでいる企業は DX の基本的な取組領域で成果が出ていた。",
                "fact": "基本的な領域で成果が出ていた",
            },
        ]
        q01_groups = _group_complementary_facts(
            q01_facts[:2],
            "従業員規模が小さくなると、企業のDX取組割合はどのように変化しますか？",
        )
        self.assertEqual(len(q01_groups), 1)
        q01_skeleton = _build_skeleton_claim(q01_groups[0])
        self.assertEqual(q01_skeleton["references"], ["資料1", "資料2"])
        self.assertIn("[資料1][資料2]。", q01_skeleton["formatted_claim"])

        q02_groups = _group_complementary_facts(
            [q01_facts[1], q01_facts[2]],
            "従業員100人以下の企業におけるDXの取組状況には、どのような特徴がありますか？",
        )
        self.assertEqual(len(q02_groups), 2)


class Step3QualityVerificationTests(unittest.TestCase):
    def test_saas_llm_claim_omitting_numbers_and_modality_falls_back_to_python_baseline(self):
        question = "DXの成果を出している企業では、SaaSやパブリッククラウドをどのように活用していますか？"
        quote = (
            "SaaS を活用している企業の達成度は活用していない企業の約 2 倍、"
            "パブリッククラウドでは約 3 倍であり、両技術が迅速かつ拡張性の高い IT システム導入に寄与したと推察される。"
        )
        verified_facts = [
            {
                "topic": "具体的な数値・割合",
                "reference": "資料4",
                "quote": quote,
                "fact": "SaaSやパブリッククラウドの活用と達成度の関係",
            }
        ]
        sources = {
            "資料4": {
                "reference": "資料4",
                "document_id": "DOC-001",
                "page_number": 5,
                "chunk_number": 2,
                "content": quote,
            }
        }
        degraded_stage2 = response_json(
            [
                claim(
                    "DXの成果を出している企業では、SaaSやパブリッククラウドを迅速かつ拡張性の高い IT システム導入に寄与しています。",
                    "資料4",
                    quote,
                )
            ]
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=degraded_stage2,
        ):
            result = _generate_answer_from_facts(question, verified_facts, sources)

        self.assertEqual(
            result["answer"],
            "SaaSを活用している企業の達成度は活用していない企業の約2倍、パブリッククラウドでは約3倍であり、"
            "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察される[資料4]。",
        )
        self.assertEqual(
            result["sources"],
            [
                {
                    "reference": "資料4",
                    "document_id": "DOC-001",
                    "page_number": 5,
                    "chunk_number": 2,
                }
            ],
        )

    def test_q04_hollow_difference_and_meta_refusal_claims_are_rejected(self):
        question = "従業員規模によって、DXの成果が出ている割合にはどのような違いがありますか？"
        scope_quote = (
            "従業員 100 人以下の企業について、DX に関係する取組内容について、"
            "それぞれ「DX 取組なし」「DX 成果なし」「DX 成果あり」の 3 区分に分けて比較した結果を図表- 3 に示す。"
        )
        verified_facts = [
            {
                "topic": "比較区分",
                "reference": "資料2",
                "quote": scope_quote,
                "fact": "3区分で比較",
            }
        ]
        sources = {
            "資料1": {
                "reference": "資料1",
                "document_id": "DOC-001",
                "page_number": 2,
                "chunk_number": 1,
                "content": "DXの成果が出ている割合で見ると従業員規模による差は小さく、100人以下でも成果が出ている。",
            },
            "資料2": {
                "reference": "資料2",
                "document_id": "DOC-001",
                "page_number": 3,
                "chunk_number": 1,
                "content": scope_quote,
            },
        }
        bad_stage2 = response_json(
            [
                claim(
                    "DXの成果が出ている割合には、従業員規模によって違いがあります。",
                    "資料2",
                    scope_quote,
                ),
                claim(
                    "具体的な数値は記載されていません。",
                    "資料2",
                    scope_quote,
                ),
            ]
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=bad_stage2,
        ):
            result = _generate_answer_from_facts(question, verified_facts, sources)

        self.assertNotIn("違いがあります", result["answer"])
        self.assertNotIn("具体的な数値は記載されていません", result["answer"])
        self.assertIn("確認できた関連資料の原文を表示します。", result["answer"])

    def test_ambiguous_mapping_falls_back_to_python_baseline(self):
        question = "A社とB社の割合は？"
        quote = "A社とB社ではそれぞれ2割、5割となっている。"
        skel = _build_skeleton_claim(
            [{"topic": "割合", "reference": "資料1", "quote": quote, "fact": "割合"}]
        )
        self.assertFalse(
            _is_verified_llm_claim(
                "A社は2割、B社は5割です。", [quote], question, skel
            )
        )

    def test_swapped_numbers_between_subjects_are_rejected_and_valid_mapping_is_accepted(self):
        question = "DXの成果を出している企業では、SaaSやパブリッククラウドをどのように活用していますか？"
        quote = (
            "SaaS を活用している企業の達成度は活用していない企業の約 2 倍、"
            "パブリッククラウドでは約 3 倍であり、両技術が迅速かつ拡張性の高い IT システム導入に寄与したと推察される。"
        )
        skel = _build_skeleton_claim(
            [{"topic": "活用状況", "reference": "資料4", "quote": quote, "fact": "SaaSとクラウド"}]
        )

        # 正しい対応関係の回答は採用される
        self.assertTrue(
            _is_verified_llm_claim(
                "SaaSを活用している企業の達成度は活用していない企業の約2倍、パブリッククラウドでは約3倍であり、"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )
        self.assertTrue(
            _is_verified_llm_claim(
                "SaaSは活用していない企業と比較して約2倍、パブリッククラウドは約3倍であり、"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )
        # 主語の順序を入れ替えても対応関係が正しければ採用される
        self.assertTrue(
            _is_verified_llm_claim(
                "パブリッククラウドは活用していない企業と比較して約3倍、SaaSは約2倍であり、"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )

        # 数値を入れ替えた誤回答（文形式・コロン形式・主語順序入れ替え・比較対象入れ替え）はすべて棄却される
        self.assertFalse(
            _is_verified_llm_claim(
                "活用していない企業の達成度はSaaSを活用している企業の約2倍、パブリッククラウドでは約3倍であり、"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )
        self.assertFalse(
            _is_verified_llm_claim(
                "SaaSを活用している企業の達成度は活用していない企業の約3倍、パブリッククラウドでは約2倍であり、"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )
        self.assertFalse(
            _is_verified_llm_claim(
                "- SaaS：約3倍\n- パブリッククラウド：約2倍\n"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )
        self.assertFalse(
            _is_verified_llm_claim(
                "- SaaS：活用していない企業と比較して約3倍\n"
                "- パブリッククラウド：活用していない企業と比較して約2倍\n"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )
        self.assertFalse(
            _is_verified_llm_claim(
                "パブリッククラウドの達成度は活用していない企業の約2倍、SaaSを活用している企業では約3倍であり、"
                "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                [quote],
                question,
                skel,
            )
        )

        # 第2段階で数値が入れ替わった場合、Python基準回答へフォールバックする
        verified_facts = [
            {
                "topic": "具体的な数値・割合",
                "reference": "資料4",
                "quote": quote,
                "fact": "SaaSやパブリッククラウドの活用と達成度の関係",
            }
        ]
        sources = {
            "資料4": {
                "reference": "資料4",
                "document_id": "DOC-001",
                "page_number": 5,
                "chunk_number": 2,
                "content": quote,
            }
        }
        swapped_stage2 = response_json(
            [
                claim(
                    "SaaSを活用している企業の達成度は活用していない企業の約3倍、パブリッククラウドでは約2倍であり、"
                    "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。",
                    "資料4",
                    quote,
                )
            ]
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=swapped_stage2,
        ):
            result = _generate_answer_from_facts(question, verified_facts, sources)

        self.assertEqual(
            result["answer"],
            "SaaSを活用している企業の達成度は活用していない企業の約2倍、パブリッククラウドでは約3倍であり、"
            "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察される[資料4]。",
        )

    def test_build_skeleton_claim_does_not_append_period_not_in_original_quote(self):
        skel = _build_skeleton_claim(
            [
                {
                    "topic": "テスト",
                    "reference": "資料1",
                    "quote": "句点で終わらない原文引用",
                    "fact": "事実",
                }
            ]
        )
        self.assertEqual(skel["text"], "句点で終わらない原文引用")
        self.assertEqual(skel["formatted_claim"], "句点で終わらない原文引用[資料1]")

    def test_extra_unbacked_subject_number_or_comparison_is_rejected_and_falls_back(self):
        # 1. 原文にない対象と数値の組み合わせをLLMが追加した場合、拒否されPython基準回答にフォールバックする
        saas_question = "DXの成果を出している企業では、SaaSやパブリッククラウドをどのように活用していますか？"
        saas_quote = (
            "SaaS を活用している企業の達成度は活用していない企業の約 2 倍、"
            "パブリッククラウドでは約 3 倍であり、両技術が迅速かつ拡張性の高い IT システム導入に寄与したと推察される。"
        )
        saas_skel = _build_skeleton_claim(
            [{"topic": "活用状況", "reference": "資料4", "quote": saas_quote, "fact": "SaaSとクラウド"}]
        )
        extra_subject_text = (
            "SaaSを活用している企業の達成度は活用していない企業の約2倍、パブリッククラウドでは約3倍、"
            "オンプレミスでは約2倍であり、両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。"
        )
        self.assertFalse(
            _is_verified_llm_claim(
                extra_subject_text,
                [saas_quote],
                saas_question,
                saas_skel,
            )
        )
        saas_facts = [
            {
                "topic": "具体的な数値・割合",
                "reference": "資料4",
                "quote": saas_quote,
                "fact": "SaaSやパブリッククラウドの活用と達成度の関係",
            }
        ]
        saas_sources = {
            "資料4": {
                "reference": "資料4",
                "document_id": "DOC-001",
                "page_number": 5,
                "chunk_number": 2,
                "content": saas_quote,
            }
        }
        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=response_json([claim(extra_subject_text, "資料4", saas_quote)]),
        ):
            saas_fallback_res = _generate_answer_from_facts(
                saas_question, saas_facts, saas_sources
            )
        self.assertEqual(
            saas_fallback_res["answer"],
            "SaaSを活用している企業の達成度は活用していない企業の約2倍、パブリッククラウドでは約3倍であり、"
            "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察される[資料4]。",
        )

        # 1-b. 原文の同じ数値セット（SaaS：約2倍）を重複して出力した場合も一対一照合で拒否される
        duplicated_saas_text = (
            "SaaSを活用している企業の達成度は活用していない企業の約2倍、パブリッククラウドでは約3倍、"
            "またSaaSでは活用していない企業の約2倍であり、両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。"
        )
        self.assertFalse(
            _is_verified_llm_claim(
                duplicated_saas_text,
                [saas_quote],
                saas_question,
                saas_skel,
            )
        )
        duplicated_only_saas_text = (
            "SaaSを活用している企業の達成度は活用していない企業の約2倍、またSaaSでは約2倍であり、"
            "両技術が迅速かつ拡張性の高いITシステム導入に寄与したと推察されます。"
        )
        self.assertFalse(
            _is_verified_llm_claim(
                duplicated_only_saas_text,
                [saas_quote],
                saas_question,
                saas_skel,
            )
        )

        # 2. 原文に比較対象がないのに、LLMが比較対象を追加した場合、拒否されPython基準回答にフォールバックする
        org_question = "組織連携の実施割合にどのような違いがありますか？"
        org_quote = "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる。"
        org_skel = _build_skeleton_claim(
            [{"topic": "組織連携", "reference": "資料1", "quote": org_quote, "fact": "組織連携の割合"}]
        )
        # 比較対象なしの正しい回答は許可される
        self.assertTrue(
            _is_verified_llm_claim(
                "組織連携の実施割合は、DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られます。",
                [org_quote],
                org_question,
                org_skel,
            )
        )
        # 原文にない比較対象（「DX取組なしの企業と比較して」等）を追加した回答は拒否される
        self.assertFalse(
            _is_verified_llm_claim(
                "組織連携の実施割合は、DX取組なしの企業と比較してDX成果なしの企業で3割5分、"
                "DX成果ありの企業で6割前後と差が見られます。",
                [org_quote],
                org_question,
                org_skel,
            )
        )

        org_facts = [
            {
                "topic": "組織連携",
                "reference": "資料1",
                "quote": org_quote,
                "fact": "DX成果なしは3割5分、DX成果ありは6割前後である。",
            }
        ]
        org_sources = {
            "資料1": {
                "reference": "資料1",
                "document_id": "doc-1",
                "page_number": 3,
                "chunk_number": 1,
                "content": org_quote,
            }
        }
        added_comp_stage2 = response_json(
            [
                claim(
                    "組織連携の実施割合は、DX取組なしの企業と比較してDX成果なしの企業で3割5分、"
                    "DX成果ありの企業で6割前後と差が見られます。",
                    "資料1",
                    org_quote,
                )
            ]
        )
        with patch(
            "app.generation.generate_answer._call_ollama",
            return_value=added_comp_stage2,
        ):
            fallback_res = _generate_answer_from_facts(org_question, org_facts, org_sources)

        self.assertEqual(
            fallback_res["answer"],
            "DX成果なしの企業で3割5分、DX成果ありの企業で6割前後と差が見られる[資料1]。",
        )


if __name__ == "__main__":
    unittest.main()
