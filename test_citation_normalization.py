import unittest
from unittest.mock import patch

import app.generation.generate_answer as generation


CHUNK_1 = {
    "document_id": "doc-1",
    "page_number": 1,
    "chunk_number": 1,
    "content": "本文1",
}
CHUNK_2 = {
    "document_id": "doc-2",
    "page_number": 2,
    "chunk_number": 1,
    "content": "本文2",
}


class LeadingCitationNormalizationTests(unittest.TestCase):
    def test_move_single_leading_citation_to_sentence_end(self):
        self.assertEqual(
            generation._move_leading_citations_to_sentence_end(
                "[資料1] 本文です。"
            ),
            "本文です[資料1]。",
        )

    def test_move_multiple_leading_citations_to_sentence_end(self):
        self.assertEqual(
            generation._move_leading_citations_to_sentence_end(
                "[資料1][資料2] 本文です。"
            ),
            "本文です[資料1][資料2]。",
        )

    def test_already_trailing_citation_is_unchanged(self):
        text = "本文です[資料1]。"
        self.assertEqual(
            generation._move_leading_citations_to_sentence_end(text),
            text,
        )

    def test_multiple_sentences_with_different_leading_citations(self):
        self.assertEqual(
            generation._move_leading_citations_to_sentence_end(
                "[資料1] 一文目です。[資料2] 二文目です。"
            ),
            "一文目です[資料1]。二文目です[資料2]。",
        )

    def test_citation_separated_by_newline_is_not_moved_across(self):
        text = "[資料1]\n本文です。"
        self.assertEqual(
            generation._move_leading_citations_to_sentence_end(text),
            text,
        )


class CitationValidationIntegrationTests(unittest.TestCase):
    def test_leading_citation_answer_passes_after_normalization(self):
        extraction = {
            "reference": "資料1",
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_number": 1,
            "facts": ["従業員100人以下の中小企業では、DXに取組む予定はないが3割強である。"],
        }
        with (
            patch.object(
                generation,
                "_extract_facts_from_chunk",
                return_value=extraction,
            ),
            patch.object(
                generation,
                "_generate_answer_from_facts",
                return_value=(
                    "[資料1] 従業員100人以下の中小企業では、"
                    "「DXに取組む予定はない」が3割強です。"
                ),
            ),
        ):
            result = generation.generate_answer("質問", [CHUNK_1])

        self.assertEqual(
            result["answer"],
            "従業員100人以下の中小企業では、「DXに取組む予定はない」が3割強です[資料1]。",
        )
        self.assertEqual(result["sources"][0]["reference"], "資料1")

    def test_post_punctuation_citation_still_normalized(self):
        extraction = {
            "reference": "資料1",
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_number": 1,
            "facts": ["事実A"],
        }
        with (
            patch.object(
                generation,
                "_extract_facts_from_chunk",
                return_value=extraction,
            ),
            patch.object(
                generation,
                "_generate_answer_from_facts",
                return_value="本文です。[資料1]",
            ),
        ):
            result = generation.generate_answer("質問", [CHUNK_1])

        self.assertEqual(result["answer"], "本文です[資料1]。")

    def test_disallowed_citation_is_removed_before_validation(self):
        extraction = {
            "reference": "資料1",
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_number": 1,
            "facts": ["事実A"],
        }
        with (
            patch.object(
                generation,
                "_extract_facts_from_chunk",
                return_value=extraction,
            ),
            patch.object(
                generation,
                "_generate_answer_from_facts",
                return_value="[資料9] 本文です。",
            ),
        ):
            result = generation.generate_answer("質問", [CHUNK_1])

        self.assertEqual(
            result,
            {"answer": generation.NO_SOURCE_ANSWER, "sources": []},
        )

    def test_answer_without_citation_is_rejected(self):
        extraction = {
            "reference": "資料1",
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_number": 1,
            "facts": ["事実A"],
        }
        with (
            patch.object(
                generation,
                "_extract_facts_from_chunk",
                return_value=extraction,
            ),
            patch.object(
                generation,
                "_generate_answer_from_facts",
                return_value="引用のない文章です。",
            ),
        ):
            result = generation.generate_answer("質問", [CHUNK_1])

        self.assertEqual(
            result,
            {"answer": generation.NO_SOURCE_ANSWER, "sources": []},
        )

    def test_newline_separated_leading_citation_is_rejected(self):
        extraction = {
            "reference": "資料1",
            "document_id": "doc-1",
            "page_number": 1,
            "chunk_number": 1,
            "facts": ["事実A"],
        }
        with (
            patch.object(
                generation,
                "_extract_facts_from_chunk",
                return_value=extraction,
            ),
            patch.object(
                generation,
                "_generate_answer_from_facts",
                return_value="[資料1]\n本文です。",
            ),
        ):
            result = generation.generate_answer("質問", [CHUNK_1])

        self.assertEqual(
            result,
            {"answer": generation.NO_SOURCE_ANSWER, "sources": []},
        )

    def test_multi_sentence_leading_citations_keep_source_mapping(self):
        extractions = [
            {
                "reference": "資料1",
                "document_id": "doc-1",
                "page_number": 1,
                "chunk_number": 1,
                "facts": ["事実1"],
            },
            {
                "reference": "資料2",
                "document_id": "doc-2",
                "page_number": 2,
                "chunk_number": 1,
                "facts": ["事実2"],
            },
        ]

        def fake_extract(chunk, reference):
            return next(
                item for item in extractions if item["reference"] == reference
            )

        with (
            patch.object(
                generation,
                "_extract_facts_from_chunk",
                side_effect=fake_extract,
            ),
            patch.object(
                generation,
                "_generate_answer_from_facts",
                return_value="[資料1] 一文目です。[資料2] 二文目です。",
            ),
        ):
            result = generation.generate_answer("質問", [CHUNK_1, CHUNK_2])

        self.assertEqual(
            result["answer"],
            "一文目です[資料1]。二文目です[資料2]。",
        )
        self.assertEqual(
            [source["reference"] for source in result["sources"]],
            ["資料1", "資料2"],
        )


if __name__ == "__main__":
    unittest.main()
