import json
import logging
import re
import unicodedata
from urllib.request import Request, urlopen


logger = logging.getLogger(__name__)

API_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:3b-instruct"
NO_SOURCE_ANSWER = "資料から確認できません"
FACTS_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string"},
                    "quote": {"type": "string"},
                    "reference": {"type": "string"},
                    "fact": {"type": "string"},
                },
                "required": ["topic", "quote", "reference", "fact"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["facts"],
    "additionalProperties": False,
}
ANSWER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "citations": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "reference": {"type": "string"},
                                "quote": {"type": "string"},
                            },
                            "required": ["reference", "quote"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["text", "citations"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["claims"],
    "additionalProperties": False,
}
NUMBER_TOKEN_PATTERN = re.compile(
    r"\d+(?:\.\d+)?%?|[一二三四五六七八九十百千万]+(?:割|分|人|社|%)"
)


def _call_ollama(prompt, response_format):
    """OllamaにJSON形式の回答を1回要求する。"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": response_format,
        "options": {"temperature": 0},
    }
    request = Request(
        API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as error:
        logger.exception("Ollamaへのリクエストに失敗しました。")
        raise RuntimeError("Ollamaへのリクエストに失敗しました。") from error

    if not isinstance(response_data, dict) or not isinstance(
        response_data.get("response"), str
    ):
        raise RuntimeError("OllamaからJSON形式の回答を取得できませんでした。")
    return response_data["response"]


def _normalize_for_evidence_match(text):
    """NFKC正規化し、空白だけを除いて原文引用を照合する。"""
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(normalized.split())


def _is_valid_citation(reference, quote, source_by_reference):
    """referenceが検索結果に存在し、quoteが原文に一致するか検証する。"""
    if not isinstance(reference, str) or not isinstance(quote, str) or not quote.strip():
        return False
    source = source_by_reference.get(reference)
    if not source:
        return False
    normalized_quote = _normalize_for_evidence_match(quote)
    normalized_content = _normalize_for_evidence_match(source["content"])
    if not normalized_quote or normalized_quote not in normalized_content:
        logger.info("原文に存在しない引用を除外します: %s", reference)
        return False
    return True


def _has_unsupported_numbers(fact, quote):
    """quoteに存在しない数値がfactに含まれているかを補助的に検出する。"""
    normalized_fact = _normalize_for_evidence_match(fact)
    normalized_quote = _normalize_for_evidence_match(quote)
    for token in NUMBER_TOKEN_PATTERN.findall(normalized_fact):
        if token not in normalized_quote:
            return True
    return False


def _extract_verified_facts(question, context_parts, source_by_reference):
    """第1段階：質問に関係する事実と引用を抽出し、Python側で検証して返す。"""
    prompt = (
        "質問に答えるために必要な事実と原文の引用を、提示された資料本文から抽出してください。\n"
        "質問に複数の観点が含まれる場合は、以下の観点を別々の要素として漏れなく抽出してください。\n"
        "1. 対象範囲・分類条件（従業員100人以下、DX取組なし・DX成果なし・DX成果ありの3区分などの説明文）\n"
        "2. 具体的な数値・割合（組織連携の実施割合が2割前後・3割5分・6割前後など）\n"
        "3. 定性的な結論（推進体制とDX成果の関係は顕著ではないという結論など）\n"
        "図表タイトルや節見出しだけを根拠（quote）として抽出しないでください。\n"
        "数値を問われた場合は具体的な数値が書かれている本文を、定性的な結論を問われた場合はその結論が書かれている本文をquoteとして抽出してください。\n"
        "各要素では、まず観点（topic）を指定し、次に根拠となる本文（quote）を資料本文から一字一句そのまま抜き出し、そのquoteが含まれる資料番号（reference）を必ず「資料1」「資料2」のように角括弧なしで指定し、最後にquoteに基づく事実（fact）を記述してください。\n"
        "quoteに書かれていない数値や別の観点の数値をfactに含めてはいけません。\n"
        "JSON出力例：\n"
        "{\n"
        '  "facts": [\n'
        "    {\n"
        '      "topic": "組織連携の実施割合",\n'
        '      "quote": "資料本文からの原文引用",\n'
        '      "reference": "資料2",\n'
        '      "fact": "原文に基づく事実"\n'
        "    }\n"
        "  ]\n"
        "}\n"
        "質問に関係する根拠が資料本文にまったくない場合のみ、factsを空配列にしてください。\n\n"
        "【資料本文】\n"
        + "\n\n".join(context_parts)
        + f"\n\n【質問】\n{question.strip()}"
    )

    raw_response = _call_ollama(prompt, FACTS_RESPONSE_SCHEMA)
    try:
        response = json.loads(raw_response)
    except json.JSONDecodeError:
        logger.warning("第1段階の応答がJSONではないため採用しません。")
        return []

    facts = response.get("facts") if isinstance(response, dict) else None
    if not isinstance(facts, list):
        logger.warning("第1段階の応答にfacts配列がないため採用しません。")
        return []

    verified_facts = []
    for item in facts:
        if not isinstance(item, dict):
            continue
        topic = item.get("topic")
        reference = item.get("reference")
        quote = item.get("quote")
        fact = item.get("fact")
        if not isinstance(topic, str) or not topic.strip():
            continue
        if not _is_valid_citation(reference, quote, source_by_reference):
            continue

        verified_fact = None
        if isinstance(fact, str) and fact.strip():
            if _has_unsupported_numbers(fact, quote):
                logger.info("引用に存在しない数値を含むfactを破棄します: %s", reference)
            else:
                verified_fact = fact.strip()

        verified_facts.append(
            {
                "topic": topic.strip(),
                "reference": reference,
                "quote": quote.strip(),
                "fact": verified_fact,
            }
        )

    return verified_facts


def _format_claim(text, references):
    """検証済み資料番号を主張の文末に付ける。"""
    text = re.sub(r"\[資料\d+\]", "", text).strip()
    if not text:
        return ""
    citations = "".join(f"[{reference}]" for reference in references)
    punctuation = re.search(r"([。！？!?])$", text)
    if punctuation:
        return f"{text[:-1].rstrip()}{citations}{punctuation.group(1)}"
    return f"{text}{citations}"


def _is_verified_in_stage1(reference, quote, verified_facts):
    """第2段階の引用が第1段階で検証済みの(reference, quote)と対応しているか確認する。"""
    if not isinstance(reference, str) or not isinstance(quote, str) or not quote.strip():
        return False
    normalized_quote = _normalize_for_evidence_match(quote)
    if not normalized_quote:
        return False
    for item in verified_facts:
        if not isinstance(item, dict) or item.get("reference") != reference:
            continue
        stage1_quote = _normalize_for_evidence_match(item.get("quote", ""))
        if stage1_quote and normalized_quote in stage1_quote:
            return True
    logger.info("第1段階で検証されていない引用を除外します: %s", reference)
    return False


def _build_fallback_from_chunks(source_by_reference):
    """検証済みclaimが0件の場合、検索結果チャンクを原文のまま返す。"""
    if not source_by_reference:
        return {"answer": NO_SOURCE_ANSWER, "sources": []}

    blocks = ["確認できた関連資料の原文を表示します。"]
    sources = []
    for reference, source in source_by_reference.items():
        blocks.append(
            f"[{reference}] 文書ID: {source['document_id']} / "
            f"ページ: {source['page_number']} / "
            f"チャンク: {source.get('chunk_number')}\n"
            f"{source['content'].strip()}"
        )
        sources.append(
            {
                "reference": reference,
                "document_id": source["document_id"],
                "page_number": source["page_number"],
                "chunk_number": source.get("chunk_number"),
            }
        )
    return {"answer": "\n\n".join(blocks), "sources": sources}


def _generate_answer_from_facts(question, verified_facts, source_by_reference):
    """第2段階：検証済みの事実と引用から最終回答と出典を生成する。"""
    usable_facts = [
        item
        for item in verified_facts
        if isinstance(item, dict)
        and isinstance(item.get("fact"), str)
        and item["fact"].strip()
    ]
    if not usable_facts:
        return _build_fallback_from_chunks(source_by_reference)

    evidence_parts = []
    for index, item in enumerate(usable_facts, start=1):
        lines = [
            f"[根拠{index}]",
            f"観点(topic): {item['topic']}",
            f"資料番号(reference): {item['reference']}",
            f"原文引用(quote): {item['quote']}",
            f"事実の要約(fact): {item['fact']}",
        ]
        evidence_parts.append("\n".join(lines))

    prompt = (
        "質問に対し、提示された【検証済み根拠】の原文引用（quote）を根拠として日本語で回答を作成してください。\n"
        "事実の要約（fact）が記載されている場合も原文引用（quote）を優先し、factの記載がない場合は検証済みの原文引用（quote）だけを根拠として回答してください。\n"
        "原文引用（quote）に直接裏付けられない内容・数値・推測は回答に含めないでください。\n"
        "観点（topic）ごとに情報を区別して主張（claims）を分け、異なるtopicの数値や結論を別のtopicの回答に流用しないでください（例：組織連携の数値を推進体制の回答に使ってはいけません）。\n"
        "対象範囲（従業員100人以下など）・分類条件・具体的な数値が別々の根拠に分かれている場合は、対象範囲と分類条件が一致している場合のみ組み合わせ、対応する複数の引用をcitationsに含めてください。\n"
        "比較や違いを尋ねられた場合、原文引用（quote）に具体的な数値・割合が記載されている項目は、「違いがあります」だけで終わらせず、各分類条件（DX取組なし・DX成果なし・DX成果ありなど）と数値を正確に対応させてtextに明記してください。\n"
        "その項目について原文引用（quote）に具体的な数値が記載されていない場合は、推測や別項目の数値の流用をせず、原文引用に書かれている定性的な結論（差が小さい、関係は顕著ではないなど）を省略せずに回答してください。\n"
        "各claimのcitationsには、そのclaimのtextに含まれるすべての数値・対象範囲・結論を直接裏付ける【検証済み根拠】のreferenceと原文引用（quote）を一字一句変えずに {\"reference\": \"資料1\", \"quote\": \"原文の引用\"} の形式で1件以上必ず指定してください（そのclaim自身の引用にない数値をtextに含めてはいけません）。\n"
        "質問に答える根拠がない場合のみ、claimsを空配列にしてください。\n\n"
        "【検証済み根拠】\n"
        + "\n\n".join(evidence_parts)
        + f"\n\n【質問】\n{question.strip()}"
    )

    raw_response = _call_ollama(prompt, ANSWER_RESPONSE_SCHEMA)
    try:
        response = json.loads(raw_response)
    except json.JSONDecodeError:
        logger.warning("第2段階の応答がJSONではないため採用しません。")
        return _build_fallback_from_chunks(source_by_reference)

    claims = response.get("claims") if isinstance(response, dict) else None
    if not isinstance(claims, list):
        logger.warning("第2段階の応答にclaims配列がないため採用しません。")
        return _build_fallback_from_chunks(source_by_reference)

    answer_claims = []
    used_references = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = claim.get("text")
        citations = claim.get("citations")
        if not isinstance(text, str) or not text.strip() or not isinstance(citations, list):
            continue
        if text.strip() == NO_SOURCE_ANSWER:
            continue

        valid_references = []
        valid_quotes = []
        for citation in citations:
            if not isinstance(citation, dict):
                continue
            reference = citation.get("reference")
            quote = citation.get("quote")
            if not _is_valid_citation(reference, quote, source_by_reference):
                continue
            if not _is_verified_in_stage1(reference, quote, usable_facts):
                continue
            valid_quotes.append(quote)
            if reference not in valid_references:
                valid_references.append(reference)

        if not valid_references:
            continue
        if _has_unsupported_numbers(text, "\n".join(valid_quotes)):
            logger.info("引用に存在しない数値を含むclaimを除外します")
            continue

        formatted_claim = _format_claim(text, valid_references)
        if not formatted_claim:
            continue
        answer_claims.append(formatted_claim)
        for reference in valid_references:
            if reference not in used_references:
                used_references.append(reference)

    if not answer_claims:
        return _build_fallback_from_chunks(source_by_reference)

    sources = [
        {
            "reference": reference,
            "document_id": source_by_reference[reference]["document_id"],
            "page_number": source_by_reference[reference]["page_number"],
            "chunk_number": source_by_reference[reference].get("chunk_number"),
        }
        for reference in used_references
    ]
    return {"answer": "\n".join(answer_claims), "sources": sources}


def generate_answer(question, chunks):
    """検索チャンクから回答を生成し、原文引用を検証して返す。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("質問文には空でない文字列を指定してください。")
    if not isinstance(chunks, list):
        raise ValueError("chunksには資料辞書のリストを指定してください。")

    source_by_reference = {}
    context_parts = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            continue
        content = chunk.get("content")
        if not isinstance(content, str) or not content.strip():
            continue

        reference = f"資料{index}"
        source_by_reference[reference] = {
            "reference": reference,
            "document_id": chunk.get("document_id"),
            "page_number": chunk.get("page_number"),
            "chunk_number": chunk.get("chunk_number"),
            "content": content,
        }
        context_parts.append(f"[{reference}]\n{content.strip()}")

    if not context_parts:
        return {"answer": NO_SOURCE_ANSWER, "sources": []}

    try:
        verified_facts = _extract_verified_facts(
            question, context_parts, source_by_reference
        )
        if not verified_facts:
            return _build_fallback_from_chunks(source_by_reference)
        return _generate_answer_from_facts(
            question, verified_facts, source_by_reference
        )
    except RuntimeError:
        return {
            "answer": "回答の生成に失敗しました。もう一度お試しください。",
            "sources": [],
        }
