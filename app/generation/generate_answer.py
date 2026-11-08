import ast
import json
import logging
import re
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
            "items": {"type": "string"},
        }
    },
    "required": ["facts"],
}


class _OllamaRequestError(RuntimeError):
    """Ollamaへの通信または応答形式の失敗を表す。"""


class _FactExtractionError(ValueError):
    """チャンクからの事実抽出結果が不正な場合に使う。"""


def _call_ollama(prompt, response_format=None):
    """既存の設定でOllamaを呼び出し、response文字列を返す。"""
    payload = {
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},
    }
    if response_format is not None:
        payload["format"] = response_format

    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        API_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=120) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except (OSError, TimeoutError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise _OllamaRequestError(f"Ollamaへのリクエストに失敗しました: {error}") from error

    if not isinstance(response_data, dict) or not isinstance(
        response_data.get("response"), str
    ):
        details = response_data.get("error") if isinstance(response_data, dict) else None
        message = f"Ollama APIエラー: {details}" if details else (
            "Ollamaの応答に文字列のresponseがありません。"
        )
        raise _OllamaRequestError(message)

    return response_data["response"]


def _extract_facts_from_chunk(chunk, reference):
    """1チャンクから明記された事実を抽出し、Python管理の出典情報と返す。"""
    prompt = (
        "次の1つの資料チャンクから、明記されている事実だけを抽出してください。\n"
        "推測、評価、独自の解釈、資料にない結論を追加しないでください。\n"
        "数値、割合、比較対象、対象範囲、否定表現を原文どおり正確に保ってください。\n"
        "事実がない場合は facts を空配列 [] にしてください。\n"
        "出力は次のJSON Schemaに厳密に従うJSONオブジェクトだけにしてください: "
        '{"type":"object","properties":{"facts":{"type":"array","items":{"type":"string"}}},"required":["facts"]}\n'
        "facts の各要素は、事実の本文そのものを表すプレーンな文字列にしてください。\n"
        "キー名付きの辞書、オブジェクト、辞書やリストを文字列化したものは禁止です。\n"
        "正しい例: {\"facts\": [\"従業員規模が小さいほど、DXに取組んでいる企業の割合が低くなる\"]}\n"
        "正しい例: {\"facts\": [\"事実A\", \"事実B\"]} または {\"facts\": []}\n"
        "誤りの例: {\"facts\": [\"{'事実': '従業員規模が小さいほど、DXに取組んでいる企業の割合が低くなる'}\"]}\n"
        "誤りの例: {\"事実\": \"従業員規模が小さいほど、DXに取組んでいる企業の割合が低くなる\"}\n"
        "資料番号や出典情報は出力しないでください。\n\n"
        f"チャンク本文:\n{chunk['content']}"
    )
    raw_facts = _call_ollama(prompt, response_format=FACTS_RESPONSE_SCHEMA)
    logger.debug("%sの事実抽出Ollama生レスポンス: %r", reference, raw_facts)
    try:
        parsed = json.loads(raw_facts)
    except json.JSONDecodeError as error:
        raise _FactExtractionError(
            f"{reference}の抽出結果がJSON形式ではありません。"
        ) from error

    if not isinstance(parsed, dict) or not isinstance(parsed.get("facts"), list):
        raise _FactExtractionError(
            f"{reference}の抽出結果は facts 配列を持つJSONオブジェクトである必要があります。"
        )
    facts = parsed["facts"]
    for fact in facts:
        if not isinstance(fact, str) or not fact.strip():
            raise _FactExtractionError(
                f"{reference}の抽出結果は、空でない文字列のJSON配列である必要があります。"
            )
        try:
            serialized_value = ast.literal_eval(fact.strip())
        except (SyntaxError, ValueError):
            continue
        if isinstance(serialized_value, (list, dict, tuple, set)):
            raise _FactExtractionError(
                f"{reference}の抽出結果に、事実ではなく文字列化されたコンテナがあります。"
            )

    logger.debug("%sの抽出事実: %s", reference, facts)
    return {
        "reference": reference,
        "document_id": chunk["document_id"],
        "page_number": chunk["page_number"],
        "chunk_number": chunk["chunk_number"],
        "facts": facts,
    }


def _move_leading_citations_to_sentence_end(answer):
    """同一文の文頭にある引用番号を、その文の句点直前へ移す。"""

    def replace_leading_citations(match):
        citations = "".join(re.findall(r"\[資料\d+\]", match.group(2)))
        return f"{match.group(1)}{match.group(3)}{citations}{match.group(4)}"

    # 文頭（文字列先頭・句点直後・改行直後）の引用だけを対象にし、改行はまたがない。
    return re.sub(
        r"(^|(?<=[。！？!?])|(?<=\n))"
        r"((?:[ \t]*\[資料\d+\])+)"
        r"[ \t]*"
        r"([^。！？!?\n]+)"
        r"([。！？!?])",
        replace_leading_citations,
        answer,
    )


def _generate_answer_from_facts(question, extracted_facts):
    """抽出事実だけから回答を作り、元の資料番号を引用させる。"""
    fact_context = []
    for extraction in extracted_facts:
        for fact in extraction["facts"]:
            fact_context.append(
                f"[{extraction['reference']}] {fact}"
            )

    available_references = "、".join(
        dict.fromkeys(
            extraction["reference"]
            for extraction in extracted_facts
            if extraction["facts"]
        )
    )
    prompt = (
        "以下の抽出済み事実だけを使い、質問に直接、簡潔に回答してください。回答は最大3文です。\n"
        "抽出済み事実に書かれていない情報を、一般知識や推測で補わないでください。\n"
        "抽出済み事実から直接読み取れない原因、理由、背景を説明しないでください。\n"
        "説明する各文には、その文の根拠となる元の資料番号を必ず付けてください。\n"
        "引用番号を文頭に置かないでください。「[資料2]によると、」の形式は禁止です。\n"
        "引用番号は各説明文の末尾、句点の直前に付けてください。形式例: 説明文です[資料2]。\n"
        "複数の資料を使う場合は、根拠となる資料番号を続けて付けてください。形式例: 説明文です[資料1][資料2]。\n"
        "資料番号は再採番せず、抽出事実に付いている番号をそのまま使ってください。\n"
        f"使用できる資料番号: {available_references}\n"
        f"根拠が不足する場合は「{NO_SOURCE_ANSWER}」だけを返してください。\n\n"
        "以下の禁止例と期待例に従ってください。\n"
        "抽出済み事実: [資料2] 従業員100人以下の中小企業では、「DXに取組む予定はない」が3割強である。\n"
        "禁止: 「これは、DXへの取り組みに対する意欲が低いことを示しています[資料2]。」\n"
        "理由: 「意欲が低い」は抽出済み事実に直接書かれていない解釈です。\n"
        "期待する回答例: 「従業員100人以下の中小企業では、「DXに取組む予定はない」と回答した割合が3割強です[資料2]。」\n\n"
        "抽出済み事実:\n"
        + "\n".join(fact_context)
        + f"\n\n質問: {question}\n回答:"
    )
    logger.debug("第2段階のfact_context: %s", fact_context)
    logger.debug("第2段階でOllamaに送信するプロンプト全文: %s", prompt)
    answer = _call_ollama(prompt)
    logger.debug("第2段階のOllama加工前回答: %r", answer)
    return answer


def generate_answer(question, chunks):
    """検索した資料だけを根拠に回答を作り、回答で引用した出典を返す。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("質問文には空でない文字列を指定してください。")
    if not isinstance(chunks, list):
        raise ValueError("chunksには資料辞書のリストを指定してください。")

    # 参考資料がない場合は、根拠のない回答を生成しない。
    if not chunks:
        return {"answer": NO_SOURCE_ANSWER, "sources": []}

    sources = []
    extracted_facts = []
    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict):
            raise ValueError(f"{index}件目の資料が辞書形式ではありません。")

        try:
            document_id = chunk["document_id"]
            page_number = chunk["page_number"]
            content = chunk["content"]
        except KeyError as error:
            raise ValueError(
                f"{index}件目の資料に必要な項目がありません: {error.args[0]}"
            ) from error

        chunk_number = chunk.get("chunk_number")
        # 資料番号はプロンプト内の引用番号と出典一覧で共通に使う。
        reference = f"資料{index}"
        sources.append(
            {
                "reference": reference,
                "document_id": document_id,
                "page_number": page_number,
            }
        )
        logger.debug("%sの事実抽出を開始します。", reference)
        try:
            extraction = _extract_facts_from_chunk(
                {
                    "document_id": document_id,
                    "page_number": page_number,
                    "chunk_number": chunk_number,
                    "content": content,
                },
                reference,
            )
        except (_OllamaRequestError, _FactExtractionError) as error:
            logger.error("%sの事実抽出に失敗しました: %s", reference, error)
            continue

        if extraction["facts"]:
            logger.debug("%sから%d件の事実を抽出しました。", reference, len(extraction["facts"]))
            extracted_facts.append(extraction)
        else:
            logger.debug("%sから抽出できる事実はありませんでした。", reference)

    if not any(extraction["facts"] for extraction in extracted_facts):
        logger.info("すべてのチャンクに抽出可能な事実がありません。")
        return {"answer": NO_SOURCE_ANSWER, "sources": []}

    try:
        answer = _generate_answer_from_facts(question, extracted_facts)
    except _OllamaRequestError as error:
        logger.error("第2段階の回答生成に失敗しました: %s", error)
        return {"answer": NO_SOURCE_ANSWER, "sources": []}

    allowed_references = {
        extraction["reference"]
        for extraction in extracted_facts
        if extraction["facts"]
    }
    raw_citations = re.findall(r"\[資料(\d+)\]", answer)
    logger.debug("Ollamaの加工前回答: %r", answer)

    # 事実を抽出できた資料番号以外は引用として扱わず、回答文から取り除く。
    answer = re.sub(
        r"\[資料(\d+)\]",
        lambda match: (
            match.group(0)
            if f"資料{match.group(1)}" in allowed_references
            else ""
        ),
        answer,
    )
    logger.debug("許可されていない引用番号の削除後の回答: %r", answer)
    logger.debug(
        "削除後に残った引用番号: %s",
        re.findall(r"\[資料(\d+)\]", answer),
    )

    # 句点の後ろに付いた引用も、その直前の文の引用として検査できる形にする。
    # 次の文の文頭引用まで取り込まないよう、行末または文字列末尾の引用だけを対象にする。
    answer = re.sub(
        r"([。！？!?])((?:[ \t]*\[資料\d+\])+)(?=[ \t]*(?:$|\n))",
        lambda match: (
            " ".join(re.findall(r"\[資料\d+\]", match.group(2))) + match.group(1)
        ),
        answer,
    )
    # 文頭の引用も、同じ文の句点直前へ移して既存の文末検証に合わせる。
    answer = _move_leading_citations_to_sentence_end(answer)
    logger.debug("引用位置を正規化した回答: %r", answer)

    valid_citations = set(
        re.findall(r"\[資料(\d+)\]", answer)
    )
    logger.debug("最終的に検出した有効な引用番号: %s", sorted(valid_citations))
    if not valid_citations:
        if not raw_citations:
            reason = "Ollamaの生回答に引用番号がありません"
        else:
            reason = "生回答の引用番号がすべて範囲外です"
        logger.warning(
            "回答を定型文に置き換えます。理由: 有効な引用がない（%s）",
            reason,
        )
        return {"answer": NO_SOURCE_ANSWER, "sources": []}

    # 各文に有効な引用があるか確認する。定型の不明回答は引用不要とする。
    sentences = re.split(r"(?<=[。！？!?])|\n+", answer)
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if sentence.strip("。！？!?") == NO_SOURCE_ANSWER:
            logger.debug(
                "文単位検査: 文=%r、文末に有効な引用=%s、結果=定型の不明回答として許可",
                sentence,
                False,
            )
            continue
        citation_match = re.search(r"\[資料(\d+)\]\s*[。！？!?]?$", sentence)
        logger.debug(
            "文単位検査: 文=%r、文末に有効な引用=%s",
            sentence,
            bool(citation_match),
        )
        if not citation_match:
            logger.warning(
                "回答を定型文に置き換えます。理由: 引用のない説明文がある"
                "（不合格の文=%r）",
                sentence,
            )
            return {"answer": NO_SOURCE_ANSWER, "sources": []}

    # 回答中で引用された資料だけを番号順に出典情報として返す。
    cited_indexes = sorted(
        {
            int(match) - 1
            for match in valid_citations
        }
    )
    cited_sources = [sources[index] for index in cited_indexes]

    return {"answer": answer, "sources": cited_sources}
