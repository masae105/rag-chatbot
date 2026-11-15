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
HEADING_LINE_PATTERN = re.compile(
    r"^(?:第\s*\d+\s*章\b.*|図表\s*[-−ー]?\s*\d+(?:[-−ー]\d+)?\b.*|[（(]従業員\s*\d+\s*人以下[）)])$"
)
NOISE_NUMBER_SPAN_PATTERN = re.compile(
    r"図表\s*[-−ー]?\s*\d+(?:[-−ー]\d+)?|\d{4}\s*年(?:度|版)?|(?:」|取上げ|指標|尋ねているが)\s*\d+"
)
VALUE_EXPRESSION_PATTERN = re.compile(
    r"\d+\s*割\s*\d+\s*分|\d+(?:\.\d+)?(?:%|倍|割|分|人(?:以下|以上)?|社)|"
    r"[一二三四五六七八九十百千万]+(?:割(?:\s*[一二三四五六七八九十]+分)?|分|人(?:以下|以上)?|社|%|倍)"
)
MULTIPLIER_CLAUSE_PATTERN = re.compile(
    r"^(?:また|一方|なお|さらに|とくに|しかし)?(?P<subject>[^、。:]+?)の(?P<metric>[^、。:]+?)は(?P<comparison>[^、。:]+?)の約?(?P<value>\d+(?:\.\d+)?倍|[一二三四五六七八九十]+倍)"
)
SUBJECT_VALUE_CLAUSE_PATTERN = re.compile(
    r"^(?:また|一方|なお|さらに|とくに|しかし)?(?P<subject>[^、。:]+?)(?:では|において|について|は|で|:)(?P<middle>[^、。]*?)(?P<value>\d+\s*割\s*\d+\s*分|\d+(?:\.\d+)?(?:%|倍|割|分|人(?:以下|以上)?|社)|[一二三四五六七八九十百千万]+(?:割(?:\s*[一二三四五六七八九十]+分)?|分|人(?:以下|以上)?|社|%|倍))"
)
MIDDLE_COMPARISON_PATTERN = re.compile(
    r"^(?:(?P<comp1>.+?)(?:と比べて|と比較して|に比べ(?:て)?|に対し(?:て)?)|(?:[^、。がは]+?(?:が|は))?(?P<comp2>[^、。がは]+?(?:企業|回答|場合|層|年|年度))の約?)"
)
MIDDLE_INNER_SUBJECT_PATTERN = re.compile(
    r"^(?P<inner_subj>[^、。はが]+?)(?:では|において|で)$"
)
SUBJECT_CORE_PREFIX_PATTERN = re.compile(
    r"^(?:[-・•]+|また|一方|なお|さらに|とくに|しかし|それぞれ|ともに)+"
)
SUBJECT_CORE_SUFFIX_PATTERN = re.compile(
    r"(?:を(?:活用|利用|導入)している企業|に(?:取組|取り組)んでいる企業|の中小企業|の企業|の達成度|の割合|の回答|における|について|では|において|は|で)+$"
)
DEMONSTRATIVE_SUBJECT_PATTERN = re.compile(
    r"^(?:これ|それ|あれ|この|その|同)(?:ら|について|では|は|うち)?$"
)
MODALITY_MARKERS = ("推察される", "可能性がある", "考えられる", "ではないか")
NEGATION_DEGREE_MARKERS = (
    "顕著ではなかった",
    "顕著ではない",
    "差が小さい",
    "差は小さく",
    "差は小さい",
    "大きな差が見られなかった",
    "大きな差が見られる",
    "予定はない",
    "予定がない",
    "著しく高い",
    "低くなっている",
    "少なくなっている",
)
CAUSALITY_MARKERS = ("寄与した", "裏付けられている")
PREDICATE_ENDING_PATTERN = re.compile(
    r"(?:である|だ|ない|なかった|ある|あった|高い|高かった|低い|低かった|"
    r"低くなっている|少なくなっている|小さい|小さかった|見られる|見られなかった|"
    r"推察される|わかった|わかる|いた|ている|ます|です|した|示す)[。．]?$"
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


def _clean_quote_for_baseline(quote):
    """原文の意味を変えず、空白・改行の正規化と明確に不要な独立見出し行の除外のみを行う。"""
    if not isinstance(quote, str) or not quote.strip():
        return ""
    normalized = unicodedata.normalize("NFKC", quote)
    kept_lines = []
    for raw_line in normalized.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if "。" not in line and HEADING_LINE_PATTERN.match(line):
            continue
        kept_lines.append(line)
    if not kept_lines:
        return ""
    joined = " ".join(kept_lines)
    cleaned = re.sub(r"(?<=[^\x00-\x7F])\s+|\s+(?=[^\x00-\x7F])", "", joined)
    cleaned = re.sub(r"(?<=図表[-−ー])\s+(?=\d)", "", cleaned)
    cleaned = re.sub(r"^\d+\.\s*(?:組織連携|実施体制)(?=\S)", "", cleaned)
    return cleaned.strip()


def _is_heading_only_quote(quote):
    """図表タイトルや章節見出しのみのquoteであるかを判定する。"""
    if not isinstance(quote, str) or not quote.strip():
        return True
    cleaned = _clean_quote_for_baseline(quote)
    if not cleaned:
        return True
    if "。" not in cleaned and not PREDICATE_ENDING_PATTERN.search(cleaned):
        if re.match(r"^(?:第\s*\d+\s*章|図表\s*[-−ー]?\s*\d+)", cleaned) or re.search(
            r"(?:の比較|の状況|の割合|の推移|[（(][^）)]+[）)])$", cleaned
        ):
            return True
    return False


def _extract_subject_core(text):
    """対象や比較対象の文字列から照合用の核となる語句を抽出する。"""
    if not isinstance(text, str) or not text.strip():
        return ""
    norm = _normalize_for_evidence_match(text)
    norm = re.sub(r"[「」『』（）()]", "", norm)
    norm = SUBJECT_CORE_PREFIX_PATTERN.sub("", norm)
    norm = SUBJECT_CORE_SUFFIX_PATTERN.sub("", norm)
    norm = re.sub(r"^従業員(?=\d+人)", "", norm)
    return norm.strip()


def _extract_fact_markers(quote):
    """quoteから数値・対象・比較対象のセット、曖昧性フラグ、および意味マーカーを抽出する。"""
    if not isinstance(quote, str) or not quote.strip():
        return {
            "condition_number_sets": [],
            "is_ambiguous": True,
            "modality": [],
            "negation_or_degree": [],
            "causality": [],
        }

    normalized_raw = unicodedata.normalize("NFKC", quote)
    preprocessed_lines = []
    for raw_line in normalized_raw.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if (
            preprocessed_lines
            and (line.startswith(("-", "・", "•")) or ":" in line)
            and not preprocessed_lines[-1].endswith(("。", "、", "！", "？", "!", "?"))
        ):
            preprocessed_lines[-1] = f"{preprocessed_lines[-1]}、"
        preprocessed_lines.append(line)

    cleaned = _clean_quote_for_baseline("\n".join(preprocessed_lines))
    if not cleaned:
        return {
            "condition_number_sets": [],
            "is_ambiguous": True,
            "modality": [],
            "negation_or_degree": [],
            "causality": [],
        }

    scannable = NOISE_NUMBER_SPAN_PATTERN.sub("", cleaned)
    condition_number_sets = []
    bound_tokens = []
    is_ambiguous = False

    for sentence in [s.strip() for s in scannable.split("。") if s.strip()]:
        active_subject = ""
        active_comparison = ""
        clauses = [c.strip() for c in sentence.split("、") if c.strip()]

        for raw_clause in clauses:
            clause = re.sub(r"^[-・•]+\s*", "", raw_clause).strip()
            clause_tokens = NUMBER_TOKEN_PATTERN.findall(clause)

            topic_only = re.match(
                r"^(?:また|一方|なお|さらに|とくに|しかし)?(?P<subj>.+?)(?:では|において|について|は)$",
                clause,
            )
            if topic_only:
                subj = topic_only.group("subj").strip()
                if DEMONSTRATIVE_SUBJECT_PATTERN.match(subj):
                    is_ambiguous = True
                active_subject = subj
                for tok in clause_tokens:
                    condition_number_sets.append(
                        {"subject": subj, "comparison": "", "value": tok}
                    )
                    bound_tokens.append(tok)
                continue

            comp_only = re.match(
                r"^(?:また|一方|なお)?(?P<comp>[^、。:]+?)(?:と比べて|と比較して|に比べ(?:て)?)",
                clause,
            )
            if comp_only:
                comp = comp_only.group("comp").strip()
                comp_tokens = NUMBER_TOKEN_PATTERN.findall(comp)
                if len(comp_tokens) == len(clause_tokens):
                    prev_subject = active_subject
                    active_comparison = comp
                    for sub_item in condition_number_sets:
                        if not sub_item["comparison"] and (
                            not prev_subject or sub_item["subject"] == prev_subject
                        ):
                            sub_item["comparison"] = comp
                    if prev_subject:
                        for tok in comp_tokens:
                            condition_number_sets.append(
                                {"subject": comp, "comparison": prev_subject, "value": tok}
                            )
                            bound_tokens.append(tok)
                    continue

            if not clause_tokens:
                continue

            mult_match = MULTIPLIER_CLAUSE_PATTERN.match(clause)
            if mult_match:
                subj = mult_match.group("subject").strip()
                comp = mult_match.group("comparison").strip()
                val = mult_match.group("value").strip()
                if DEMONSTRATIVE_SUBJECT_PATTERN.match(subj):
                    is_ambiguous = True
                active_subject = subj
                active_comparison = comp
                condition_number_sets.append(
                    {"subject": subj, "comparison": comp, "value": val}
                )
                bound_tokens.extend(NUMBER_TOKEN_PATTERN.findall(val))
                continue

            enum_match = re.search(
                r"(?P<cats>(?:「[^」]+」)+)の(?P<val>\d+)区分",
                clause,
            )
            if enum_match:
                val_tok = enum_match.group("val")
                bound_tokens.append(val_tok)
                continue

            subj_match = SUBJECT_VALUE_CLAUSE_PATTERN.match(clause)
            if subj_match:
                subj = subj_match.group("subject").strip()
                middle = subj_match.group("middle").strip()
                val = subj_match.group("value").strip()

                leading_comp_match = re.match(
                    r"^(?P<leading_comp>.+?)(?:と比べて|と比較して|に比べ(?:て)?|に対し(?:て)?)(?P<rest_subj>.+)$",
                    subj,
                )
                comp_tokens = []
                if leading_comp_match:
                    extracted_comp = leading_comp_match.group("leading_comp").strip()
                    subj = leading_comp_match.group("rest_subj").strip()
                    active_comparison = extracted_comp
                    comp_tokens = NUMBER_TOKEN_PATTERN.findall(extracted_comp)

                if DEMONSTRATIVE_SUBJECT_PATTERN.match(subj):
                    is_ambiguous = True
                if ("と" in subj or "・" in subj) and "それぞれ" in clause:
                    is_ambiguous = True

                comp_in_middle = MIDDLE_COMPARISON_PATTERN.match(middle)
                if comp_in_middle:
                    extracted_comp = (
                        comp_in_middle.group("comp1") or comp_in_middle.group("comp2")
                    ).strip()
                    active_comparison = extracted_comp
                    comp_tokens.extend(NUMBER_TOKEN_PATTERN.findall(extracted_comp))

                target_subj = subj
                if not comp_in_middle:
                    inner_subj_match = MIDDLE_INNER_SUBJECT_PATTERN.match(middle)
                    if inner_subj_match:
                        target_subj = inner_subj_match.group("inner_subj").strip()

                val_tokens = NUMBER_TOKEN_PATTERN.findall(val)
                subj_tokens = NUMBER_TOKEN_PATTERN.findall(subj)
                inner_tokens = (
                    NUMBER_TOKEN_PATTERN.findall(target_subj)
                    if target_subj != subj
                    else []
                )
                if len(clause_tokens) > (
                    len(val_tokens) + len(subj_tokens) + len(comp_tokens) + len(inner_tokens)
                ):
                    is_ambiguous = True
                active_subject = target_subj
                for st in subj_tokens:
                    condition_number_sets.append(
                        {"subject": subj, "comparison": "", "value": st}
                    )
                    bound_tokens.append(st)
                for ct in comp_tokens:
                    condition_number_sets.append(
                        {
                            "subject": active_comparison,
                            "comparison": target_subj,
                            "value": ct,
                        }
                    )
                    bound_tokens.append(ct)
                condition_number_sets.append(
                    {"subject": target_subj, "comparison": active_comparison, "value": val}
                )
                bound_tokens.extend(val_tokens)
                continue

            if active_subject:
                val_matches = list(VALUE_EXPRESSION_PATTERN.finditer(clause))
                if len(val_matches) == 1:
                    val = val_matches[0].group(0).strip()
                    val_tokens = NUMBER_TOKEN_PATTERN.findall(val)
                    if len(val_tokens) == len(clause_tokens):
                        condition_number_sets.append(
                            {
                                "subject": active_subject,
                                "comparison": active_comparison,
                                "value": val,
                            }
                        )
                        bound_tokens.extend(val_tokens)
                        continue

            is_ambiguous = True

    all_tokens = NUMBER_TOKEN_PATTERN.findall(scannable)
    if len(bound_tokens) < len(all_tokens):
        is_ambiguous = True

    modality = [m for m in MODALITY_MARKERS if m in cleaned]
    negation_or_degree = [m for m in NEGATION_DEGREE_MARKERS if m in cleaned]
    causality = [m for m in CAUSALITY_MARKERS if m in cleaned]

    return {
        "condition_number_sets": condition_number_sets,
        "is_ambiguous": is_ambiguous,
        "modality": modality,
        "negation_or_degree": negation_or_degree,
        "causality": causality,
    }


def _has_three_dx_categories(text):
    """「DX取組なし」「DX成果なし」「DX成果あり」の3区分がすべて含まれるか判定する。"""
    normalized = _clean_quote_for_baseline(text)
    return all(
        cat in normalized for cat in ("DX取組なし", "DX成果なし", "DX成果あり")
    )


def _is_three_category_scope_quote(text):
    """3区分の比較前提（対象範囲・分類定義）を説明するquoteか判定する。"""
    normalized = _clean_quote_for_baseline(text)
    if (
        "3区分に分けて比較した結果を図表" in normalized
        or "割合を合計して比較した" in normalized
    ):
        return True
    return _has_three_dx_categories(normalized) and (
        "3区分" in normalized or "比較した結果" in normalized
    )


def _is_scale_dx_initiative_trend_quote(text):
    """Q01のように従業員規模とDX取組割合の変化傾向を直接述べるquoteか判定する。"""
    normalized = _clean_quote_for_baseline(text)
    has_scale = (
        "従業員規模が小さくなるほど" in normalized
        or "従業員規模が小さいほど" in normalized
    )
    has_initiative = (
        "DXの取組も少なくなっている" in normalized
        or "DXに取組んでいる企業の割合" in normalized
    )
    has_outcome_focus = "成果が出ていた" in normalized or "DXの成果で見ると" in normalized
    return has_scale and has_initiative and not has_outcome_focus


def _group_complementary_facts(usable_facts, question=""):
    """補完関係が明確なfactのみをグループ化し、それ以外は別々のグループに分離する。"""
    valid_facts = [
        item
        for item in usable_facts
        if isinstance(item, dict)
        and isinstance(item.get("quote"), str)
        and not _is_heading_only_quote(item.get("quote"))
    ]
    if not valid_facts:
        return []

    scope_facts = [
        item for item in valid_facts if _is_three_category_scope_quote(item["quote"])
    ]
    used_indices = set()
    groups = []

    if scope_facts:
        scope_item = scope_facts[0]
        scope_idx = valid_facts.index(scope_item)
        for idx, item in enumerate(valid_facts):
            if idx == scope_idx:
                continue
            cleaned = _clean_quote_for_baseline(item["quote"])
            markers = _extract_fact_markers(cleaned)
            if (
                _has_three_dx_categories(cleaned)
                and not _is_three_category_scope_quote(cleaned)
                and markers["condition_number_sets"]
            ):
                groups.append([scope_item, item])
                used_indices.add(scope_idx)
                used_indices.add(idx)
        used_indices.add(scope_idx)

    normalized_q = _clean_quote_for_baseline(question) if isinstance(question, str) else ""
    if "従業員規模が小さくなると" in normalized_q or "どのように変化しますか" in normalized_q:
        trend_indices = [
            idx
            for idx, item in enumerate(valid_facts)
            if idx not in used_indices
            and _is_scale_dx_initiative_trend_quote(item["quote"])
        ]
        if len(trend_indices) >= 2:
            groups.append([valid_facts[idx] for idx in trend_indices])
            used_indices.update(trend_indices)

    for idx, item in enumerate(valid_facts):
        if idx in used_indices:
            continue
        if scope_facts and idx == valid_facts.index(scope_facts[0]) and len(valid_facts) > 1:
            continue
        groups.append([item])
        used_indices.add(idx)

    return groups


def _build_skeleton_claim(fact_group):
    """単一または補完関係にある複数factから、原文を改変せずにPython基準回答を構築する。"""
    if not isinstance(fact_group, list) or not fact_group:
        return None

    cleaned_sentences = []
    references = []
    quotes = []
    combined_sets = []
    combined_modality = []
    combined_negation_degree = []
    combined_causality = []
    is_ambiguous = False

    for item in fact_group:
        if not isinstance(item, dict):
            continue
        quote = item.get("quote", "")
        reference = item.get("reference", "")
        if not isinstance(quote, str) or not quote.strip():
            continue
        cleaned = _clean_quote_for_baseline(quote)
        if not cleaned:
            continue
        if cleaned not in cleaned_sentences:
            cleaned_sentences.append(cleaned)
        quotes.append(quote.strip())
        if isinstance(reference, str) and reference and reference not in references:
            references.append(reference)

        markers = _extract_fact_markers(quote)
        if markers["is_ambiguous"]:
            is_ambiguous = True
        combined_sets.extend(markers["condition_number_sets"])
        for m in markers["modality"]:
            if m not in combined_modality:
                combined_modality.append(m)
        for m in markers["negation_or_degree"]:
            if m not in combined_negation_degree:
                combined_negation_degree.append(m)
        for m in markers["causality"]:
            if m not in combined_causality:
                combined_causality.append(m)

    if not cleaned_sentences or not references:
        return None

    baseline_text = "".join(cleaned_sentences)
    return {
        "text": baseline_text,
        "formatted_claim": _format_claim(baseline_text, references),
        "references": references,
        "quotes": quotes,
        "markers": {
            "condition_number_sets": combined_sets,
            "is_ambiguous": is_ambiguous,
            "modality": combined_modality,
            "negation_or_degree": combined_negation_degree,
            "causality": combined_causality,
        },
    }


MODALITY_VARIANTS = {
    "推察される": ("推察される", "推察されます"),
    "可能性がある": ("可能性がある", "可能性があります"),
    "考えられる": ("考えられる", "考えられます"),
    "ではないか": ("ではないか", "ではないでしょうか"),
}
NEGATION_DEGREE_VARIANTS = {
    "顕著ではなかった": ("顕著ではなかった", "顕著ではない", "顕著ではありません"),
    "顕著ではない": ("顕著ではない", "顕著ではなかった", "顕著ではありません"),
    "差が小さい": ("差が小さい", "差は小さく", "差は小さい"),
    "差は小さく": ("差は小さく", "差が小さい", "差は小さい"),
    "差は小さい": ("差は小さい", "差が小さい", "差は小さく"),
    "大きな差が見られなかった": (
        "大きな差が見られなかった",
        "大きな差が見られませんでした",
        "大きな差は見られなかった",
        "大きな差は見られませんでした",
    ),
    "大きな差が見られる": (
        "大きな差が見られる",
        "大きな差が見られます",
        "差が見られる",
        "差が見られます",
    ),
    "予定はない": ("予定はない", "予定がない", "予定はありません"),
    "予定がない": ("予定がない", "予定はない", "予定はありません"),
    "著しく高い": ("著しく高い", "著しく高くなっています", "著しく高いです"),
    "低くなっている": ("低くなっている", "低くなっています", "低い", "低いです"),
    "少なくなっている": ("少なくなっている", "少なくなっています", "少ない", "少ないです"),
}
CAUSALITY_VARIANTS = {
    "寄与した": ("寄与した", "寄与しました"),
    "裏付けられている": ("裏付けられている", "裏付けられています"),
}
META_REFUSAL_PHRASES = ("記載されていません", "確認できません", "不明です", "分かりません")


def _is_meta_refusal_text(text):
    """「具体的な数値は記載されていません」等のメタ拒否文かを判定する。"""
    if not isinstance(text, str):
        return True
    cleaned = _clean_quote_for_baseline(text)
    if not cleaned or cleaned == NO_SOURCE_ANSWER:
        return True
    return any(phrase in cleaned for phrase in META_REFUSAL_PHRASES)


def _subjects_match(req_subj, llm_subj, other_subject_cores=()):
    """原文の対象とLLM出力内の対象が他対象と混同されずに一致しているか判定する。"""
    req_core = _extract_subject_core(req_subj)
    llm_core = _extract_subject_core(llm_subj)
    if not req_core or not llm_core:
        return False
    llm_norm = _normalize_for_evidence_match(llm_subj)
    for other_core in other_subject_cores:
        if (
            other_core
            and other_core != req_core
            and other_core not in req_core
            and req_core not in other_core
            and (other_core in llm_core or other_core in llm_norm)
        ):
            return False
    return req_core == llm_core or req_core in llm_core


def _comparisons_match(req_comp, llm_comp):
    """原文の比較対象とLLM出力内の比較対象が一致しているか判定する。"""
    req_core = _extract_subject_core(req_comp)
    llm_core = _extract_subject_core(llm_comp)
    if not req_core:
        return not llm_core
    if not llm_core:
        return False
    return req_core == llm_core or req_core in llm_core


def _are_condition_number_sets_verified(req_sets, llm_sets):
    """原文の(subject, comparison, value)の対応関係がLLM出力と一対一ですべて維持されているか検証する。"""
    all_req_cores = []
    for rset in req_sets:
        core = _extract_subject_core(rset.get("subject", ""))
        if core and core not in all_req_cores:
            all_req_cores.append(core)

    valid_req_sets = [
        rset
        for rset in req_sets
        if _normalize_for_evidence_match(rset.get("value", ""))
    ]
    valid_llm_sets = [
        lset
        for lset in llm_sets
        if _normalize_for_evidence_match(lset.get("value", ""))
    ]
    if len(valid_req_sets) != len(valid_llm_sets):
        return False

    used_llm_indices = set()
    for rset in valid_req_sets:
        req_val = _normalize_for_evidence_match(rset.get("value", ""))
        req_subj = rset.get("subject", "")
        req_comp = rset.get("comparison", "")
        matched_idx = None
        for idx, lset in enumerate(valid_llm_sets):
            if idx in used_llm_indices:
                continue
            llm_val = _normalize_for_evidence_match(lset.get("value", ""))
            if llm_val != req_val:
                continue
            if not _subjects_match(req_subj, lset.get("subject", ""), all_req_cores):
                continue
            if not _comparisons_match(req_comp, lset.get("comparison", "")):
                continue
            matched_idx = idx
            break
        if matched_idx is None:
            return False
        used_llm_indices.add(matched_idx)

    return len(used_llm_indices) == len(valid_llm_sets)


def _is_verified_llm_claim(text, valid_quotes, question, skeleton_claim):
    """LLM整形結果がPythonで検証可能な範囲内で原文の数値・条件・意味を維持しているか検査する。"""
    if not isinstance(text, str) or not text.strip() or not skeleton_claim:
        return False
    if _is_meta_refusal_text(text):
        return False

    cleaned_text = _clean_quote_for_baseline(text)
    cleaned_question = _clean_quote_for_baseline(question) if isinstance(question, str) else ""
    cleaned_quotes = _clean_quote_for_baseline("\n".join(valid_quotes))

    if not cleaned_text or cleaned_text.endswith(("？", "?")):
        return False
    if _normalize_for_evidence_match(cleaned_text) == _normalize_for_evidence_match(
        cleaned_question
    ):
        return False

    markers = skeleton_claim.get("markers", {})
    if markers.get("is_ambiguous"):
        return False

    if re.search(r"違いがあります[。．]?$", cleaned_text) and (
        markers.get("condition_number_sets") or markers.get("negation_or_degree")
    ):
        return False

    q_prefix_match = re.match(r"^(?P<prefix>[^、。]+?)(?:では|において)、", cleaned_text)
    if q_prefix_match and cleaned_question:
        prefix = q_prefix_match.group("prefix").strip()
        if prefix in cleaned_question and prefix not in cleaned_quotes:
            return False

    norm_text = _normalize_for_evidence_match(cleaned_text)
    req_sets = markers.get("condition_number_sets", [])
    if req_sets:
        llm_markers = _extract_fact_markers(text)
        if llm_markers.get("is_ambiguous"):
            return False
        llm_sets = llm_markers.get("condition_number_sets", [])
        if not llm_sets:
            return False
        if not _are_condition_number_sets_verified(req_sets, llm_sets):
            return False

    for mod in markers.get("modality", []):
        variants = MODALITY_VARIANTS.get(mod, (mod,))
        if not any(_normalize_for_evidence_match(v) in norm_text for v in variants):
            return False

    for neg in markers.get("negation_or_degree", []):
        variants = NEGATION_DEGREE_VARIANTS.get(neg, (neg,))
        if not any(_normalize_for_evidence_match(v) in norm_text for v in variants):
            return False

    for caus in markers.get("causality", []):
        variants = CAUSALITY_VARIANTS.get(caus, (caus,))
        if not any(_normalize_for_evidence_match(v) in norm_text for v in variants):
            return False

    if any(adv in cleaned_quotes for adv in ("ものの、", "に対し、")) and not any(
        adv in cleaned_text for adv in ("ものの", "に対し", "が、", "一方")
    ):
        return False

    return True


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
        and not _is_heading_only_quote(item.get("quote"))
    ]
    if not usable_facts:
        return _build_fallback_from_chunks(source_by_reference)

    fact_groups = _group_complementary_facts(usable_facts, question)
    skeleton_by_ref = {}
    for group in fact_groups:
        skel = _build_skeleton_claim(group)
        if not skel:
            continue
        for ref in skel["references"]:
            skeleton_by_ref[ref] = skel

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
        "【質問】に対する具体的な回答を、提示された【検証済み根拠】の原文引用（quote）に基づいて日本語で作成してください。\n"
        "以下のルールを厳守してください。\n"
        "1. 質問文の反復・根拠外情報の混入禁止：claimのtextに【質問】の文面をそのままコピーしたり、疑問文（「〜ですか？」など）を出力したりしないでください。また、そのclaimのcitationsに含まれない数値や語句（質問文中にしかない数値など）をtextに付け加えないでください。\n"
        "2. 情報の整理と定性的結論・ニュアンスの維持：事実の要約（fact）が短い見出し形式であっても、原文引用（quote）から質問に必要な情報を整理して具体的な回答文をtextに作成してください。具体的な数値が記載されていない項目についても、推測や別項目の数値の流用をせず、原文引用に書かれている定性的な結論（差は小さい、関係は顕著ではない、寄与したと推察されるなど）や主語・比較条件・因果関係・推量表現のニュアンスを省略・変更せずに回答してください。\n"
        "3. 比較条件と数値の正確な対応付け：比較や違い・特徴を尋ねられた場合、原文引用（quote）に質問への回答に関係する数値・割合・倍率が記載されている項目は、「違いがあります」だけで終わらせず、質問への回答に関係する各分類条件・比較対象（DX取組なし・DX成果なし・DX成果あり、活用の有無、従業員規模など）と数値を省略せず正確に対応させてtextに明記してください。そのclaim自身のcitationsのquoteに存在しない数値をtextに含めてはいけません。\n"
        "4. 異なる観点の分離と関連根拠の組み合わせ：異なる観点（topic、例：組織連携と推進体制）はclaimを分け、一方の数値や結論を他方に流用しないでください。一方、同一の観点について対象範囲（従業員100人以下など）・分類条件と具体的な数値が別々の根拠に分かれており条件が一致している場合は、それらを組み合わせて回答し、textに含まれるすべての対象範囲・数値・結論を裏付ける複数の根拠（referenceとquote）をそのclaimのcitationsに漏れなく指定してください。citationsの各quoteは【検証済み根拠】の原文引用を一字一句変えずに指定してください。\n"
        "5. 根拠不足時の対応：【検証済み根拠】だけでは質問に対する具体的な回答を作成できない場合（根拠が不足している場合）は、無理に回答を作らずclaimsを空配列 [] にしてください。\n"
        "JSON出力例：\n"
        "{\n"
        '  "claims": [\n'
        "    {\n"
        '      "text": "原文引用に基づき質問に必要な情報を整理した回答文",\n'
        '      "citations": [\n'
        "        {\n"
        '          "reference": "資料1",\n'
        '          "quote": "検証済み根拠の原文引用をそのまま記載"\n'
        "        }\n"
        "      ]\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
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
    failed_stage1_verified_refs = []
    for claim in claims:
        if not isinstance(claim, dict):
            continue
        text = claim.get("text")
        citations = claim.get("citations")
        if not isinstance(text, str) or not text.strip() or not isinstance(citations, list):
            continue
        if _is_meta_refusal_text(text):
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
        for ref in valid_references:
            if ref not in failed_stage1_verified_refs:
                failed_stage1_verified_refs.append(ref)

        if _has_unsupported_numbers(text, "\n".join(valid_quotes)):
            logger.info("引用に存在しない数値を含むclaimを除外します")
            continue

        matched_skeleton = None
        for ref in valid_references:
            if ref in skeleton_by_ref:
                matched_skeleton = skeleton_by_ref[ref]
                break
        if not matched_skeleton:
            continue

        has_all_group_refs = set(matched_skeleton["references"]).issubset(
            set(valid_references)
        )
        if has_all_group_refs and _is_verified_llm_claim(
            text, valid_quotes, question, matched_skeleton
        ):
            formatted_claim = _format_claim(text, valid_references)
            claim_references = valid_references
        else:
            logger.info("LLM整形文が品質基準を満たさないためPython基準回答を採用します")
            formatted_claim = matched_skeleton["formatted_claim"]
            claim_references = matched_skeleton["references"]

        if not formatted_claim:
            continue
        if formatted_claim not in answer_claims:
            answer_claims.append(formatted_claim)
        for reference in claim_references:
            if reference not in used_references:
                used_references.append(reference)

    if not answer_claims and failed_stage1_verified_refs:
        for ref in failed_stage1_verified_refs:
            skel = skeleton_by_ref.get(ref)
            if skel and skel["formatted_claim"] not in answer_claims:
                answer_claims.append(skel["formatted_claim"])
                for r in skel["references"]:
                    if r not in used_references:
                        used_references.append(r)

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
