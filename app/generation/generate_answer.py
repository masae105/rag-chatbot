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
        "text": {"type": "string"},
    },
    "required": ["text"],
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


def _is_only_figure_or_analysis_intro(quote):
    """図表の案内・分析目的だけで、実質的な結果を述べていないquoteか判定する。"""
    cleaned = _clean_quote_for_baseline(quote)
    if not cleaned or _is_three_category_scope_quote(cleaned):
        return False

    has_result = re.search(
        r"差.{0,8}(?:小|大)|上回|下回|関係は|傾向|増加|減少|"
        r"(?:割合|回答).{0,8}(?:高い|低い|多い|少ない)|顕著|大きな差|小さな差",
        cleaned,
    )
    is_figure_intro = bool(
        re.search(r"^(?:図表|グラフ|表)", cleaned)
        and re.search(
            r"(?:見たものである|示したものである|まとめたものである)[。．]?$",
            cleaned,
        )
    ) or bool(
        re.search(r"(?:図表|グラフ|表)", cleaned)
        and re.search(
            r"(?:を.{0,12}(?:示す|示した|掲載する|掲載した|まとめる|まとめた|"
            r"掲げる|掲げた|表す|表した)|"
            r"(?:に|では).{0,12}(?:示す|示した|掲載する|掲載した|まとめる|まとめた|"
            r"掲げる|掲げた|表す|表した))",
            cleaned,
        )
    )
    is_analysis_purpose = bool(
        re.search(
            r"(?:分析|検討|把握)するため|(?:分析|検討|把握)を行った|"
            r"(?:分析|検討|把握)することを目的",
            cleaned,
        )
    )
    return (is_figure_intro or is_analysis_purpose) and not has_result


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

    has_substantive_fact = any(
        not _is_only_figure_or_analysis_intro(item["quote"]) for item in valid_facts
    )
    for idx, item in enumerate(valid_facts):
        if idx in used_indices:
            continue
        if (
            has_substantive_fact
            and len(valid_facts) > 1
            and _is_only_figure_or_analysis_intro(item["quote"])
        ):
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
    topics = []
    facts = []
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
        topic = item.get("topic")
        fact = item.get("fact")
        if isinstance(topic, str) and topic.strip() and topic.strip() not in topics:
            topics.append(topic.strip())
        if isinstance(fact, str) and fact.strip() and fact.strip() not in facts:
            facts.append(fact.strip())
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
        "topics": topics,
        "facts": facts,
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


def _is_question_and_evidence_composite_term(term, question, evidence):
    """質問の修飾語と根拠中の語を組み合わせた用語か判定する。"""
    normalized_term = _normalize_for_evidence_match(term)
    normalized_question = _normalize_for_evidence_match(question)
    normalized_evidence = _normalize_for_evidence_match(evidence)
    for split_at in range(2, len(normalized_term) - 1):
        if (
            normalized_term[:split_at] in normalized_question
            and normalized_term[split_at:] in normalized_evidence
        ) or (
            normalized_term[:split_at] in normalized_evidence
            and normalized_term[split_at:] in normalized_question
        ):
            return True
    return False


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
    if _has_unsupported_numbers(text, "\n".join(valid_quotes)):
        return False
    req_sets = markers.get("condition_number_sets", [])
    condition_sets_verified = not req_sets
    if req_sets:
        llm_markers = _extract_fact_markers(text)
        if llm_markers.get("is_ambiguous"):
            return False
        llm_sets = llm_markers.get("condition_number_sets", [])
        if not llm_sets:
            return False
        if not _are_condition_number_sets_verified(req_sets, llm_sets):
            return False
        condition_sets_verified = True

    evidence_context = "\n".join(
        [cleaned_quotes]
        + skeleton_claim.get("topics", [])
        + skeleton_claim.get("facts", [])
    )
    normalized_evidence_context = _normalize_for_evidence_match(evidence_context)
    for term in _extract_content_terms(cleaned_text):
        if term in normalized_evidence_context:
            continue
        if term == "比較" and condition_sets_verified and req_sets:
            continue
        if _is_question_and_evidence_composite_term(
            term, cleaned_question, evidence_context
        ):
            continue
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

    comparison_conclusions = (
        (
            "全体平均を上回",
            ("全体平均を上回", "企業全体平均を上回", "全体平均より高", "企業全体平均より高"),
        ),
        (
            "全体の平均を上回",
            ("全体の平均を上回", "全体平均を上回", "平均より高"),
        ),
        ("平均を上回", ("平均を上回", "平均より高")),
        ("全体平均を下回", ("全体平均を下回", "企業全体平均を下回", "全体平均より低", "企業全体平均より低")),
        (
            "全体の平均を下回",
            ("全体の平均を下回", "全体平均を下回", "平均より低"),
        ),
        ("平均を下回", ("平均を下回", "平均より低")),
    )
    for marker, variants in comparison_conclusions:
        if marker in cleaned_quotes and not any(variant in cleaned_text for variant in variants):
            return False

    if any(adv in cleaned_quotes for adv in ("ものの、", "に対し、")) and not any(
        adv in cleaned_text for adv in ("ものの", "に対し", "が、", "一方")
    ):
        return False

    return True


def _extract_content_terms(text):
    """テキストから照合用の内容語（漢字2文字以上・カタカナ2文字以上・英数字2文字以上）を抽出する。"""
    if not isinstance(text, str) or not text.strip():
        return []
    cleaned = re.sub(r"\[?資料\d+\]?", " ", text)
    terms = re.findall(r"[一-龥]{2,}|[ァ-ヴー]{2,}|[A-Za-z0-9]{2,}", cleaned)
    stopwords = {"資料", "企業", "場合", "割合", "状況", "結果", "関係", "違い", "について", "それぞれ"}
    seen = []
    for t in terms:
        nt = _normalize_for_evidence_match(t)
        if nt and t not in stopwords and nt not in seen:
            seen.append(nt)
    return seen


def _candidate_sentences_from_chunk(chunk_text):
    """チャンク原文から、図表・節見出しを除いた完全文（および逆接後半の独立節）の候補リストを返す。"""
    if not isinstance(chunk_text, str) or not chunk_text.strip():
        return []
    candidates = []
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line:
            continue
        for sent in re.split(r"(?<=[。！？!?])", line):
            sent = sent.strip()
            if not sent or _is_heading_only_quote(sent):
                continue
            if not (sent.endswith("。") or PREDICATE_ENDING_PATTERN.search(sent)):
                continue
            if sent not in candidates:
                candidates.append(sent)
            if "が、" in sent:
                _, tail = sent.split("が、", 1)
                tail = tail.strip()
                if tail and len(tail) >= 15 and tail not in candidates:
                    candidates.append(tail)
    return candidates


def _resolve_verbatim_quote(
    reference, topic, quote, fact, source_by_reference, question=""
):
    """Stage1の出力から、検証条件を満たす場合のみチャンク内の正確な原文文（(ref, quote)）を特定する。"""
    raw_refs = re.findall(r"資料\d+", f"{reference or ''} {quote or ''}")
    ref_candidates = []
    for r in raw_refs:
        if r in source_by_reference and r not in ref_candidates:
            ref_candidates.append(r)
    if not ref_candidates:
        return reference, ""

    cleaned_quote = _clean_quote_for_baseline(quote) if isinstance(quote, str) else ""
    norm_quote = _normalize_for_evidence_match(cleaned_quote)
    norm_quote_loose = re.sub(r"(?<=」)[、，](?=「)", "", norm_quote)
    has_fact = isinstance(fact, str) and bool(fact.strip())
    cleaned_fact = _clean_quote_for_baseline(fact) if has_fact else ""
    fact_neg_markers = [
        m for m in NEGATION_DEGREE_MARKERS if m in cleaned_fact
    ]

    question_terms = _extract_content_terms(question)
    topic_terms = _extract_content_terms(topic)
    fact_terms = _extract_content_terms(fact)
    quote_terms = _extract_content_terms(cleaned_quote)

    # 段階A: LLMのquote内にチャンクの原文候補文が正確に含まれている場合（連続する複数文も保持）
    if norm_quote_loose:
        for ref in ref_candidates:
            chunk_text = source_by_reference[ref].get("content", "")
            full_sents = [
                cand
                for cand in _candidate_sentences_from_chunk(chunk_text)
                if _normalize_for_evidence_match(cand) in norm_quote_loose
                or re.sub(r"(?<=」)[、，](?=「)", "", _normalize_for_evidence_match(cand))
                in norm_quote_loose
            ]
            filtered_sents = []
            for s in full_sents:
                ns = _normalize_for_evidence_match(s)
                if any(
                    ns != _normalize_for_evidence_match(other)
                    and ns in _normalize_for_evidence_match(other)
                    for other in full_sents
                ):
                    continue
                qh = sum(1 for t in question_terms if t in ns)
                th = sum(1 for t in topic_terms if t in ns)
                fh = sum(1 for t in fact_terms if t in ns)
                if qh < 1 and (th + fh) < 2:
                    continue
                if (qh + th + fh) < 2:
                    continue
                markers = _extract_fact_markers(s)
                has_substance = bool(
                    markers["condition_number_sets"]
                    or markers["negation_or_degree"]
                    or markers["modality"]
                    or markers["causality"]
                    or _is_scale_dx_initiative_trend_quote(s)
                    or (fact_terms and (fh / len(fact_terms)) >= 0.7)
                )
                if not has_substance:
                    continue
                filtered_sents.append(s)

            if filtered_sents:
                combined = "".join(filtered_sents)
                if _is_valid_citation(ref, combined, source_by_reference):
                    if not (has_fact and _has_unsupported_numbers(fact, combined)):
                        norm_comb = _normalize_for_evidence_match(combined)
                        neg_ok = not any(
                            not any(
                                _normalize_for_evidence_match(v) in norm_comb
                                for v in NEGATION_DEGREE_VARIANTS.get(neg, (neg,))
                            )
                            for neg in fact_neg_markers
                        )
                        if neg_ok:
                            return ref, combined

    # 段階B: LLMがquoteを要約・省略した場合、fact_terms（70%以上かつ2語以上）でチャンク内の単一原文文を特定
    best_match = None
    best_score = None

    for ref in ref_candidates:
        chunk_text = source_by_reference[ref].get("content", "")
        for cand in _candidate_sentences_from_chunk(chunk_text):
            if not _is_valid_citation(ref, cand, source_by_reference):
                continue

            # 1. factに数値がある場合、その数値がすべて候補文に存在することを必須とする
            if has_fact and _has_unsupported_numbers(fact, cand):
                continue

            norm_cand = _normalize_for_evidence_match(cand)

            # 1b. factに否定・程度・結論表現がある場合、候補文にも同義表現が存在することを必須とする
            if any(
                not any(
                    _normalize_for_evidence_match(v) in norm_cand
                    for v in NEGATION_DEGREE_VARIANTS.get(neg, (neg,))
                )
                for neg in fact_neg_markers
            ):
                continue

            # 2. topicとfactの主要な内容語が候補文で裏付けられることを確認する
            topic_hits = sum(1 for t in topic_terms if t in norm_cand)
            fact_hits = sum(1 for t in fact_terms if t in norm_cand)
            quote_hits = sum(1 for t in quote_terms if t in norm_cand)

            # factがある場合はfactの主要語が70%以上、かつ2語以上（1語のみなら1語）一致すること
            if fact_terms:
                min_fact_hits = 1 if len(fact_terms) == 1 else 2
                if fact_hits < min_fact_hits or (fact_hits / len(fact_terms)) < 0.7:
                    continue
            # topicとfact/quoteの合計でも裏付けが不十分な場合は不採用
            if (topic_hits + fact_hits) < 2 and (topic_hits + quote_hits) < 2:
                continue

            # 3. 必須条件を満たした候補の中で一致スコアを算出する（同点時は短い候補を優先）
            score = (fact_hits, topic_hits, -len(norm_cand))

            if best_score is None or score > best_score:
                best_score = score
                best_match = (ref, cand)

    if best_match is not None:
        return best_match
    return reference, ""


def _extract_verified_facts(question, context_parts, source_by_reference):
    """第1段階：質問に関係する事実と引用を抽出し、Python側で検証して返す。"""
    prompt = (
        "【質問】に答えるために必要な事実と原文の引用を、提示された【資料本文】から抽出してください。\n"
        "質問文と各資料の内容を照らし合わせ、質問に直接関係する以下の観点を必要に応じて別々の要素として漏れなく抽出してください。\n"
        "1. 対象範囲・分類条件・比較の前提（調査対象の規模や比較区分の説明文など）\n"
        "2. 具体的な数値・割合・比較結果（質問で問われている項目の数値・割合・倍率や比較対象との差など）\n"
        "3. 定性的な結論・傾向（差が大きい・差は小さい・関係は顕著ではない・平均を上回るなどの結論や傾向）\n"
        "質問に複数の観点や比較項目が含まれる場合は、質問を観点ごとに分解し、それぞれについて独立したFactを抽出してください。一つの観点を抽出できても、ほかの観点の抽出を省略してはいけません。\n"
        "資料本文全体を確認し、関連情報が複数のチャンクに分かれている場合は、それぞれのチャンクから必要なFactを抽出してください。\n"
        "各Factのtopic、quote、factは必ず同じ観点の内容にしてください。異なる観点の情報や数値を一つのFactに混在させないでください。\n"
        "数値や割合をfactに記載する場合は、その数値が実際に記載されている原文をquoteとして選択してください。数値の根拠が書かれていないチャンクを引用元にせず、比較条件や対象範囲が別のチャンクにある場合は必要に応じて別々のFactとして抽出してください。\n"
        "「関係は顕著ではなかった」「実施割合が高かった」「明確な差が見られなかった」など、数値がなくても質問に関係する定性的な結論や傾向を抽出してください。\n"
        "quoteには、資料本文に存在する意味の完結した文を文頭から文末まで一字一句そのまま引用してください。文の途中から引用を開始したり、文の途中で終了したりしないでください。\n"
        "複数の文が必要な場合は、質問に必要な各文をそれぞれ原文どおりに含めてください。\n"
        "quoteに含まれる数値、割合、単位、記号、表記は原文のまま維持してください。例えば、原文の「2割」を「20%」に変換してはいけません。factでも原文に記載された数値や単位の表現を維持し、独自の数値変換を行わないでください。\n"
        "質問に答えるために必要な結論、条件、比較対象、数値などがquoteに含まれるようにしてください。長い文章から引用する場合も必要な情報を省略せず、質問と関係のない説明まで無理に引用する必要はありません。\n"
        "図表タイトルや節見出しだけを根拠（quote）として抽出しないでください。\n"
        "数値を問われた場合は具体的な数値が書かれている本文を、定性的な違いや特徴・結論を問われた場合はその内容が書かれている本文をquoteとして抽出してください。\n"
        "各要素では、まず質問に応じた観点（topic）を指定し、次に根拠となる本文（quote）を資料本文から文頭から文末まで一字一句そのまま抜き出し、そのquoteが含まれる資料番号（reference）を必ず「資料1」のように角括弧なしで指定し、最後にquoteに基づく事実（fact）を記述してください。\n"
        "quoteに書かれていない数値や別の観点の数値をfactに含めてはいけません。\n"
        "JSON出力例：\n"
        "{\n"
        '  "facts": [\n'
        "    {\n"
        '      "topic": "質問の観点",\n'
        '      "quote": "資料本文からの原文引用",\n'
        '      "reference": "該当する資料番号",\n'
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

    logger.info("LLMが返した生のFact件数: %d", len(facts))
    verified_facts = []
    for index, item in enumerate(facts, start=1):
        if not isinstance(item, dict):
            logger.info(
                "生Fact %dをスキップ: 要素が辞書ではありません (value=%r)",
                index,
                item,
            )
            continue
        logger.info(
            "生Fact %d: topic=%r, reference=%r, quote=%r, fact=%r",
            index,
            item.get("topic"),
            item.get("reference"),
            item.get("quote"),
            item.get("fact"),
        )
        logger.info("生Fact %d: 要素形式の検証を通過", index)
        topic = item.get("topic")
        reference = item.get("reference")
        quote = item.get("quote")
        fact = item.get("fact")
        if not isinstance(topic, str) or not topic.strip():
            logger.info("生Fact %dをスキップ: topicが空、または文字列ではありません", index)
            continue
        logger.info("生Fact %d: topicの検証を通過", index)
        reference, quote = _resolve_verbatim_quote(
            reference, topic, quote, fact, source_by_reference, question
        )
        logger.info(
            "生Fact %d: 引用補正後 reference=%r, quote=%r",
            index,
            reference,
            quote,
        )
        if not _is_valid_citation(reference, quote, source_by_reference):
            logger.info(
                "生Fact %dをスキップ: referenceまたはquoteが資料本文の引用として無効です",
                index,
            )
            continue
        logger.info("生Fact %d: reference/quoteの検証を通過", index)

        verified_fact = None
        if isinstance(fact, str) and fact.strip():
            if _has_unsupported_numbers(fact, quote):
                logger.info(
                    "生Fact %d: factに引用にない数値があるためfact=Noneとし、Factレコードは保持します",
                    index,
                )
            else:
                verified_fact = fact.strip()
                logger.info("生Fact %d: factの数値チェックを通過", index)
        elif isinstance(fact, str):
            logger.info("生Fact %d: factが空のためfact=Noneとして保持します", index)
        else:
            logger.info(
                "生Fact %d: factが文字列ではないためfact=Noneとして保持します",
                index,
            )

        verified_facts.append(
            {
                "topic": topic.strip(),
                "reference": reference,
                "quote": quote.strip(),
                "fact": verified_fact,
            }
        )
        logger.info("生Fact %d: 検証済みFactとして追加 (fact=%r)", index, verified_fact)

    logger.info("検証後のFact件数: %d", len(verified_facts))
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
    """第2段階：検証済みFactを回答文に整え、出典はPython側で付与する。"""
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
    skeletons = []
    for group in fact_groups:
        skel = _build_skeleton_claim(group)
        if skel:
            skeletons.append((skel, group))

    answer_claims = []
    used_references = []
    for skel, group in skeletons:
        evidence_parts = [
            f"原文引用: {item['quote']}\n事実: {item['fact']}" for item in group
        ]
        prompt = (
            "【質問】に対する回答文を、提示された1つの検証済みFactグループだけに基づいて"
            "自然な日本語で1つ作成してください。\n"
            "引用や出典情報は出力せず、JSONのtextだけを返してください。"
            "根拠にない情報・数値を追加せず、対象と数値の対応、否定・程度・推量・因果・"
            "逆接を含む重要な意味や結論を省略・変更しないでください。\n"
            '出力形式: {"text": "回答文"}\n\n'
            "【検証済みFactグループ】\n"
            + "\n\n".join(evidence_parts)
            + f"\n\n【質問】\n{question.strip()}"
        )

        formatted_claim = skel["formatted_claim"]
        try:
            raw_response = _call_ollama(prompt, ANSWER_RESPONSE_SCHEMA)
        except RuntimeError:
            logger.warning("第2段階の応答を取得できないためPython基準回答を採用します。")
        else:
            try:
                response = json.loads(raw_response)
            except json.JSONDecodeError:
                logger.warning("第2段階の応答がJSONではないためPython基準回答を採用します。")
            else:
                text = response.get("text") if isinstance(response, dict) else None
                if isinstance(text, str) and _is_verified_llm_claim(
                    text, skel["quotes"], question, skel
                ):
                    formatted_claim = _format_claim(text, skel["references"])
                else:
                    logger.info(
                        "LLM整形文が品質基準を満たさないためPython基準回答を採用します"
                    )

        if not formatted_claim:
            continue
        if formatted_claim not in answer_claims:
            answer_claims.append(formatted_claim)
        for reference in skel["references"]:
            if reference not in used_references:
                used_references.append(reference)

    if not answer_claims:
        return {"answer": NO_SOURCE_ANSWER, "sources": []}

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
        logger.info("Stage1に渡すcontext_parts件数: %d", len(context_parts))
        for index, (context_part, source) in enumerate(
            zip(context_parts, source_by_reference.values()), start=1
        ):
            logger.info(
                "Stage1 context_part %d: document_id=%s, page_number=%s, "
                "chunk_number=%s, content=%s",
                index,
                source.get("document_id"),
                source.get("page_number"),
                source.get("chunk_number"),
                context_part,
            )
        verified_facts = _extract_verified_facts(
            question, context_parts, source_by_reference
        )
        logger.info("Stage1で検証されたFact件数: %d", len(verified_facts))
        for index, item in enumerate(verified_facts, start=1):
            logger.info(
                "Stage1検証済みFact %d: topic=%s, reference=%s, fact=%s, quote=%s",
                index,
                item.get("topic"),
                item.get("reference"),
                item.get("fact"),
                item.get("quote"),
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
