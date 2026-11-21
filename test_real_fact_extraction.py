from dotenv import load_dotenv

from app.generation.generate_answer import (
    _extract_verified_facts,
    generate_answer,
)
from app.retrieval.search_chunks import search_chunks


QUESTIONS = [
    (
        "Q04",
        "従業員規模によって、DXの成果が出ている割合にはどのような違いがありますか？",
    ),
    (
        "Q05",
        "従業員100人以下の企業では、DXの成果の有無によって組織連携の実施割合や推進体制にどのような違いがありますか？",
    ),
]


def build_context_and_sources(chunks):
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
    return context_parts, source_by_reference


def main():
    load_dotenv()

    for qid, question in QUESTIONS:
        print(f"\n==================== {qid} ====================")
        print(f"質問: {question}")

        try:
            chunks = search_chunks(question, limit=10)
        except Exception as error:
            print(f"検索に失敗しました: {error}")
            continue

        if not chunks:
            print("検索結果が0件でした。")
            continue

        print(f"検索結果: {len(chunks)}件")
        for rank, chunk in enumerate(chunks, start=1):
            print(
                f"  順位{rank} (資料{rank}): "
                f"ページ番号={chunk['page_number']}, "
                f"チャンク番号={chunk['chunk_number']}, "
                f"類似度={chunk['similarity']}"
            )

        context_parts, source_by_reference = build_context_and_sources(chunks)

        try:
            verified_facts = _extract_verified_facts(
                question, context_parts, source_by_reference
            )
        except Exception as error:
            print(f"Stage1事実抽出に失敗しました: {error}")
            continue

        print(f"\n--- Stage1の抽出結果 ({len(verified_facts)}件) ---")
        for idx, item in enumerate(verified_facts, start=1):
            print(f"[Fact {idx}]")
            print(f"  topic: {item.get('topic')}")
            print(f"  fact: {item.get('fact')}")
            print(f"  引用補正後のreference: {item.get('reference')}")
            print(f"  引用補正後のquote: {item.get('quote')}")

        try:
            result = generate_answer(question, chunks)
        except Exception as error:
            print(f"回答生成に失敗しました: {error}")
            continue

        print(f"\n--- 最終的な generate_answer() の回答 ---")
        print(f"回答:\n{result['answer']}")
        print("出典:")
        for source in result["sources"]:
            print(
                f"- {source['reference']}: {source['document_id']} / "
                f"ページ{source['page_number']} / "
                f"チャンク{source.get('chunk_number')}"
            )


if __name__ == "__main__":
    main()
