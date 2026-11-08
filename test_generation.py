from app.generation.generate_answer import generate_answer
from app.retrieval.search_chunks import search_chunks


def main():
    question = "従業員100人以下の中小企業では、DXに取り組む予定がない企業はどのくらいありますか？"
    chunks = search_chunks(question, limit=5)

    if not chunks:
        print("検索結果が0件のため、回答生成を行いませんでした。")
        return

    print("\n===== 検索されたチャンク一覧 =====")
    for i, chunk in enumerate(chunks, start=1):
        print(f"\n--- 資料{i} ---")
        print(f"ページ番号: {chunk['page_number']}")
        print(chunk["content"])

    result = generate_answer(question, chunks)
    print("回答:")
    print(result["answer"])
    print("\n出典一覧:")
    for source in result["sources"]:
        chunk_index = int(source["reference"][2:]) - 1
        print(
            f"{source['reference']}: 文書ID={source['document_id']}, "
            f"ページ番号={source['page_number']}\n"
            f"本文: {chunks[chunk_index]['content']}"
        )


if __name__ == "__main__":
    main()
