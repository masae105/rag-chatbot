from app.retrieval.search_chunks import search_chunks


def main():
    question = "従業員100人以下の中小企業では、DX推進にどのような課題がありますか？"
    results = search_chunks(question, limit=10)

    if not results:
        print("検索結果はありませんでした。")
        return

    for rank, result in enumerate(results, start=1):
        print(f"検索順位: {rank}")
        print(f"ページ番号: {result['page_number']}")
        print(f"チャンク番号: {result['chunk_number']}")
        print(f"distance: {result['distance']}")
        print(f"similarity: {result['similarity']}")
        print(f"content: {result['content']}")
        print()


if __name__ == "__main__":
    main()
