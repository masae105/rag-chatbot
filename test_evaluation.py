from dotenv import load_dotenv

from app.retrieval.search_chunks import search_chunks


EVALUATION_QUESTIONS = [
    {
        "id": "Q01",
        "question": "従業員規模が小さくなると、企業のDX取組割合はどのように変化しますか？",
        "expected_chunks": [("DOC-001", 1, 2), ("DOC-001", 1, 3)],
    },
    {
        "id": "Q02",
        "question": "従業員100人以下の企業におけるDXの取組状況には、どのような特徴がありますか？",
        "expected_chunks": [("DOC-001", 1, 3)],
    },
    {
        "id": "Q03",
        "question": "従業員100人以下の企業で、DXに取り組む予定がない企業の割合は、従業員101人以上の企業と比べてどうですか？",
        "expected_chunks": [("DOC-001", 1, 3)],
    },
    {
        "id": "Q04",
        "question": "従業員規模によって、DXの成果が出ている割合にはどのような違いがありますか？",
        "expected_chunks": [("DOC-001", 2, 1)],
    },
    {
        "id": "Q05",
        "question": "従業員100人以下の企業では、DXの成果の有無によって組織連携の実施割合や推進体制にどのような違いがありますか？",
        "expected_chunks": [
            ("DOC-001", 4, 1),
            ("DOC-001", 4, 2),
            ("DOC-001", 7, 2),
        ],
    },
    {
        "id": "Q06",
        "question": "SaaSやパブリッククラウドを活用する企業では、ITシステムの迅速な更新や段階的な拡張の達成度にどのような差がありますか？",
        "expected_chunks": [("DOC-001", 5, 2)],
    },
    {
        "id": "Q07",
        "question": "従業員100人以下のDX取組なし企業では、データ利活用の実施状況や今後の予定にどのような特徴がありますか？",
        "expected_chunks": [("DOC-001", 6, 1), ("DOC-001", 7, 2)],
    },
    {
        "id": "Q08",
        "question": "DX取組なし企業では、どのような取組の実施割合が低く、今後の推進について何が述べられていますか？",
        "expected_chunks": [("DOC-001", 7, 3)],
    },
    {
        "id": "Q09",
        "question": "引用されている中小企業白書では、従業員規模によってデジタル化のきっかけにどのような違いがありますか？",
        "expected_chunks": [("DOC-001", 7, 3), ("DOC-001", 8, 1)],
    },
    {
        "id": "Q10",
        "question": "DX取組なしの中小企業がDXを進める際、初めに取り組むことや活用できる指標として何が挙げられていますか？",
        "expected_chunks": [("DOC-001", 8, 1)],
    },
]


def main():
    load_dotenv()
    total_expected = 0
    total_retrieved = 0
    question_recalls = []

    for evaluation in EVALUATION_QUESTIONS:
        question_id = evaluation["id"]
        question = evaluation["question"]
        expected_chunks = evaluation["expected_chunks"]
        results = search_chunks(question, limit=5)

        print(f"\n===== {question_id} =====")
        print(f"質問: {question}")
        if not results:
            print("検索結果はありませんでした。")
        else:
            for rank, result in enumerate(results, start=1):
                print(f"\n検索順位: {rank}")
                print(f"文書ID: {result['document_id']}")
                print(f"ページ番号: {result['page_number']}")
                print(f"チャンク番号: {result['chunk_number']}")
                print(f"類似度: {result['similarity']}")
                print(f"本文: {result['content']}")

        retrieved_chunks = {
            (result["document_id"], result["page_number"], result["chunk_number"])
            for result in results
        }
        retrieved_expected = sum(
            expected_chunk in retrieved_chunks for expected_chunk in expected_chunks
        )
        expected_count = len(expected_chunks)
        recall_at_5 = (
            retrieved_expected / expected_count * 100 if expected_count else 0.0
        )

        print(f"\n正解チャンク総数: {expected_count}")
        print(f"検索できた正解チャンク数: {retrieved_expected}")
        print(f"Recall@5: {recall_at_5:.1f}%")

        question_recalls.append(recall_at_5)
        total_expected += expected_count
        total_retrieved += retrieved_expected

    macro_recall_at_5 = (
        sum(question_recalls) / len(question_recalls) if question_recalls else 0.0
    )
    micro_recall_at_5 = (
        total_retrieved / total_expected * 100 if total_expected else 0.0
    )
    print("\n===== 全体 =====")
    print(f"正解チャンク総数: {total_expected}")
    print(f"検索できた正解チャンク数: {total_retrieved}")
    print(f"Recall@5 Macro平均: {macro_recall_at_5:.1f}%")
    print(f"Recall@5 Micro平均: {micro_recall_at_5:.1f}%")


if __name__ == "__main__":
    main()
