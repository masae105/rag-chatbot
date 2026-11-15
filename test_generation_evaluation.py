import logging
from dotenv import load_dotenv

from app.generation.generate_answer import generate_answer
from app.retrieval.search_chunks import search_chunks
from test_evaluation import EVALUATION_QUESTIONS


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main(target_ids=None):
    load_dotenv()
    success_count = 0
    failure_count = 0
    no_results_count = 0

    for evaluation in EVALUATION_QUESTIONS:
        question_id = evaluation["id"]
        if target_ids is not None and question_id not in target_ids:
            continue
        question = evaluation["question"]
        print(f"\n===== {question_id} =====")
        print(f"質問: {question}")

        try:
            chunks = search_chunks(question, limit=5)
        except Exception as error:
            failure_count += 1
            print(f"検索に失敗しました: {error}")
            continue

        if not chunks:
            no_results_count += 1
            print("検索結果が0件でした。")
            continue

        try:
            result = generate_answer(question, chunks)
        except Exception as error:
            failure_count += 1
            print(f"回答生成に失敗しました: {error}")
            continue

        success_count += 1
        print(f"回答: {result['answer']}")
        print("出典:")
        if result["sources"]:
            for source in result["sources"]:
                print(
                    f"- {source['reference']}: "
                    f"{source['document_id']} / ページ{source['page_number']}"
                )
        else:
            print("- なし")

    print("\n===== 集計 =====")
    print(f"成功件数: {success_count}")
    print(f"失敗件数: {failure_count}")
    print(f"検索結果なし件数: {no_results_count}")


if __name__ == "__main__":
    main()
