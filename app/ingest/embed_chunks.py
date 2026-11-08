import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
EXPECTED_DIMENSIONS = 1024


class DatabaseSaveError(Exception):
    """PostgreSQLのトランザクション登録失敗を表す。"""


def load_chunks(input_path):
    """UTF-8のJSONファイルからチャンクを読み込む。"""
    with input_path.open("r", encoding="utf-8") as input_file:
        chunks = json.load(input_file)

    if not isinstance(chunks, list) or not chunks:
        raise ValueError("JSONの内容が空か、チャンクの一覧形式ではありません。")

    for index, chunk in enumerate(chunks, start=1):
        if not isinstance(chunk, dict) or not isinstance(chunk.get("text"), str):
            raise ValueError(f"{index}件目のチャンクに文字列のtextがありません。")
        if not isinstance(chunk.get("document_id"), str):
            raise ValueError(f"{index}件目のチャンクに文字列のdocument_idがありません。")
        if not isinstance(chunk.get("page_number"), int):
            raise ValueError(f"{index}件目のチャンクに整数のpage_numberがありません。")
        if not isinstance(chunk.get("chunk_index"), int):
            raise ValueError(f"{index}件目のチャンクに整数のchunk_indexがありません。")

    return chunks


def get_embeddings(texts):
    """複数の文章をOllama APIに送り、Embeddingを受け取る。"""
    payload = json.dumps(
        {"model": MODEL, "input": texts},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8"))

    if not isinstance(result, dict):
        raise ValueError("API応答がJSONオブジェクト形式ではありません。")
    if "error" in result:
        raise RuntimeError(f"Ollama APIエラー: {result['error']}")

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise ValueError(
            f"Embedding件数がチャンク数と一致しません"
            f"（チャンク: {len(texts)}件、Embedding: "
            f"{len(embeddings) if isinstance(embeddings, list) else '不明'}件）。"
        )

    for index, vector in enumerate(embeddings, start=1):
        if not isinstance(vector, list) or not all(
            isinstance(value, (int, float)) for value in vector
        ):
            raise ValueError(f"{index}件目のEmbeddingが数値ベクトルではありません。")
        if len(vector) != EXPECTED_DIMENSIONS:
            raise ValueError(
                f"{index}件目のEmbeddingが{EXPECTED_DIMENSIONS}次元ではありません"
                f"（実際: {len(vector)}次元）。"
            )

    return embeddings


def save_chunks(chunks, embeddings, database_url):
    """既存の同一文書データを置き換え、チャンクをDBに保存する。"""
    try:
        import psycopg
        from pgvector.psycopg import register_vector
    except ImportError as error:
        raise RuntimeError(
            "PostgreSQL保存に必要なライブラリがありません。"
            "requirements.txtの依存関係を、実行中のPython環境にインストールしてください。"
        ) from error

    rows = [
        (
            chunk["document_id"],
            chunk["page_number"],
            chunk["chunk_index"],
            chunk["text"],
            embedding,
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]
    document_ids = dict.fromkeys(chunk["document_id"] for chunk in chunks)

    # 接続コンテキストの終了時に成功ならコミット、例外時ならロールバックされる。
    try:
        with psycopg.connect(database_url) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                for document_id in document_ids:
                    cursor.execute(
                        "DELETE FROM document_chunks WHERE document_id = %s",
                        (document_id,),
                    )
                cursor.executemany(
                    """
                    INSERT INTO document_chunks
                        (document_id, page_number, chunk_number, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    rows,
                )
    except psycopg.Error as error:
        raise DatabaseSaveError(str(error)) from error

    return len(rows)


def main():
    """チャンクをEmbedding化し、確認結果とDB登録件数を表示する。"""
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "data" / "chunks.json"

    try:
        chunks = load_chunks(input_path)
        texts = [chunk["text"] for chunk in chunks]
        embeddings = get_embeddings(texts)
    except FileNotFoundError:
        print(f"入力ファイルが見つかりません: {input_path}")
        return
    except OSError as error:
        print(f"入力ファイルを読み込めませんでした: {error}")
        return
    except UnicodeDecodeError as error:
        print(f"ファイルまたはAPI応答をUTF-8として読み取れませんでした: {error}")
        return
    except json.JSONDecodeError as error:
        print(f"JSONの形式が正しくありません: {error}")
        return
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        print(f"Embedding APIがHTTPエラーを返しました（{error.code}）: {details}")
        return
    except URLError as error:
        print(f"Ollamaに接続できません。起動状態とURLを確認してください: {error.reason}")
        return
    except TimeoutError:
        print("Embedding APIへの接続がタイムアウトしました。Ollamaの状態を確認してください。")
        return
    except (RuntimeError, ValueError) as error:
        print(f"Embeddingの処理に失敗しました: {error}")
        return

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        print(
            "PostgreSQLに接続できません。環境変数DATABASE_URLを設定してください。"
        )
        return

    try:
        inserted_count = save_chunks(chunks, embeddings, database_url)
    except RuntimeError as error:
        print(f"PostgreSQL保存の準備に失敗しました: {error}")
        return
    except DatabaseSaveError as error:
        # 接続コンテキストが例外時にトランザクションをロールバックする。
        print(f"PostgreSQLへの登録に失敗しました。トランザクションをロールバックしました: {error}")
        return

    first_chunk = chunks[0]
    first_vector = embeddings[0]
    print(f"総チャンク数: {len(chunks)}")
    print(f"Embedding件数: {len(embeddings)}")
    print(f"Embeddingの次元数: {len(first_vector)}")
    print(f"PostgreSQL登録件数: {inserted_count}")
    print("\n最初のチャンク:")
    print(f"文書ID: {first_chunk.get('document_id', '不明')}")
    print(f"ページ番号: {first_chunk.get('page_number', '不明')}")
    print(f"チャンク番号: {first_chunk.get('chunk_index', '不明')}")
    print(f"ベクトルの先頭5個: {first_vector[:5]}")


if __name__ == "__main__":
    main()
