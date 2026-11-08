import os

from app.ingest.embed_chunks import get_embeddings


def search_chunks(question, limit=5):
    """質問に近いチャンクを検索し、距離と類似度を含む辞書のリストで返す。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("質問文には空でない文字列を指定してください。")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limitには1以上の整数を指定してください。")

    # 取り込み時と同じ関数で質問文をベクトルに変換する。
    query_embedding = get_embeddings([question])[0]

    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "PostgreSQLに接続できません。環境変数DATABASE_URLを設定してください。"
        )

    try:
        import psycopg
        from pgvector import Vector
        from pgvector.psycopg import register_vector
    except ImportError as error:
        raise RuntimeError(
            "PostgreSQL検索に必要なライブラリがありません。"
            "requirements.txtの依存関係を、実行中のPython環境にインストールしてください。"
        ) from error

    # Pythonの数値リストではなく、pgvectorがPostgreSQLのvector型として扱う値にする。
    query_vector = Vector(query_embedding)

    try:
        # 接続コンテキストを抜けると、接続とトランザクションが終了する。
        with psycopg.connect(database_url) as connection:
            register_vector(connection)
            with connection.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT document_id, page_number, chunk_number, content,
                           embedding <=> %s AS distance
                    FROM document_chunks
                    ORDER BY embedding <=> %s ASC
                    LIMIT %s
                    """,
                    (query_vector, query_vector, limit),
                )
                rows = cursor.fetchall()
    except psycopg.Error as error:
        raise RuntimeError(f"PostgreSQLの類似検索に失敗しました: {error}") from error

    # pgvectorのコサイン距離から、値が大きいほど似ている類似度を求める。
    return [
        {
            "document_id": document_id,
            "page_number": page_number,
            "chunk_number": chunk_number,
            "content": content,
            "distance": distance,
            "similarity": 1 - distance,
        }
        for document_id, page_number, chunk_number, content, distance in rows
    ]
