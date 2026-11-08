-- pgvector拡張を有効にします。ベクトル型を使うために必要です。
CREATE EXTENSION IF NOT EXISTS vector;

-- PDFから分割したチャンクと、そのEmbeddingを保存するテーブルを作成します。
CREATE TABLE IF NOT EXISTS document_chunks (
    -- 行ごとに自動で採番される主キーです。
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    -- 元文書を識別するIDです。
    document_id TEXT NOT NULL,

    -- チャンクが含まれていたPDFのページ番号です。
    page_number INTEGER NOT NULL,

    -- 文書内のチャンク番号です。
    chunk_number INTEGER NOT NULL,

    -- チャンクに分割した本文です。
    content TEXT NOT NULL,

    -- qwen3-embedding:0.6bなどのEmbeddingを1024次元で保存します。
    embedding vector(1024) NOT NULL,

    -- 行の登録時に現在日時を自動で設定します。
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);
