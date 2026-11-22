# RAG型チャットボット

公開されているPDF資料を検索し、資料の記述を根拠として回答するRAG型チャットボットです。

Python、Streamlit、Ollama、PostgreSQL、pgvectorを使用して開発しています。

## 1. プロジェクト概要

中小製造業の経営者・経営幹部を主な利用者として想定し、IPAなどの公的機関が公開する資料から、経営判断に必要な情報を検索・整理することを目的としています。

公開資料には、業界動向や中小企業の取り組み、DX推進など、経営に役立つ情報が含まれています。一方で、必要な情報をPDFの中から探し出し、内容を確認するには時間と手間がかかります。

本システムでは、PDFから抽出した情報を検索可能な形式で保存し、利用者の質問に関連する箇所を検索します。検索結果をもとにLLMが回答を生成し、根拠となる資料のページ番号などを表示します。

**開発状況：** Phase 1の基本機能を実装し、回答精度と根拠検証機能を継続的に改善中。

## 2. 主な機能

- PDFからのテキスト抽出
- テキストの正規化とチャンク分割
- OllamaによるEmbedding生成
- PostgreSQL + pgvectorへのベクトル保存
- 質問文のEmbedding生成
- コサイン距離による類似チャンク検索
- 検索結果を利用したLLMによる回答生成
- 回答の根拠となる情報の抽出・検証
- 回答と出典情報の表示
- 根拠を確認できない場合の回答抑制

## 3. システム構成

### 使用技術

| 分類 | 技術 | 用途 |
|---|---|---|
| 言語 | Python | アプリケーション・検索・生成処理 |
| UI | Streamlit | チャット画面 |
| LLM実行環境 | Ollama | ローカルLLMの実行 |
| LLM | qwen2.5:3b-instruct | 事実抽出・回答生成 |
| Embeddingモデル | qwen3-embedding:0.6b | テキストのベクトル化 |
| データベース | PostgreSQL 17 | チャンク・Embeddingの保存 |
| ベクトル検索 | pgvector | ベクトル類似検索 |
| DB接続 | psycopg | PostgreSQLとの接続 |
| PDF処理 | pypdf | PDFからのテキスト抽出 |
| コンテナ | Docker Compose | PostgreSQL環境の構築 |

### RAGの処理フロー

#### ① 文書取り込み

- PDFからページ単位でテキストを抽出
- ヘッダーやページ番号などを整理
- 文章や段落のまとまりを考慮してチャンクに分割
- OllamaでEmbeddingを生成
- PostgreSQL + pgvectorに保存

#### ② 質問・回答生成

- Streamlitで利用者が質問を入力
- 質問文をEmbeddingに変換
- pgvectorで質問に関連するチャンクを検索
- 検索結果から回答に利用する事実を抽出
- Python側で引用文や数値、条件などを検証
- 検証済みの情報をもとにLLMが回答文を生成
- 回答と出典情報をStreamlitに表示

### 検索方式

pgvectorのコサイン距離演算子 `<=>` を使用しています。

質問文のEmbeddingと保存済みチャンクのEmbeddingを比較し、距離が小さい順に検索結果を取得します。

デフォルトの検索件数は上位5チャンクです。

## 4. 回答生成における設計上の工夫

本プロジェクトでは、LLMに回答生成のすべてを任せるのではなく、LLMとPythonの役割を分離する設計を採用しています。

### LLMの役割

- 資料から質問に関連する事実を抽出
- 抽出した情報をもとに回答文を生成

### Pythonの役割

- 抽出された事実の検証
- 原文引用との整合性確認
- 数値や比較条件、否定表現などの確認
- 出典情報の管理
- 根拠が不十分な場合の回答抑制

また、LLMの出力にはJSON Schemaを利用し、事実抽出と回答生成の出力形式を制御しています。

このように処理を分離することで、回答の根拠を確認しやすくし、誤った情報を回答に含めるリスクの低減を目指しています。

## 5. チャンク分割の仕様

現在の実装では、次の方針でチャンクを生成しています。

| 項目 | 内容 |
|---|---|
| 基本チャンクサイズ | 約500文字 |
| オーバーラップ | 最大100文字 |
| 分割単位 | 段落・文章の区切りを考慮 |
| 章見出し | 本文との関係を考慮して分割 |
| メタデータ | 文書ID・ページ番号・チャンク番号 |

単純な文字数による分割だけでなく、文章のまとまりをできるだけ保持することを意識しています。

## 6. データベース構成

PostgreSQLにpgvector拡張を導入し、`document_chunks` テーブルでチャンクとEmbeddingを管理しています。

主なカラムは次のとおりです。

| カラム | 内容 |
|---|---|
| id | 主キー |
| document_id | 文書ID |
| page_number | PDFのページ番号 |
| chunk_number | チャンク番号 |
| content | チャンク本文 |
| embedding | 1024次元のベクトル |
| created_at | 登録日時 |

テーブル定義は `docs/create_tables.sql` に記載しています。

## 7. ディレクトリ構成

```text
rag-chatbot/
├── app/
│   ├── generation/
│   │   └── generate_answer.py
│   ├── ingest/
│   │   ├── pdf_reader.py
│   │   ├── chunker.py
│   │   └── embed_chunks.py
│   └── retrieval/
│       └── search_chunks.py
├── data/
├── docs/
│   ├── create_tables.sql
│   ├── phase3_evaluation.md
│   └── requirements.md
├── streamlit_app.py
├── compose.yaml
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## 8. 動作環境

以下の環境での開発を想定しています。

- Windows 11
- Python 3.13系
- Docker Desktop
- Ollama
- PostgreSQL 17（Dockerコンテナ）
- VS Code

## 9. 環境構築・起動方法

### ① リポジトリの取得

```powershell
git clone https://github.com/masae105/rag-chatbot.git
cd rag-chatbot
```

### ② Python仮想環境の作成

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### ③ Pythonライブラリのインストール

```powershell
pip install -r requirements.txt
```

### ④ 環境変数の設定

`.env.example` をコピーして `.env` を作成します。

```powershell
Copy-Item .env.example .env
```

`.env` の `POSTGRES_PASSWORD` を任意の値に変更してください。

実際のパスワードをGitHubに公開しないでください。

### ⑤ PostgreSQLの起動

```powershell
docker compose up -d
```

Docker ComposeによってPostgreSQL 17とpgvectorを含むコンテナが起動します。

接続ポートはホスト側の5433番です。

### ⑥ データベースのテーブル作成

以下のコマンドで、pgvector拡張とチャンク保存用テーブルを作成します。

```powershell
Get-Content docs\create_tables.sql -Raw | docker exec -i rag-postgres psql -U rag_user -d rag_db
```

### ⑦ DATABASE_URLの設定

PowerShellで接続情報を設定します。

```powershell
$env:DATABASE_URL = "postgresql://rag_user:設定したパスワード@localhost:5433/rag_db"
```

### ⑧ Ollamaモデルの準備

Ollamaを起動し、以下のモデルを取得します。

```powershell
ollama pull qwen2.5:3b-instruct
ollama pull qwen3-embedding:0.6b
```

- qwen2.5:3b-instruct：事実抽出・回答生成
- qwen3-embedding:0.6b：テキストのベクトル化（1024次元）

### ⑨ PDFの配置とデータ登録

対象PDFを `data` フォルダに配置します。

現在の取り込みスクリプトでは、次のファイル名を使用しています。

```text
data/
└── DOC-001_ipa_dxwp2023-sme.pdf
```

PDFのテキスト抽出、チャンク分割、Embedding生成・DB登録を順に実行します。

**1. PDFからテキストを抽出**

```powershell
python -m app.ingest.pdf_reader
```

PDFのテキストを抽出し、確認用の `data/extracted_text.txt` を生成します。

**2. チャンク分割**

```powershell
python -m app.ingest.chunker
```

PDFを読み込み、テキストの正規化とチャンク分割を行い、`data/chunks.json` を生成します。

**3. Embedding生成・DB登録**

```powershell
python -m app.ingest.embed_chunks
```

チャンクのEmbeddingを生成し、PostgreSQLに登録します。

※ `pdf_reader.py` と `chunker.py` は、それぞれPDFを読み込んで処理します。`chunker.py` は `extracted_text.txt` ではなく、元のPDFからチャンクを生成します。

### ⑩ Streamlitの起動

```powershell
streamlit run streamlit_app.py
```

ブラウザに表示される画面から質問を入力できます。

## 10. 評価・改善の取り組み

回答の品質を高めるため、実際の質問を用いた評価と改善を継続しています。

主な取り組みは次のとおりです。

- 質問に対する検索結果の確認
- 検索結果と回答内容の整合性確認
- 事実抽出の精度改善
- 引用文と原文の一致確認
- 数値・比較条件・対象範囲の保持
- 根拠が不十分な場合の回答抑制
- LLMとPythonの処理責務の分離

評価結果は、今後の改善に利用する開発記録として保存しています。

## 11. 現在の開発状況と今後の予定

### Phase 1：PDF検索と根拠付き回答

基本機能を実装済みです。現在は回答精度と根拠検証機能を継続的に改善しています。

### Phase 2：複数資料の比較

今後の実装予定です。

- 複数資料の共通点・相違点の整理
- 年度ごとの情報の比較
- 資料間の関連情報の抽出

### Phase 3：経営情報の整理支援

将来的な拡張予定です。

- 経営課題に関係する公開情報の整理
- 業界動向の把握支援
- 経営判断に必要な情報の比較・整理

## 12. 注意事項

- 本システムは開発・検証段階のプロジェクトです。
- 生成された回答は必ず原資料を確認してください。
- 回答を経営判断や専門家の助言に代わるものとして利用することは想定していません。
- PDFなどの元データや生成データは、このリポジトリには含めていません。
- 利用する資料の著作権・利用条件を確認したうえで運用してください。