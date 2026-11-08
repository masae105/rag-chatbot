
from pathlib import Path
from pypdf import PdfReader
import json
import re

def normalize_text(text):
    """PDFから抽出した文章の改行を整理する。"""

    lines = [line.strip() for line in text.splitlines()]

    paragraphs = []
    current_paragraph = []

    for line in lines:
        if not line:
            if current_paragraph:
                paragraphs.append("".join(current_paragraph))
                current_paragraph = []
            continue

        if line == "IPA 調査分析ディスカッション・ペーパー2023-02":
            if current_paragraph:
                paragraphs.append("".join(current_paragraph))
                current_paragraph = []
            continue

        # ページ番号だけの行は除外
        if re.fullmatch(r"\d+", line):
            continue

        caption_match = re.search(
            r"図表-\s*\d+\s+(?!は|が|を|に|で|の|へ|と|も|や)\S.*$",
            line,
        )
        if caption_match:
            caption_start = caption_match.start()
            if caption_start == 0:
                if current_paragraph:
                    paragraphs.append("".join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(caption_match.group())
                continue

            preceding_text = line[:caption_start].rstrip()
            if re.search(r"[。！？]$", preceding_text):
                current_paragraph.append(preceding_text)
                if current_paragraph:
                    paragraphs.append("".join(current_paragraph))
                    current_paragraph = []
                paragraphs.append(caption_match.group())
                continue

        # 章見出しは独立させる
        if re.match(r"^第\d+章", line):
            if current_paragraph:
                paragraphs.append("".join(current_paragraph))
                current_paragraph = []

            paragraphs.append(line)
            continue

        current_paragraph.append(line)

    if current_paragraph:
        paragraphs.append("".join(current_paragraph))

    return "\n\n".join(paragraphs)

def split_text(text, chunk_size=500):
    """章見出しと直後の本文を保ち、段落のまとまりを優先して分割する。"""

    paragraphs = text.split("\n\n")

    chunks = []
    current_chunk = ""
    chapter_needs_body = False

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        if re.match(r"^第\d+章", paragraph):
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = paragraph
            chapter_needs_body = True
            continue

        if chapter_needs_body:
            candidate = current_chunk + "\n\n" + paragraph

            if len(candidate) <= chunk_size:
                current_chunk = candidate
            else:
                sentences = [
                    sentence
                    for sentence in re.split(r"(?<=[。！？])", paragraph)
                    if sentence
                ]
                if not sentences:
                    sentences = [paragraph]

                current_chunk += "\n\n" + sentences[0]

                for sentence in sentences[1:]:
                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence

            chapter_needs_body = False
            continue

        if (
            re.match(r"^図表-\s*\d+\s+(?!は|が|を|に|で|の|へ|と|も|や)\S", paragraph)
            and current_chunk
        ):
            current_chunk += "\n\n" + paragraph
            continue

        # 既存チャンクに段落を追加できるか確認
        candidate = current_chunk + "\n\n" + paragraph if current_chunk else paragraph

        if len(candidate) <= chunk_size:
            current_chunk = candidate
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # 長い段落は文章単位で分割する
            if len(paragraph) > chunk_size:
                sentences = re.split(r"(?<=[。！？])", paragraph)
                current_chunk = ""

                for sentence in sentences:
                    if not sentence:
                        continue

                    if len(current_chunk) + len(sentence) <= chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = sentence
            else:
                current_chunk = paragraph

    if current_chunk:
        chunks.append(current_chunk)

    return chunks


def main():
    # プロジェクトのルートフォルダ
    project_root = Path(__file__).resolve().parents[2]

    # PDFファイル
    pdf_path = project_root / "data" / "DOC-001_ipa_dxwp2023-sme.pdf"

    if not pdf_path.exists():
        print("PDFファイルが見つかりません。")
        return

    # PDF読み込み
    reader = PdfReader(pdf_path)

    all_chunks = []

        # ページごとにチャンク化
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        page_text = normalize_text(page_text)

        chunks = split_text(page_text)

        for chunk_index, chunk in enumerate(chunks, start=1):
            all_chunks.append({
                "document_id": "DOC-001",
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk
            })

    # チャンクをUTF-8のJSONファイルに保存
    output_path = project_root / "data" / "chunks.json"
    try:
        with output_path.open("w", encoding="utf-8") as output_file:
            json.dump(all_chunks, output_file, ensure_ascii=False, indent=2)
    except OSError as error:
        print(f"チャンクJSONを保存できませんでした：{output_path} ({error})")
        return

    print(f"チャンクJSONを保存しました：{output_path}")

    # 結果表示
    print(f"総チャンク数：{len(all_chunks)}")

    for chunk in all_chunks[:5]:
        print("\n--------------------")
        print(f"文書ID：{chunk['document_id']}")
        print(f"ページ番号：{chunk['page_number']}")
        print(f"チャンク番号：{chunk['chunk_index']}")
        print(f"文字数：{len(chunk['text'])}")
        print(chunk["text"])


if __name__ == "__main__":
    main()
