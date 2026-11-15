
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
    has_new_content = False

    def get_overlap(chunk):
        overlap_size = min(100, max(0, chunk_size // 5))
        if not overlap_size:
            return ""
        tail = chunk[-overlap_size:]
        boundaries = [match.end() for match in re.finditer(r"[。！？]", tail)]
        if boundaries:
            if len(boundaries) > 1:
                return tail[boundaries[-2]:]

            tail_start = len(chunk) - len(tail)
            sentence_end = tail_start + boundaries[0]
            previous_boundaries = [
                match.end()
                for match in re.finditer(r"[。！？]", chunk[:sentence_end - 1])
            ]
            sentence_start = previous_boundaries[-1] if previous_boundaries else 0
            if tail_start and not previous_boundaries:
                return ""

            sentence = chunk[sentence_start:sentence_end]
            return sentence if len(sentence) <= overlap_size else ""
        return tail

    def split_at_boundary(value, limit):
        prefix = value[:limit]
        boundaries = [match.end() for match in re.finditer(r"[。！？]", prefix)]
        return boundaries[-1] if boundaries else limit

    def add_paragraph(paragraph, separator="\n\n"):
        nonlocal current_chunk, has_new_content

        sentences = [
            sentence
            for sentence in re.split(r"(?<=[。！？])", paragraph)
            if sentence
        ] or [paragraph]

        for index, sentence in enumerate(sentences):
            pending = (separator if index == 0 and current_chunk else "") + sentence

            while pending:
                available = chunk_size - len(current_chunk)
                if len(pending) <= available:
                    current_chunk += pending
                    has_new_content = True
                    break

                if current_chunk and has_new_content:
                    chunks.append(current_chunk)
                    current_chunk = get_overlap(current_chunk)
                    has_new_content = False
                    continue

                split_index = split_at_boundary(pending, available)
                current_chunk += pending[:split_index]
                pending = pending[split_index:]
                has_new_content = True
                if pending:
                    chunks.append(current_chunk)
                    current_chunk = get_overlap(current_chunk)
                    has_new_content = False

    for paragraph in paragraphs:
        if not paragraph.strip():
            continue

        if re.match(r"^第\d+章", paragraph):
            if current_chunk and has_new_content:
                chunks.append(current_chunk)
            current_chunk = ""
            has_new_content = False
            add_paragraph(paragraph, separator="")
            continue

        add_paragraph(paragraph)

    if current_chunk and has_new_content:
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
