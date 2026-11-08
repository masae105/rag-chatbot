from pathlib import Path
from pypdf import PdfReader


def main():
    # プロジェクトのルートフォルダ
    project_root = Path(__file__).resolve().parents[2]

    # 読み込むPDF
    pdf_path = project_root / "data" / "DOC-001_ipa_dxwp2023-sme.pdf"

    # PDFファイルの存在確認
    if not pdf_path.exists():
        print("PDFファイルが見つかりません。")
        return

    # PDFを読み込む
    reader = PdfReader(pdf_path)

    print("PDF読み込み成功！")
    print(f"ページ数：{len(reader.pages)}")

    # 抽出結果の保存先
    output_path = project_root / "data" / "extracted_text.txt"

    # 全ページからテキストを抽出して保存
    with open(output_path, "w", encoding="utf-8") as output_file:
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""

            print(f"{page_number}ページ目：{len(page_text)}文字")

            output_file.write(f"\n--- {page_number}ページ目 ---\n")
            output_file.write(page_text)
            output_file.write("\n")

    print(f"\n抽出結果を保存しました：{output_path}")


if __name__ == "__main__":
    main()