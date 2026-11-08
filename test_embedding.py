import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
TEXT = "日本語の文章をベクトル化して、RAG検索に利用します。"


def main():
    payload = json.dumps(
        {"model": MODEL, "input": TEXT},
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        API_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read().decode("utf-8"))
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
    except UnicodeDecodeError as error:
        print(f"API応答をUTF-8として読み取れませんでした: {error}")
        return
    except json.JSONDecodeError as error:
        print(f"API応答がJSON形式ではありません: {error}")
        return

    if not isinstance(result, dict):
        print("API応答のJSONがオブジェクト形式ではありません。")
        return

    if "error" in result:
        print(f"Ollama APIエラー: {result['error']}")
        return

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        print("API応答にembeddingsがありません。モデル名とAPI応答を確認してください。")
        return

    vector = embeddings[0]
    if not isinstance(vector, list) or not all(
        isinstance(value, (int, float)) for value in vector
    ):
        print("API応答のembeddingが数値ベクトルではありません。")
        return

    print(f"モデル: {MODEL}")
    print(f"ベクトルの次元数: {len(vector)}")
    print(f"先頭5つの数値: {vector[:5]}")


if __name__ == "__main__":
    main()
