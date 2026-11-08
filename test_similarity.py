import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


API_URL = "http://localhost:11434/api/embed"
MODEL = "qwen3-embedding:0.6b"
TEXTS = [
    "製造業ではDXの推進が重要です。",
    "工場のデジタル化を進めることが大切です。",
    "今日はカレーを食べました。",
]


def get_embeddings(texts):
    """Ollama APIに文章を送り、各文章のEmbeddingを取得する。"""
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

    with urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))

    if not isinstance(result, dict):
        raise ValueError("API応答がJSONオブジェクト形式ではありません。")
    if "error" in result:
        raise RuntimeError(f"Ollama APIエラー: {result['error']}")

    embeddings = result.get("embeddings")
    if not isinstance(embeddings, list) or len(embeddings) != len(texts):
        raise ValueError("API応答のEmbedding数が入力した文章数と一致しません。")
    if any(
        not isinstance(vector, list)
        or not vector
        or not all(isinstance(value, (int, float)) for value in vector)
        for vector in embeddings
    ):
        raise ValueError("API応答に有効な数値ベクトルが含まれていません。")

    return embeddings


def cosine_similarity(vector_a, vector_b):
    """2つのベクトルのコサイン類似度を計算する。"""
    if len(vector_a) != len(vector_b):
        raise ValueError("ベクトルの次元数が一致しません。")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b))
    norm_a = math.sqrt(sum(value * value for value in vector_a))
    norm_b = math.sqrt(sum(value * value for value in vector_b))
    if norm_a == 0 or norm_b == 0:
        raise ValueError("ゼロベクトルの類似度は計算できません。")

    return dot_product / (norm_a * norm_b)


def main():
    """3文をEmbedding化し、指定されたペアの類似度を表示する。"""
    try:
        embeddings = get_embeddings(TEXTS)
        similarity_ab = cosine_similarity(embeddings[0], embeddings[1])
        similarity_ac = cosine_similarity(embeddings[0], embeddings[2])
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
    except json.JSONDecodeError as error:
        print(f"API応答がJSON形式ではありません: {error}")
        return
    except (RuntimeError, ValueError) as error:
        print(f"Embeddingまたは類似度の計算に失敗しました: {error}")
        return

    print(f"モデル: {MODEL}")
    print(f"A: {TEXTS[0]}")
    print(f"B: {TEXTS[1]}")
    print(f"C: {TEXTS[2]}")
    print(f"\nA-B のコサイン類似度: {similarity_ab:.4f}")
    print(f"A-C のコサイン類似度: {similarity_ac:.4f}")


if __name__ == "__main__":
    main()
