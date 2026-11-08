import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main():
    # OllamaのローカルAPIに送るURLを指定します。
    url = "http://localhost:11434/api/generate"

    # 使用するモデル、質問文、ストリーミング設定を指定します。
    payload = {
        "model": "qwen2.5:1.5b-instruct",
        "prompt": "製造業のDXとは何ですか？初心者向けに説明してください。",
        "stream": False,
    }

    # リクエスト内容をJSON形式に変換し、UTF-8のバイト列にします。
    request_body = json.dumps(payload, ensure_ascii=False).encode("utf-8")

    # JSONを送信するためのヘッダーを付けてHTTPリクエストを作成します。
    request = Request(
        url,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        # Ollamaへリクエストを送り、最大120秒間レスポンスを待ちます。
        with urlopen(request, timeout=120) as response:
            # APIから返されたJSONを読み取り、Pythonのデータに変換します。
            response_data = json.loads(response.read().decode("utf-8"))

        # レスポンスから生成された回答を取り出して表示します。
        answer = response_data["response"]
        print("回答:")
        print(answer)
    except HTTPError as error:
        # APIがエラー応答を返した場合、HTTPステータスを案内します。
        print(
            f"Ollama APIからエラーが返されました（HTTP {error.code}）。"
            "Ollamaの状態とモデル名を確認してください。"
        )
    except URLError as error:
        # Ollamaに接続できない場合、起動状態やURLの確認を案内します。
        print(
            "Ollamaに接続できませんでした。Ollamaが起動していることを確認してください。"
        )
        print(f"詳細: {error.reason}")
    except TimeoutError:
        # Ollamaから時間内に応答がない場合、再試行を案内します。
        print("Ollamaからの応答がタイムアウトしました。少し待ってから再実行してください。")
    except (json.JSONDecodeError, UnicodeDecodeError):
        # レスポンスをJSONまたはUTF-8として読み取れない場合に案内します。
        print("Ollamaの応答を読み取れませんでした。Ollamaの状態を確認してください。")
    except KeyError:
        # レスポンスに回答の項目が含まれない場合に案内します。
        print("Ollamaの応答に回答が含まれていません。Ollamaの状態を確認してください。")


if __name__ == "__main__":
    # このファイルが直接実行された場合に処理を開始します。
    main()
