import streamlit as st

from app.generation.generate_answer import generate_answer
from app.retrieval.search_chunks import search_chunks


st.set_page_config(page_title="RAGチャットボット", page_icon="💬")
st.title("RAGチャットボット")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_question" not in st.session_state:
    st.session_state.pending_question = None
if "processing" not in st.session_state:
    st.session_state.processing = False


def display_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for source in message.get("sources", []):
            caption = (
                f"{source['reference']} · 文書ID: {source['document_id']} · "
                f"ページ: {source['page_number']}"
            )
            if source.get("chunk_number") is not None:
                caption += f" · チャンク: {source['chunk_number']}"
            st.caption(caption)


for chat_message in st.session_state.messages:
    display_message(chat_message)

question = st.chat_input(
    "質問を入力してください",
    disabled=st.session_state.processing,
)

if question and question.strip() and not st.session_state.processing:
    user_message = {"role": "user", "content": question}
    st.session_state.messages.append(user_message)
    st.session_state.pending_question = question
    st.session_state.processing = True
    st.rerun()

if st.session_state.pending_question is not None:
    pending_question = st.session_state.pending_question

    with st.chat_message("assistant"):
        with st.status("関連資料を検索しています…", expanded=True) as status:
            try:
                chunks = search_chunks(pending_question)
            except Exception as error:
                assistant_message = {
                    "role": "assistant",
                    "content": f"検索中にエラーが発生しました: {error}",
                }
                status.update(label="検索に失敗しました", state="error")
            else:
                if not chunks:
                    assistant_message = {
                        "role": "assistant",
                        "content": "関連する資料が見つかりませんでした。",
                    }
                    status.update(label="検索が完了しました", state="complete")
                else:
                    status.update(
                        label="回答を生成しています…",
                        state="running",
                    )
                    try:
                        result = generate_answer(pending_question, chunks)
                    except Exception as error:
                        assistant_message = {
                            "role": "assistant",
                            "content": f"回答の作成中にエラーが発生しました: {error}",
                        }
                        status.update(label="回答の生成に失敗しました", state="error")
                    else:
                        assistant_message = {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["sources"],
                        }
                        status.update(label="回答を作成しました", state="complete")

    st.session_state.messages.append(assistant_message)
    st.session_state.pending_question = None
    st.session_state.processing = False
    st.rerun()
