from pathlib import Path
from dotenv import load_dotenv
from test_evaluation import EVALUATION_QUESTIONS
from app.retrieval.search_chunks import search_chunks
from app.generation.generate_answer import generate_answer
load_dotenv(dotenv_path=Path('.env'))
lines = ['# Q01-Q05 実回答評価（回答生成改善後）', '', 'DB操作: SELECTのみ。全質問で検索結果Top5をそのまま回答生成に使用。', '']
for item in EVALUATION_QUESTIONS[:5]:
    qid, question = item['id'], item['question']
    print(f'===== {qid} =====', flush=True)
    try:
        results = search_chunks(question, limit=5)
    except Exception as exc:
        reason = getattr(exc, 'reason', None)
        detail = type(reason).__name__ if reason is not None else ''
        print(f'SEARCH_ERROR {type(exc).__name__} {detail}', flush=True)
        lines += [f'## {qid}', f'質問: {question}', f'検索失敗 ({type(exc).__name__}) {detail}', '']
        continue
    print('CHUNKS', [(r['document_id'], r['page_number'], r['chunk_number']) for r in results], flush=True)
    try:
        result = generate_answer(question, results)
    except Exception as exc:
        print(f'GENERATION_ERROR {type(exc).__name__}', flush=True)
        lines += [f'## {qid}', f'質問: {question}', f'回答生成失敗 ({type(exc).__name__})', '']
        continue
    refs = [(s['reference'], s['document_id'], s['page_number']) for s in result['sources']]
    print('ANSWER', result['answer'], flush=True)
    print('SOURCES', refs, flush=True)
    lines += [f'## {qid}', f'質問: {question}', f'回答: {result["answer"]}', '出典:']
    lines += [f'- {ref}: {doc} p.{page}' for ref, doc, page in refs] or ['- なし']
    lines += ['']
Path('real_answer_evaluation_hybrid_q01-q05.md').write_text('\n'.join(lines), encoding='utf-8')
print('REPORT_WRITTEN', flush=True)
