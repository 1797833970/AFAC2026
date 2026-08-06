"""
QA Agent — 4-step pipeline orchestrator.
"""
import os
import time

from dotenv import load_dotenv

load_dotenv()

# Clear proxy
for key in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
    os.environ.pop(key, None)
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
import urllib.request
urllib.request.getproxies = lambda: {}

import dashscope
dashscope.api_key = os.environ.get('DASHSCOPE_API_KEY', '')
dashscope.base_http_api_url = os.environ.get('DASHSCOPE_BASE_URL')

from .analyze_step import Analyzer
from .retrieve_step import Retriever
from .compress_step import Compressor
from .answer_step import Answerer
from .settings import MODEL

class QAAgent:
    """Financial long-text QA agent with 4-step pipeline."""

    def __init__(self, analyze_model: str = None,
                 answer_model: str = None):
        self.analyzer = Analyzer(model=analyze_model or MODEL)
        self.retriever = Retriever()
        self.compressor = Compressor(model=analyze_model or MODEL)
        self.answerer = Answerer(model=answer_model or MODEL)

    def answer_single(self, question_data: dict, verbose: bool = True) -> dict:
        """Answer a single question. Returns full result dict.
        Also stores intermediate results in self.trace for debug/demo."""
        qid = question_data.get('qid', 'unknown')
        doc_ids = question_data.get('doc_ids', [])  # 保持原始顺序，不转set
        af = question_data.get('answer_format', 'mcq')
        qtype = question_data.get('type', '')

        if verbose:
            print(f'\n{"=" * 60}')
            print(f'Processing {qid}  format={af}  type={qtype}')
            print(f'Doc IDs: {doc_ids}')
            print(f'Question: {question_data.get("question", "")[:120]}...')
            print(f'{"=" * 60}')

        t0 = time.time()

        # Step 1: Analyze
        if verbose: print(f'  [Step 1] Analyzing...')
        analyses, analyze_usage = self.analyzer.analyze(question_data)
        if not analyses:
            return {'qid': qid, 'answer': '', 'evidence_retrieval': []}

        if verbose:
            doc_analysis = getattr(self.analyzer, '_doc_analysis', [])
            if doc_analysis:
                print(f'  文档相关性分析 ({len(doc_analysis)} docs):')
                for da in doc_analysis:
                    did = da.get('doc_id', '?')
                    rel = da.get('relevance', '?')
                    reason = da.get('reason', '')
                    print(f'    [{rel:>6s}] {did}: {reason}')
            rounds = {}
            for a in analyses:
                r = a.get('round', 1)
                rounds[r] = rounds.get(r, 0) + 1
            round_str = ', '.join(f'R{r}:{n}个目标' for r, n in sorted(rounds.items()))
            print(f'  检索计划: {len(analyses)}个目标 ({round_str})')

        # Step 2: Retrieve
        if verbose: print(f'  [Step 2] Retrieving for {len(analyses)} queries...')
        retrieval = self.retriever.retrieve(analyses, doc_ids)

        # Step 3: Compress
        if verbose: print(f'  [Step 3] Compressing...')
        compressed, compress_usage = self.compressor.compress(retrieval, question_data)

        # Step 4: Answer
        if verbose: print(f'  [Step 4] Answering...')
        result = self.answerer.answer(question_data, compressed)
        answer_usage = result.pop('usage', {'input': 0, 'output': 0})

        # ── 暴露中间结果供 demo 打印 ──
        self.trace = {
            'plan': getattr(self.analyzer, '_last_plan', {}),
            'analyses': analyses,
            'retrieval': retrieval,
            'compressed': compressed,
            'raw_analyze': getattr(self.analyzer, '_last_raw', ''),
            'raw_compress': getattr(self.compressor, '_last_raw', ''),
            'raw_answer': getattr(self.answerer, '_last_raw', ''),
            'raw_codegen': getattr(self.answerer, '_last_codegen_raw', ''),
            'calc_result': getattr(self.answerer, '_last_calc_result', {}),
            'llm_ok': getattr(self.analyzer, '_llm_ok', False),
            'analyze_usage': analyze_usage,
            'compress_usage': compress_usage,
            'answer_usage': answer_usage,
        }

        answer = result.get('answer', '')
        evidence_retrieval = result.get('evidence_retrieval', [])
        elapsed = time.time() - t0

        if verbose:
            print(f'  Answer: {answer or "(empty)"} | {len(evidence_retrieval)} evidence | {elapsed:.0f}s')

        final = {
            'qid': qid,
            'question': question_data.get('question', ''),
            'options': question_data.get('options', {}),
            'answer': answer,
            'correct_options': result.get('correct_options', []),
            'evidence_retrieval': evidence_retrieval,
            'trace': self.trace,
            'tokens': {
                'input': (analyze_usage.get('input', 0)
                          + compress_usage.get('input', 0)
                          + answer_usage.get('input', 0)),
                'output': (analyze_usage.get('output', 0)
                           + compress_usage.get('output', 0)
                           + answer_usage.get('output', 0)),
            },
        }
        return final
