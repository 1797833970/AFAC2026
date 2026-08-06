"""Step 1: Retrieval Planning — 智能文档理解 + 查询拆解 + 多轮策略。"""

import json
import re

import jieba

from .prompts import get_retrieval_system, build_unified_prompt
from .llm_utils import llm_chat


class Analyzer:

    def __init__(self, model: str = 'qwen3.7-plus'):
        self.model = model

    def analyze(self, question_data: dict) -> tuple[list[dict], dict]:
        """返回 (analyses, usage)。analyses 是展平后的检索目标列表（第一轮+第二轮）。"""
        content, usage = llm_chat(
            self.model, get_retrieval_system(question_data), build_unified_prompt(question_data),
            max_tokens=2048, label='Analyze')
        self._last_raw = content
        plan = self._parse_plan(content, question_data)
        self._last_plan = plan
        self._llm_ok = bool(content and plan.get('retrieval_plan'))
        self._doc_analysis = plan.get('doc_analysis', [])
        analyses = self._flatten(plan)
        return analyses, usage

    def _parse_plan(self, content: str, q: dict) -> dict:
        if not content:
            return self._fallback(q)

        for extract in [lambda c: json.loads(c),
                        lambda c: json.loads(re.search(r'\{.*\}', c, re.DOTALL).group())]:
            try:
                result = extract(content)
                if self._valid(result):
                    return result
            except (json.JSONDecodeError, AttributeError):
                continue

        return self._fallback(q)

    @staticmethod
    def _valid(data: dict) -> bool:
        plan = data.get('retrieval_plan', [])
        if not (isinstance(data, dict)
                and isinstance(plan, list) and len(plan) > 0):
            return False
        # 检查每个检索目标的关键字段（兼容旧格式 doc_id 字符串和新格式 doc_ids 列表）
        for item in plan:
            if not isinstance(item, dict):
                return False
            if 'semantic_query' not in item:
                return False
            if 'doc_id' not in item and 'doc_ids' not in item:
                return False
        return True

    def _fallback(self, q: dict) -> dict:
        doc_ids = q.get('doc_ids', [])
        question = q.get('question', '')

        if doc_ids:
            plan = [{
                'round': 1,
                'description': '在全部分档中检索题干相关信息',
                'semantic_query': question[:60],
                'keywords': ' '.join(jieba.cut_for_search(question))[:80],
                'answer_type': '条款',
                'doc_ids': doc_ids,
            }]
        else:
            plan = [{
                'round': 1,
                'description': '全库检索题干相关信息',
                'semantic_query': question[:60],
                'keywords': ' '.join(jieba.cut_for_search(question))[:80],
                'answer_type': '条款',
                'doc_ids': [],
            }]

        return {
            'doc_analysis': [{'doc_id': d, 'relevance': 'high', 'reason': '默认全查'} for d in doc_ids],
            'retrieval_plan': plan,
            'needs_second_round': False,
            'second_round_reason': '',
        }

    def _flatten(self, plan: dict) -> list[dict]:
        """把 retrieval_plan 展平为检索目标列表，按 round 排序。"""
        analyses = []
        for item in plan.get('retrieval_plan', []):
            desc = item.get('description', '')
            semantic_query = item.get('semantic_query', '')
            keywords = item.get('keywords', '')
            answer_type = item.get('answer_type', '')
            round_num = item.get('round', 1)

            # 兼容旧格式：doc_id 字符串 → doc_ids 列表
            if 'doc_ids' in item:
                doc_id_list = item['doc_ids'] if isinstance(item['doc_ids'], list) else []
            elif 'doc_id' in item:
                doc_id_str = item.get('doc_id', '')
                doc_id_list = [d.strip() for d in doc_id_str.split(',') if d.strip()] if doc_id_str else []
            else:
                doc_id_list = []

            if semantic_query:
                analyses.append({
                    'round': round_num,
                    'description': desc,
                    'semantic_query': semantic_query,
                    'keywords': keywords or semantic_query,
                    'answer_type': answer_type,
                    'doc_ids': doc_id_list,
                })
        return analyses
