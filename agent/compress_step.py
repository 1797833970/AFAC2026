"""Step 3: LLM compresses each chunk's text based on description + query."""

import json
import re

from .prompts import COMPRESS_SYSTEM, build_compress_prompt
from .llm_utils import llm_chat


class Compressor:

    def __init__(self, model: str = 'qwen3.7-plus'):
        self.model = model

    def compress(self, retrieval_results: list[dict], question_data=None) -> tuple[list[dict], dict]:
        """[{description, semantic_query, chunks}, ...] → ([{description, chunk_ids, semantic_query, compressed_text}, ...], usage)"""
        groups = {}
        for item in retrieval_results:
            key = (item.get('description', ''), item.get('semantic_query', ''))
            if key not in groups:
                groups[key] = {'chunk_ids': [], 'text_parts': [], 'doc_ids': item.get('doc_ids', [])}
            for chunk in item.get('chunks', []):
                groups[key]['chunk_ids'].append(chunk.get('chunk_id', ''))
                groups[key]['text_parts'].append(chunk.get('text', ''))

        pairs = []
        for (desc, query), group in groups.items():
            pairs.append({
                'description': desc,
                'semantic_query': query,
                'chunk_ids': group['chunk_ids'],
                'text': '\n---\n'.join(group['text_parts']),
                'doc_ids': group.get('doc_ids', []),
            })

        if not pairs:
            return [], {}

        has_empty = any(not p['text'].strip() for p in pairs)

        if has_empty and len(pairs) <= 10:
            result = self._compress_with_negatives(pairs)
            return result, {}

        content, usage = llm_chat(
            self.model, COMPRESS_SYSTEM, build_compress_prompt(pairs),
            max_tokens=4096, label='Compress')
        self._last_raw = content
        result = self._parse(content, pairs)
        return result, usage

    def _compress_with_negatives(self, pairs: list[dict]) -> list[dict]:
        """压缩并为检索为空的目标添加否定验证标记。"""
        non_empty = [p for p in pairs if p['text'].strip()]
        empty = [p for p in pairs if not p['text'].strip()]

        result = []

        if non_empty:
            from .llm_utils import llm_chat
            content, _ = llm_chat(
                self.model, COMPRESS_SYSTEM, build_compress_prompt(non_empty),
                max_tokens=4096, label='Compress')
            result = self._parse(content, non_empty)

        for p in empty:
            desc = p.get('description', '')
            doc_ids = p.get('doc_ids', [])
            query = p.get('semantic_query', '')
            result.append({
                'description': desc,
                'chunk_ids': [],
                'semantic_query': query,
                'compressed_text': (
                    f"【NEGATIVE_VERIFIED】已在文档 {doc_ids} 中用关键词"
                    f"[{query}]检索，未找到相关内容。"
                ),
            })

        return result

    def _parse(self, content: str, pairs: list[dict]) -> list[dict]:
        if not content:
            return self._fallback(pairs)

        for fn in [lambda c: json.loads(c),
                   lambda c: json.loads(re.search(r'\[.*\]', c, re.DOTALL).group())]:
            try:
                data = fn(content)
                if isinstance(data, list):
                    return data
            except (json.JSONDecodeError, AttributeError):
                continue

        return self._fallback(pairs)

    def _fallback(self, pairs):
        return [{'description': p['description'],
                 'chunk_ids': p['chunk_ids'],
                 'semantic_query': p['semantic_query'],
                 'compressed_text': p['text'][:500]} for p in pairs]  # 保留更多原文
