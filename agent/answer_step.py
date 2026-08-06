"""Step 4: Holistic Answer Synthesis.

支持两轮流程：
- 含数值比较/计算题：第一轮让 LLM 生成 Python 代码 → 框架受限执行 → 第二轮 LLM 据执行结果作答
- 纯定性题：保持原单轮流程
"""

import io
import json
import re
import threading

from .prompts import (
    get_answer_system, build_answer_prompt,
    get_answer_codegen_system, build_answer_codegen_prompt,
    build_answer_prompt_with_calc,
)
from .llm_utils import llm_chat


# 数值比较触发词（用于 _needs_calc 扫描）
_CALC_KEYWORDS = (
    '大于', '小于', '不低于', '不超过', '高于', '低于', '至少', '至多',
    '超过', '介于', '范围', '以内', '以上', '以下', '不足', '多于', '少于',
    '同比增长', '同比下降', '增幅', '降幅', '占比', '比率', '倍',
)
# 数字模式：纯数字、百分比、带万元/亿元的金额
_NUM_PATTERN = re.compile(
    r'\d+(?:\.\d+)?\s*(?:%|‰|万元|亿元|元|万|亿|美元|港元|倍|个|天|日|年|月|次|人|份|笔)?'
)


class Answerer:
    """Single-call comprehensive answerer — the LLM sees everything."""

    def __init__(self, model: str = 'qwen3.7-plus'):
        self.model = model
        # 暴露中间结果供 demo 打印
        self._last_raw = ''
        self._last_codegen_raw = ''
        self._last_calc_result = {}

    def answer(self, question_data: dict, compressed: list[dict]) -> dict:
        af = question_data.get('answer_format', 'mcq')

        # ── 判定是否走两轮代码执行流程 ──
        if self._needs_calc(question_data, compressed):
            result = self._answer_with_calc(question_data, compressed)
        else:
            # 原单轮流程
            user_prompt = build_answer_prompt(question_data, compressed)
            content, usage = llm_chat(
                self.model, get_answer_system(question_data), user_prompt,
                label='Answer')
            self._last_raw = content
            self._last_codegen_raw = ''
            self._last_calc_result = {}
            result = self._parse_response(content)
            result['usage'] = usage

        answer = self._format_answer(result.get('correct_options', []), af)
        return {
            'answer': answer,
            'evidence_retrieval': result.get('evidence_retrieval', []),
            'correct_options': result.get('correct_options', []),
            'usage': result.get('usage', {'input': 0, 'output': 0}),
        }

    # ---------------------------------------------------------------
    # 两轮代码执行流程
    # ---------------------------------------------------------------

    def _answer_with_calc(self, question_data: dict, compressed: list[dict]) -> dict:
        """第一轮生成代码 → 执行 → 第二轮基于结果作答。"""
        # ── 第一轮：代码生成 ──
        codegen_system = get_answer_codegen_system(question_data)
        codegen_user = build_answer_codegen_prompt(question_data, compressed)
        codegen_content, codegen_usage = llm_chat(
            self.model, codegen_system, codegen_user,
            label='Answer.CodeGen')
        self._last_codegen_raw = codegen_content

        codegen_result = self._parse_response(codegen_content)
        need_calc = codegen_result.get('need_calc', False)
        code = codegen_result.get('code', '') or ''

        # ── 执行代码（仅当 LLM 声明 need_calc=true 且给出代码）──
        calc_result = {'ok': False, 'error': 'no code generated'}
        if need_calc and code.strip():
            calc_result = self._run_calc_code(code, timeout=5.0)
        else:
            # LLM 判定无需计算，标记跳过
            calc_result = {'ok': False, 'error': 'need_calc=false, skipped'}
        self._last_calc_result = calc_result

        # ── 第二轮：基于执行结果作答 ──
        final_user = build_answer_prompt_with_calc(question_data, compressed, calc_result)
        final_content, final_usage = llm_chat(
            self.model, get_answer_system(question_data), final_user,
            label='Answer.Final')
        self._last_raw = final_content

        result = self._parse_response(final_content)
        result['usage'] = {
            'input': codegen_usage.get('input', 0) + final_usage.get('input', 0),
            'output': codegen_usage.get('output', 0) + final_usage.get('output', 0),
        }
        return result

    # ---------------------------------------------------------------
    # 受限代码执行
    # ---------------------------------------------------------------

    def _run_calc_code(self, code: str, timeout: float = 5.0) -> dict:
        """受限 exec 执行 LLM 生成的 Python 代码。

        - 白名单内置函数，禁用 import/open/eval/exec
        - 捕获 result / results 变量
        - 超时保护（threading，Windows 兼容）
        """
        # 安全检查：禁用危险关键字
        dangerous = re.search(
            r'\b(import|open|eval|exec|__import__|os\.|sys\.|subprocess|'
            r'__builtins__|getattr|setattr|delattr|globals|locals|compile|'
            r'input|breakpoint|exit|quit)\s*\(',
            code,
        )
        if dangerous:
            return {'ok': False, 'error': f'forbidden keyword: {dangerous.group()}'}
        if 'import ' in code or 'import\t' in code:
            return {'ok': False, 'error': 'forbidden: import statement'}

        # 白名单内置
        safe_builtins = {
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'len': len, 'sorted': sorted, 'range': range,
            'int': int, 'float': float, 'str': str, 'bool': bool,
            'list': list, 'dict': dict, 'tuple': tuple, 'set': set,
            'enumerate': enumerate, 'zip': zip, 'map': map, 'filter': filter,
            'any': any, 'all': all, 'repr': repr, 'True': True, 'False': False,
            'None': None,
        }
        # 捕获 print 输出
        stdout_buf = io.StringIO()
        safe_builtins['print'] = lambda *a, **k: print(*a, file=stdout_buf, **k)

        sandbox_globals = {'__builtins__': safe_builtins}
        local_ns: dict = {}

        result = {'ok': False, 'result': None, 'results': None,
                  'stdout': '', 'error': ''}

        def worker():
            try:
                exec(code, sandbox_globals, local_ns)
                result['result'] = local_ns.get('result')
                result['results'] = local_ns.get('results')
                result['stdout'] = stdout_buf.getvalue()
                result['ok'] = True
            except Exception as e:
                result['error'] = f'{type(e).__name__}: {e}'
                result['stdout'] = stdout_buf.getvalue()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            result['ok'] = False
            result['error'] = f'timeout after {timeout}s'
        return result

    # ---------------------------------------------------------------
    # 是否需要走代码执行流程的预判
    # ---------------------------------------------------------------

    def _needs_calc(self, question_data: dict, compressed: list[dict]) -> bool:
        """判定本题是否需要走两轮代码执行流程。

        - 计算题（type 含'计算'）：必走
        - mcq/multi：扫描题干+选项+compressed，含数字+比较词时走
        - tf：不走（用户选择范围外）
        """
        qtype = question_data.get('type', '')
        af = question_data.get('answer_format', 'mcq')

        # 计算题必走
        if '计算' in qtype:
            return True

        # 仅 mcq/multi 走扫描
        if af not in ('mcq', 'multi'):
            return False

        # 拼接所有文本
        question = question_data.get('question', '')
        options = question_data.get('options', {})
        opt_text = ' '.join(str(v) for v in options.values())
        comp_text = ' '.join(
            item.get('compressed_text', '') for item in (compressed or [])
        )
        all_text = f'{question} {opt_text} {comp_text}'

        # 必须同时含数字和比较词才触发（避免纯数字无比较的题目误触发）
        has_num = bool(_NUM_PATTERN.search(all_text))
        has_kw = any(kw in all_text for kw in _CALC_KEYWORDS)
        return has_num and has_kw

    # ---------------------------------------------------------------
    # Response parsing
    # ---------------------------------------------------------------

    def _parse_response(self, content: str) -> dict:
        """Parse LLM JSON response, strip markdown/thinking wrappers."""
        if not content:
            return {'correct_options': [], 'evidence_retrieval': []}

        # 1. 去掉 markdown 代码块围栏
        content = re.sub(r'```[a-z]*\s*', '', content)

        # 2. 去掉 <thinking>...</thinking> 标签（如果存在）
        content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)

        # 3. Try direct JSON parse
        try:
            return json.loads(content.strip())
        except json.JSONDecodeError:
            pass

        # 4. Extract first { ... } block
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

        print(f'  [WARN] JSON parse failed, raw content (first 200): {content[:200]}')
        return {'correct_options': [], 'evidence_retrieval': []}

    # ---------------------------------------------------------------
    # Answer formatting
    # ---------------------------------------------------------------

    def _format_answer(self, correct_options: list[str], af: str) -> str:
        """Convert correct_options list to output answer string."""
        if not correct_options:
            return ''

        if af == 'tf':
            return 'A' if 'A' in correct_options else 'B'
        elif af == 'mcq':
            opt = correct_options[0]
            return opt if len(opt) == 1 and opt in 'ABCD' else opt
        else:
            # multi: correct_options like ["ABD"] or ["A","B","D"]
            raw = ''.join(correct_options)
            return ''.join(sorted(set(ch for ch in raw if ch in 'ABCD')))
