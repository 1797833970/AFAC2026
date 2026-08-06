"""
Data processing package for AFAC2026.

Modules:
  _utils          — Shared utilities: tok, _match, _cn2int, _split_long, _number, etc.
  chunker_md      — Markdown/TXT chunking engine (Convention A/B heading rules)
  chunker_lc      — LangChain-based chunking (uses MarkdownHeaderTextSplitter)
  mineru          — MinerU API batch pipeline (PDF/TXT/HTML → Markdown)
  document_parser — PDF/TXT/HTML → section tree parsing (merged from process_docs + document_processor)
"""

from data_processing._utils import (
    # Token & text
    tok, _split_long, _number, _pack, _make,
    # Table handling
    _html_table_to_md, _split_table_rows, _table_aware_split,
    # Heading matching
    _match, _detect_convention, _WEAK_RE,
    # Chinese numeral
    _cn2int, _arabic_seq, _arabic_prefix,
    # Rule sets
    _HEADING_RULES, _RULES_A, _RULES_B,
    # Types
    T_ARABIC, T_CHINESE, T_PLAIN,
)

from data_processing.chunker_md import (
    chunk_markdown,
    chunk_txt_articles,
    run as run_chunk_md,
)

from data_processing.chunker_lc import (
    chunk_markdown as chunk_markdown_lc,
    run as run_chunk_lc,
)

from data_processing.mineru import (
    process_all as mineru_process_all,
    validate_mapping as mineru_validate_mapping,
    build_mapping as mineru_build_mapping,
    collect_files,
    fix_proxy as mineru_fix_proxy,
)

from data_processing.document_parser import (
    Section,
    parse_pdf,
    parse_txt,
    parse_html,
    parse_sections,
    is_header,
    build_mapping,
    validate_mapping,
    show_qa_mapping,
    process_file,
    process_one,
    process_all,
    DOMAINS,
    _DOMAIN_CN,
)
