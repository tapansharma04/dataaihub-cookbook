"""Query transformation tests without network calls."""

from rag.query_transformer import build_multi_queries, parse_alternative_queries


def test_parse_structured_output_json_array():
    raw = '["tls handshake failure checklist", "check clock skew for tls"]'
    out = parse_alternative_queries(raw, max_alternatives=3)
    assert out == ["tls handshake failure checklist", "check clock skew for tls"]


def test_parse_strips_markdown_code_fence():
    raw = '```json\n["keepalive timeout relay", "idle session RST"]\n```'
    out = parse_alternative_queries(raw, max_alternatives=2)
    assert out == ["keepalive timeout relay", "idle session RST"]


def test_bound_alternative_queries_count():
    raw = '["a","b","c","d"]'
    out = parse_alternative_queries(raw, max_alternatives=2)
    assert out == ["a", "b"]


def test_build_multi_queries_retains_original_first():
    out = build_multi_queries(
        "original question",
        max_alternative_queries=2,
        transform_fn=lambda _q, _m: ["alt one", "alt two", "alt three"],
    )
    assert out[0] == "original question"
    assert len(out) == 3
