"""Tests for mine_patterns_tool in tools_patterns.py."""

import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


def _setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session


def test_mine_patterns_zero_terms_edge_case():
    """Test variance and term_frequency filtering produce 0 patterns due to percentile cutoff."""
    session = _setup_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # Test variance filtering for catA
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_var["n_patterns"] == 0
    assert len(res_var["patterns"]) == 0
    assert "no terms survived filtering" in res_var.get("note", "")

    # Test term_frequency filtering for catA
    res_tf = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "term_frequency",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tf["n_patterns"] == 0
    assert len(res_tf["patterns"]) == 0
    assert "no terms survived filtering" in res_tf.get("note", "")


def test_mine_patterns_fpgrowth_tfidf():
    """Test Known Answer #17: tfidf filtering with fpgrowth algorithm (min_sup=3)."""
    session = _setup_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA with fpgrowth
    res_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    patterns_a = {p["pattern"].strip(): p["support"] for p in res_a["patterns"]}
    assert "alpha" in patterns_a
    assert patterns_a["alpha"] == 3

    # catB with fpgrowth
    res_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    patterns_b = {p["pattern"].strip(): p["support"] for p in res_b["patterns"]}
    assert "gamma" in patterns_b
    assert patterns_b["gamma"] == 4


def test_mine_patterns_topk_and_maxfpgrowth():
    """Test Known Answer #20: topk (k=1) and maxfpgrowth (min_sup=1) algorithms."""
    session = _setup_session()
    tools = make_tools(session)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA with topk (k=1)
    res_topk_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    patterns_topk_a = {p["pattern"].strip(): p["support"] for p in res_topk_a["patterns"]}
    assert "alpha" in patterns_topk_a
    assert patterns_topk_a["alpha"] == 3

    # catB with topk (k=1)
    res_topk_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    patterns_topk_b = {p["pattern"].strip(): p["support"] for p in res_topk_b["patterns"]}
    assert "gamma" in patterns_topk_b
    assert patterns_topk_b["gamma"] == 4

    # catA with maxfpgrowth (min_sup=1) -> 0 patterns on single surviving term case
    res_max_a = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_a["n_patterns"] == 0
    assert len(res_max_a["patterns"]) == 0

    # catB with maxfpgrowth (min_sup=1) -> 0 patterns on single surviving term case
    res_max_b = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_b["n_patterns"] == 0
    assert len(res_max_b["patterns"]) == 0
