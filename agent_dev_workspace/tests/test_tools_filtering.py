import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_filtering_tools_comprehensive():
    session = SessionState()

    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # Load fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM first
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})

    # Instantiate filtering tools
    filtering_tools = make_filtering_tools(session)
    variance_filter = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    pearson_filter = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_filter = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # 1. Test variance_filter_tool against Known Answer 5
    var_res = variance_filter.invoke({"threshold": 0.15})
    assert set(var_res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert var_res["removed_terms"] == ["always"]
    assert pytest.approx(var_res["variances"]["delta"], abs=1e-3) == 0.25
    assert pytest.approx(var_res["variances"]["always"], abs=1e-3) == 0.109375

    # Verify full_report stored for visualize_result_tool
    var_stored = session.results_store[var_res["result_id"]]
    assert list(var_stored["full_report"].columns) == ["term", "variance"]

    # 2. Test pearson_filter_tool against Known Answer 6
    pearson_res = pearson_filter.invoke({"target_class": "catB"})
    p_scores = pearson_res["pearson_scores"]
    assert pytest.approx(p_scores["delta"], abs=1e-3) == 1.0
    assert pytest.approx(p_scores["gamma"], abs=1e-3) == 0.8307
    assert pytest.approx(p_scores["alpha"], abs=1e-3) == -0.6882

    pearson_stored = session.results_store[pearson_res["result_id"]]
    assert list(pearson_stored["full_report"].columns) == ["term", "pearson_r"]

    # 3. Test spearman_filter_tool against Known Answer 6
    spearman_res = spearman_filter.invoke({"target_class": "catB"})
    s_scores = spearman_res["spearman_scores"]
    assert pytest.approx(s_scores["delta"], abs=1e-3) == 1.0
    assert pytest.approx(s_scores["gamma"], abs=1e-3) == 0.9363
    assert pytest.approx(s_scores["alpha"], abs=1e-3) == -0.7500

    spearman_stored = session.results_store[spearman_res["result_id"]]
    assert list(spearman_stored["full_report"].columns) == ["term", "spearman_r"]


if __name__ == "__main__":
    pytest.main([__file__])
