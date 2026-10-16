import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_exploration_tools_comprehensive():
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

    # Build DTM
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})

    # Instantiate exploration tools
    exploration_tools = make_exploration_tools(session)
    cosine_sim = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")
    feature_corr = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # 1. Test cosine_similarity_tool against Known Answer 16
    sim_0_1 = cosine_sim.invoke({"doc_index_1": 0, "doc_index_2": 1})["cosine_similarity"]
    sim_0_3 = cosine_sim.invoke({"doc_index_1": 0, "doc_index_2": 3})["cosine_similarity"]
    sim_3_4 = cosine_sim.invoke({"doc_index_1": 3, "doc_index_2": 4})["cosine_similarity"]
    sim_0_7 = cosine_sim.invoke({"doc_index_1": 0, "doc_index_2": 7})["cosine_similarity"]

    assert pytest.approx(sim_0_1, abs=1e-3) == 0.9847
    assert pytest.approx(sim_0_3, abs=1e-3) == 0.0909
    assert pytest.approx(sim_3_4, abs=1e-3) == 0.9847
    assert pytest.approx(sim_0_7, abs=1e-3) == 0.0

    # 2. Test feature_correlation_matrix_tool against Known Answer 19
    corr_res = feature_corr.invoke({"top_k": 20})
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert corr_res["terms"] == expected_terms

    corr_mat = corr_res["correlation_matrix"]
    # alpha (index 0) vs beta (index 3) -> 0.8885
    assert pytest.approx(corr_mat[0][3], abs=1e-3) == 0.8885
    # gamma (index 1) vs delta (index 2) -> 0.8307
    assert pytest.approx(corr_mat[1][2], abs=1e-3) == 0.8307
    # alpha vs gamma -> -0.5718
    assert pytest.approx(corr_mat[0][1], abs=1e-3) == -0.5718

    assert session.pending_figure is not None


if __name__ == "__main__":
    pytest.main([__file__])
