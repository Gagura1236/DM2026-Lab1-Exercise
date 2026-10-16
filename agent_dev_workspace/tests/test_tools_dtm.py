import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_dtm_tools_comprehensive():
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

    # Instantiate student DTM tools
    dtm_tools = make_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_freq = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    dtm_heatmap = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    # 1. Test build_dtm_tool against Known Answer 4
    dtm_res = build_dtm.invoke({})
    assert dtm_res["n_documents"] == 8
    assert dtm_res["vocabulary_size"] == 5
    assert dtm_res["non_zero"] == 21
    assert dtm_res["total_elements"] == 40
    assert pytest.approx(dtm_res["sparsity_pct"], abs=1e-3) == 47.5
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.artifacts.get("count_vectorizer") is not None

    # 2. Test term_frequency_tool against Known Answer 14
    tf_res = term_freq.invoke({})
    expected_freqs = {
        "always": 7,
        "gamma": 7,
        "alpha": 6,
        "delta": 4,
        "beta": 3
    }
    assert tf_res["term_frequencies"] == expected_freqs

    # 3. Test dtm_heatmap_tool against Known Answer 21
    heatmap_res = dtm_heatmap.invoke({"n_terms": 20, "n_documents": 20})
    assert heatmap_res["sliced_shape"] == [8, 5]
    assert heatmap_res["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    expected_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert heatmap_res["matrix_slice"] == expected_matrix
    assert session.pending_figure is not None


if __name__ == "__main__":
    pytest.main([__file__])
