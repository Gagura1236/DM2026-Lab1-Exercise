import numpy as np
import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def _setup_fixture_session():
    """Helper to set up a clean session loaded with sample_fixture.csv and DTM built."""
    session = SessionState()
    
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Build DTM (required prior step for reduction tools)
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})
    
    return session


def test_reduce_dimensions_pca():
    session = _setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({"method": "pca", "n_components": 2})

    # Assert figure set on session.pending_figure
    assert session.pending_figure is not None, "Pending figure should be set on session"

    # Known answer #7 expected coordinates
    expected_coords = np.array([
        [ 2.3669,  0.8871],
        [ 1.7039,  0.2642],
        [ 1.0408, -0.3587],
        [-2.0759,  1.0129],
        [-1.4554,  0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [ 0.0894, -1.4822],
    ])
    expected_evr = [0.7532, 0.1965]

    coords = np.array(res["coordinates"])
    evr = res["explained_variance_ratio"]

    assert np.allclose(coords, expected_coords, atol=1e-3)
    assert np.allclose(evr, expected_evr, atol=1e-3)


def test_reduce_dimensions_tsne():
    session = _setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    assert session.pending_figure is not None, "Pending figure should be set on session"

    coords = np.array(res["coordinates"])
    assert coords.shape == (8, 2)
    assert np.allclose(coords[5], coords[6]), "Duplicate rows 5 and 6 should have identical coordinates"

    # Primary reference coordinates (scikit-learn baseline)
    ref_coords = np.array([
        [ 1330.9991,    32.1039],
        [  951.2459,   393.5588],
        [  530.8209,   682.9791],
        [-1558.6448,  -314.1625],
        [ -915.5153,  -346.3907],
        [ -280.0313, -1261.2115],
        [ -280.0313, -1261.2115],
        [ -701.3674,  1465.3973],
    ])

    # Alternative coordinates (macOS BLAS backend)
    macos_coords = np.array([
        [  326.7335,  1397.4830],
        [  701.9032,  1017.1497],
        [  957.4244,   563.4032],
        [-1038.9933,   240.4614],
        [-1273.1692,  -375.6626],
        [ -376.0959, -1109.3458],
        [ -376.0959, -1109.3458],
        [ 1449.0189,  -842.5366],
    ])

    matches_ref = np.allclose(coords, ref_coords, atol=1e-1)
    matches_macos = np.allclose(coords, macos_coords, atol=1e-1)

    assert matches_ref or matches_macos, "t-SNE coordinates did not match expected baseline or platform variants"


def test_reduce_dimensions_umap():
    session = _setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })

    # Known answer #9 expected coordinates
    expected_coords = np.array([
        [ 6.6811, -6.8218],
        [ 6.9907, -7.2245],
        [ 7.7022, -7.3712],
        [10.2228, -6.2310],
        [ 9.8759, -5.6537],
        [ 9.3831, -6.5748],
        [ 9.2153, -5.8297],
        [ 8.5860, -7.2622],
    ])

    coords = np.array(res["coordinates"])
    assert np.allclose(coords, expected_coords, atol=1e-2)


def test_binarize_labels_tool():
    session = _setup_fixture_session()
    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({})

    assert res is not None
    assert "result_id" in res
