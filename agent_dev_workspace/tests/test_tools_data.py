import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def _setup_session_with_data():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load sample fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    tools = make_tools(session)
    tools_dict = {t.name: t for t in tools}
    return session, tools_dict


def test_inspect_data_tool():
    session, tools = _setup_session_with_data()
    inspect_data_tool = tools["inspect_data_tool"]
    
    res = inspect_data_tool.invoke({"n": 3})
    assert res["status"] == "success"
    assert res["n_rows"] == 3
    assert len(res["head"]) == 3


def test_check_missing_tool():
    session, tools = _setup_session_with_data()
    check_missing_tool = tools["check_missing_tool"]
    
    res = check_missing_tool.invoke({})
    assert res["status"] == "success"
    assert res["missing_rows"] == 1


def test_check_duplicates_tool():
    session, tools = _setup_session_with_data()
    check_duplicates_tool = tools["check_duplicates_tool"]
    
    res = check_duplicates_tool.invoke({"drop": True})
    assert res["status"] == "success"
    assert res["duplicate_count"] == 1
    assert res["remaining_rows"] == 6


def test_sample_data_tool():
    session, tools = _setup_session_with_data()
    sample_data_tool = tools["sample_data_tool"]
    
    res = sample_data_tool.invoke({"n": 4, "random_state": 42})
    assert res["status"] == "success"
    assert res["sample_indices"] == [1, 5, 0, 7]


def test_describe_data_tool():
    session, tools = _setup_session_with_data()
    describe_data_tool = tools["describe_data_tool"]
    
    res = describe_data_tool.invoke({})
    assert res["status"] == "success"
    assert res["overall"]["count"] == 8
    assert res["overall"]["max"] == 30
    assert session.pending_figure is not None


def test_tokenize_tool():
    session, tools = _setup_session_with_data()
    tokenize_tool = tools["tokenize_tool"]
    
    res = tokenize_tool.invoke({})
    assert res["status"] == "success"
    assert session.dataframe["unigrams"].iloc[0][:3] == ["always", "alpha", "alpha"]
    assert session.dataframe["unigrams"].iloc[7] == []
    assert "tokens" in session.artifacts


def test_list_files_tool():
    session, tools = _setup_session_with_data()
    list_files_tool = tools["list_files_tool"]
    
    res = list_files_tool.invoke({"subdirectory": "newdataset"})
    assert res["status"] == "success"
    assert res["files"] == ["Reddit-stock-sentiment.csv"]
    assert res["directories"] == []
