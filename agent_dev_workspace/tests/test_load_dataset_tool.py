import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool_sample_fixture():
    # Initialize session state and load premade tools
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    
    # Locate load_dataset_tool
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Invoke load_dataset_tool with sample_fixture.csv
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Assert documented Known Answer 1: 8 total documents, split evenly 4 catA and 4 catB
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}


if __name__ == "__main__":
    pytest.main([__file__])
