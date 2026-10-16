import os
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import seaborn as sns
from langchain_core.tools import tool

# Ensure nltk tokenizer data is downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)


def make_tools(session):
    """
    Factory function returning the 7 data pipeline tools bound to the provided session instance.
    """

    @tool
    def inspect_data_tool(n: int = 5) -> dict:
        """
        Inspect the first n rows of the working DataFrame in session.dataframe.

        Args:
            n: Number of rows to inspect (default: 5).

        Returns:
            Summary dictionary with status, row count shown, and head records.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded. Run load_dataset_tool first."}

        df_head = session.dataframe.head(n)
        head_records = df_head.to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "n_rows": len(df_head),
            "head": head_records
        }
        session.store_result("inspect_data_tool", {"n": n}, summary)
        return summary

    @tool
    def check_missing_tool() -> dict:
        """
        Identify missing or empty text rows in session.dataframe.

        Returns:
            Summary dictionary with total rows, missing count, empty text count, and valid row count.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        text_series = df["text"].astype(str)

        null_mask = df["text"].isna()
        empty_mask = text_series.str.strip() == ""
        invalid_mask = null_mask | empty_mask

        n_total = len(df)
        n_missing = int(invalid_mask.sum())
        n_valid = n_total - n_missing

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": n_total,
            "missing_rows": n_missing,
            "valid_rows": n_valid
        }
        session.store_result("check_missing_tool", {}, summary)
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """
        Check for duplicate documents in text. When drop=True, drops all copies of duplicated rows using keep=False.

        Args:
            drop: If True, remove all copies of duplicated text rows from session.dataframe.

        Returns:
            Summary dictionary with total rows, duplicate count, unique count, and drop status.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        dup_mask = df.duplicated(subset=["text"], keep=False)
        n_total = len(df)
        n_duplicates = int(df.duplicated(subset=["text"], keep="first").sum())

        if drop and n_duplicates > 0:
            # Drop ALL copies of duplicated rows (keep=False)
            session.dataframe = df[~dup_mask].reset_index(drop=True)
            if "category" in session.dataframe.columns:
                session.set_labels(session.dataframe["category_name"].values)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_rows": n_total,
            "duplicate_count": n_duplicates,
            "dropped": drop,
            "remaining_rows": len(session.dataframe)
        }
        session.store_result("check_duplicates_tool", {"drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(n: int, random_state: int = 42) -> dict:
        """
        Sample n rows from session.dataframe using DataFrame.sample(n=n, random_state=random_state).

        Args:
            n: Number of rows to sample.
            random_state: Seed for random sampling reproducible results (default: 42).

        Returns:
            Summary dictionary with sampled row count and records.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        if n > len(df):
            n = len(df)

        sampled_df = df.sample(n=n, random_state=random_state)

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "sampled_count": len(sampled_df),
            "sample_indices": sampled_df.index.tolist()
        }
        session.store_result("sample_data_tool", {"n": n, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool() -> dict:
        """
        Compute text_length stats overall and per category_name, and plot a seaborn boxplot on session.pending_figure.

        Returns:
            Summary dictionary with overall and per-category descriptive statistics.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe
        df["text_length"] = df["text"].astype(str).apply(len)

        overall_stats = df["text_length"].describe().to_dict()

        category_stats = {}
        if "category_name" in df.columns:
            for cat, group in df.groupby("category_name"):
                category_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Generate boxplot
        fig, ax = plt.subplots(figsize=(8, 5))
        if "category_name" in df.columns:
            sns.boxplot(data=df, x="category_name", y="text_length", ax=ax)
            ax.set_title("Text Length by Category")
            ax.set_xlabel("Category")
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution")

        ax.set_ylabel("Text Length (characters)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "status": "success",
            "overall": overall_stats,
            "by_category": category_stats
        }
        session.store_result("describe_data_tool", {}, summary)
        return summary

    @tool
    def tokenize_tool() -> dict:
        """
        Tokenize each document text into lowercase unigrams using nltk.word_tokenize.
        Stores output in session.dataframe['unigrams'] and session.artifacts['tokens'].

        Returns:
            Summary dictionary with tokenization statistics.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded. Run load_dataset_tool first."}

        df = session.dataframe

        def _tokenize(text):
            text_str = str(text) if pd.notna(text) else ""
            return nltk.word_tokenize(text_str.lower())

        unigrams = df["text"].apply(_tokenize)
        session.dataframe["unigrams"] = unigrams

        # Store tokens in artifacts
        tokens_list = unigrams.tolist()
        session.artifacts["tokens"] = tokens_list

        total_tokens = sum(len(t) for t in tokens_list)
        avg_tokens = float(total_tokens / len(tokens_list)) if tokens_list else 0.0

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "status": "success",
            "total_documents": len(df),
            "total_tokens": total_tokens,
            "avg_tokens_per_doc": round(avg_tokens, 2)
        }
        session.store_result("tokenize_tool", {}, summary)
        return summary

    @tool
    def list_files_tool(subdirectory: str = ".") -> dict:
        """
        List files and subdirectories inside a given directory path.

        Args:
            subdirectory: Directory path relative to workspace (default: ".").

        Returns:
            Summary dictionary with directory listing details.
        """
        target_dir = subdirectory if subdirectory else "."
        if not os.path.exists(target_dir):
            return {"status": "error", "message": f"Directory '{target_dir}' does not exist."}

        items = os.listdir(target_dir)
        files = []
        directories = []

        for item in items:
            full_path = os.path.join(target_dir, item)
            if os.path.isdir(full_path):
                directories.append(item)
            else:
                files.append(item)

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "status": "success",
            "subdirectory": subdirectory,
            "files": sorted(files),
            "directories": sorted(directories)
        }
        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    return [
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
        tokenize_tool,
        list_files_tool,
    ]
