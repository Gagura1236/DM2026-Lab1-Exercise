from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):

    @tool
    def variance_filter_tool(
        threshold: float = 0.0
    ) -> Dict[str, Any]:
        """Calculates per-term population variance (ddof=0) across all documents and filters out terms below threshold.

        Args:
            threshold: Minimum variance required to keep a term. Default is 0.0 (keeps all terms).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0, ddof=0)

        feature_names = session.feature_names
        term_variances = {name: float(v) for name, v in zip(feature_names, variances)}

        kept_terms = [name for name, v in term_variances.items() if v >= threshold]
        removed_terms = [name for name, v in term_variances.items() if v < threshold]

        # full_report formatted specifically for visualize_result_tool
        # Column 1: term, Column 2: variance, sorted descending by variance
        df_report = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(by="variance", ascending=False).reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("variance"),
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "variances": {name: round(v, 6) for name, v in term_variances.items()}
        }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = -1.0
    ) -> Dict[str, Any]:
        """Computes Pearson correlation coefficient between each term's frequency and a one-vs-rest binary target class.

        Args:
            target_class: Category name to construct the binary target vector against.
            threshold: Minimum Pearson r score required to keep a term. Default is -1.0 (keeps all terms).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or missing 'category_name' in session.dataframe."}

        # Binary indicator vector (1 for target_class, 0 otherwise)
        y_binary = (session.dataframe["category_name"].values == target_class).astype(float)

        if np.std(y_binary) == 0:
            return {"error": f"Target class '{target_class}' has no variance (not present or present in all rows)."}

        dense_matrix = session.feature_matrix.toarray()
        pearson_scores = {}

        for i, term in enumerate(session.feature_names):
            term_counts = dense_matrix[:, i]
            if np.std(term_counts) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(term_counts, y_binary)
                if np.isnan(r):
                    r = 0.0
            pearson_scores[term] = float(r)

        feature_names = session.feature_names
        kept_terms = [term for term in feature_names if pearson_scores[term] >= threshold]
        removed_terms = [term for term in feature_names if pearson_scores[term] < threshold]

        # Format full_report for visualize_result_tool
        # Column 1: term, Column 2: pearson_r, sorted descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "pearson_r": [pearson_scores[t] for t in feature_names]
        }).sort_values(by="pearson_r", ascending=False).reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("pearson"),
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "pearson_scores": {t: round(r, 4) for t, r in pearson_scores.items()}
        }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = -1.0
    ) -> Dict[str, Any]:
        """Computes Spearman rank correlation coefficient between each term's frequency and a one-vs-rest binary target class.

        Args:
            target_class: Category name to construct the binary target vector against.
            threshold: Minimum Spearman r score required to keep a term. Default is -1.0 (keeps all terms).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        if session.dataframe is None or "category_name" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or missing 'category_name' in session.dataframe."}

        # Binary indicator vector (1 for target_class, 0 otherwise)
        y_binary = (session.dataframe["category_name"].values == target_class).astype(float)

        if np.std(y_binary) == 0:
            return {"error": f"Target class '{target_class}' has no variance (not present or present in all rows)."}

        dense_matrix = session.feature_matrix.toarray()
        spearman_scores = {}

        for i, term in enumerate(session.feature_names):
            term_counts = dense_matrix[:, i]
            if np.std(term_counts) == 0:
                r = 0.0
            else:
                res = spearmanr(term_counts, y_binary)
                r = res.statistic if hasattr(res, "statistic") else res[0]
                if np.isnan(r):
                    r = 0.0
            spearman_scores[term] = float(r)

        feature_names = session.feature_names
        kept_terms = [term for term in feature_names if spearman_scores[term] >= threshold]
        removed_terms = [term for term in feature_names if spearman_scores[term] < threshold]

        # Format full_report for visualize_result_tool
        # Column 1: term, Column 2: spearman_r, sorted descending
        df_report = pd.DataFrame({
            "term": feature_names,
            "spearman_r": [spearman_scores[t] for t in feature_names]
        }).sort_values(by="spearman_r", ascending=False).reset_index(drop=True)

        summary = {
            "result_id": session.next_result_id("spearman"),
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(feature_names),
            "kept_terms_count": len(kept_terms),
            "removed_terms_count": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "spearman_scores": {t: round(r, 4) for t, r in spearman_scores.items()}
        }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={"target_class": target_class, "threshold": threshold},
            summary=summary,
            full_report=df_report
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
