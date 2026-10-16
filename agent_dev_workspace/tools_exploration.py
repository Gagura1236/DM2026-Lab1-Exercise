from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):

    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two documents in session.feature_matrix.

        Args:
            doc_index_1: Index of the first document row.
            doc_index_2: Index of the second document row.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        total_docs = session.feature_matrix.shape[0]
        if not (0 <= doc_index_1 < total_docs) or not (0 <= doc_index_2 < total_docs):
            return {"error": f"doc_index out of bounds (valid range: 0 to {total_docs - 1})."}

        vec1 = session.feature_matrix[doc_index_1]
        vec2 = session.feature_matrix[doc_index_2]

        sim = cosine_similarity(vec1, vec2)[0][0]
        if np.isnan(sim):
            sim = 0.0

        summary = {
            "result_id": session.next_result_id("cosine_sim"),
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": round(float(sim), 4)
        }

        session.store_result("cosine_similarity_tool", {
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2
        }, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20
    ) -> Dict[str, Any]:
        """Computes Pearson correlation matrix across top top_k terms by variance and sets a heatmap on session.pending_figure.

        Args:
            top_k: Number of highest-variance terms to include in the correlation matrix (default: 20).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        variances = np.var(dense_matrix, axis=0, ddof=0)

        n_terms = len(session.feature_names)
        actual_k = min(top_k, n_terms)

        # Select top-k term indices by variance
        top_indices = np.argsort(variances)[::-1][:actual_k]
        top_terms = [session.feature_names[i] for i in top_indices]
        top_matrix = dense_matrix[:, top_indices]

        # Compute Pearson correlation matrix across columns (rowvar=False)
        corr_matrix = np.corrcoef(top_matrix, rowvar=False)
        # Handle single term edge case or NaNs
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        if actual_k == 1:
            corr_matrix = np.array([[1.0]])

        fig, ax = plt.subplots(figsize=(max(6, actual_k * 0.8), max(5, actual_k * 0.7)))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".4f",
            cmap="vlag",
            ax=ax
        )
        ax.set_title(f"Top-{actual_k} Feature Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig

        corr_list = np.round(corr_matrix, 4).tolist()

        summary = {
            "result_id": session.next_result_id("feature_corr"),
            "top_k": actual_k,
            "terms": top_terms,
            "correlation_matrix": corr_list
        }

        session.store_result("feature_correlation_matrix_tool", {"top_k": top_k}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
