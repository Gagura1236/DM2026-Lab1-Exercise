from typing import List, Optional, Dict, Any
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
import seaborn as sns


def make_tools(session):

    @tool
    def build_dtm_tool(
        ngram_range: List[int] = [1, 1],
        max_df: float = 1.0,
        min_df: int = 1,
        max_features: Optional[int] = None,
        stop_words: Optional[str] = None,
        lowercase: bool = True
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from document text using CountVectorizer.

        Args:
            ngram_range: Lower and upper boundary of range of n-values for different n-grams to be extracted (e.g. [1, 1]).
            max_df: Ignore terms that have a document frequency strictly higher than the given threshold.
            min_df: Ignore terms that have a document frequency strictly lower than the given threshold.
            max_features: Build a vocabulary that only considers the top max_features ordered by term frequency.
            stop_words: Language for stop words, e.g., 'english' or None.
            lowercase: Whether to convert all characters to lowercase before tokenizing.
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"error": "No dataframe loaded or missing 'text' column in session.dataframe."}

        # Convert list to tuple for sklearn CountVectorizer
        ngram_tuple = tuple(ngram_range)

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            stop_words=stop_words,
            lowercase=lowercase
        )

        texts = session.dataframe["text"].fillna("").astype(str).tolist()
        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        # Update session state
        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)

        # Calculate sparsity
        n_rows, n_cols = feature_matrix.shape
        total_elements = n_rows * n_cols
        non_zero = feature_matrix.nnz
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        summary = {
            "result_id": session.next_result_id("dtm"),
            "n_documents": n_rows,
            "vocabulary_size": n_cols,
            "non_zero": int(non_zero),
            "total_elements": int(total_elements),
            "sparsity_pct": round(sparsity_pct, 4)
        }

        session.store_result("build_dtm_tool", {
            "ngram_range": ngram_range,
            "max_df": max_df,
            "min_df": min_df,
            "max_features": max_features,
            "stop_words": stop_words,
            "lowercase": lowercase
        }, summary)

        return summary

    @tool
    def term_frequency_tool(
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Aggregates total occurrences for each term across the global document-term matrix.

        Args:
            top_k: If specified, returns only the top-k most frequent terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        # Sum occurrences per column (term) across all documents (rows)
        counts = session.feature_matrix.sum(axis=0).A1
        term_freqs = {name: int(count) for name, count in zip(session.feature_names, counts)}

        # Sort terms by frequency descending
        sorted_freqs = dict(sorted(term_freqs.items(), key=lambda item: item[1], reverse=True))

        if top_k is not None and top_k > 0:
            result_freqs = dict(list(sorted_freqs.items())[:top_k])
        else:
            result_freqs = sorted_freqs

        summary = {
            "result_id": session.next_result_id("term_freq"),
            "total_terms": len(sorted_freqs),
            "term_frequencies": result_freqs
        }

        session.store_result("term_frequency_tool", {"top_k": top_k}, summary)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Renders a positional slice heatmap of raw document-term counts from session.feature_matrix.

        Args:
            n_terms: Number of terms (columns) to slice from the vocabulary.
            n_documents: Number of documents (rows) to slice from the dataset.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        total_docs, total_terms = session.feature_matrix.shape
        doc_slice = min(n_documents, total_docs)
        term_slice = min(n_terms, total_terms)

        sliced_matrix = session.feature_matrix[0:doc_slice, 0:term_slice].toarray()
        sliced_terms = session.feature_names[0:term_slice]
        doc_labels = [f"Doc {i}" for i in range(doc_slice)]

        fig, ax = plt.subplots(figsize=(max(6, term_slice * 0.8), max(4, doc_slice * 0.4)))
        sns.heatmap(
            sliced_matrix,
            xticklabels=sliced_terms,
            yticklabels=doc_labels,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax
        )
        ax.set_title("Document-Term Matrix Slice")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        session.pending_figure = fig

        summary = {
            "result_id": session.next_result_id("dtm_heatmap"),
            "sliced_shape": [doc_slice, term_slice],
            "terms": sliced_terms,
            "matrix_slice": sliced_matrix.tolist()
        }

        session.store_result("dtm_heatmap_tool", {
            "n_terms": n_terms,
            "n_documents": n_documents
        }, summary)

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
