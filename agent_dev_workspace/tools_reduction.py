from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):

    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        random_state: Optional[int] = 42,
        perplexity: float = 30.0,
        n_neighbors: int = 15
    ) -> Dict[str, Any]:
        """Reduces feature matrix dimensions to 2D using PCA, t-SNE, or UMAP, and plots a scatter plot colored by category.

        Args:
            method: Dimensionality reduction algorithm ('pca', 'tsne', or 'umap').
            n_components: Target number of dimensions (default: 2).
            random_state: Random seed for reproducibility.
            perplexity: Perplexity parameter for t-SNE (adjusted if >= n_samples).
            n_neighbors: Number of neighbors parameter for UMAP (adjusted if >= n_samples).
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix found in session. Call build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        n_samples = dense_matrix.shape[0]

        if n_samples < 2:
            return {"error": "At least 2 documents are required for dimensionality reduction."}

        explained_var = None

        if method == "pca":
            from sklearn.decomposition import PCA
            model = PCA(n_components=n_components, random_state=random_state)
            coords = model.fit_transform(dense_matrix)
            explained_var = model.explained_variance_ratio_.tolist()

        elif method == "tsne":
            from sklearn.manifold import TSNE
            # t-SNE perplexity must be strictly less than n_samples
            safe_perplexity = perplexity
            if safe_perplexity >= n_samples:
                safe_perplexity = max(1.0, float(n_samples - 1))

            model = TSNE(
                n_components=n_components,
                perplexity=safe_perplexity,
                random_state=random_state
            )
            coords = model.fit_transform(dense_matrix)

        elif method == "umap":
            import umap
            # UMAP n_neighbors must be strictly less than n_samples and >= 2
            safe_n_neighbors = n_neighbors
            if safe_n_neighbors >= n_samples:
                safe_n_neighbors = max(2, n_samples - 1)

            model = umap.UMAP(
                n_components=n_components,
                n_neighbors=safe_n_neighbors,
                random_state=random_state
            )
            coords = model.fit_transform(dense_matrix)
        else:
            return {"error": f"Unsupported method '{method}'. Choose 'pca', 'tsne', or 'umap'."}

        # Store reduced coordinates in session.artifacts
        session.artifacts["reduced_coordinates"] = coords

        # Resolve category labels with robust fallback
        category_labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                category_labels = session.dataframe["category_name"].values
            elif "category" in session.dataframe.columns:
                category_labels = session.dataframe["category"].values

        if category_labels is None and session.labels is not None:
            category_labels = session.labels

        # Plot 2D scatter plot
        fig, ax = plt.subplots(figsize=(8, 6))
        if category_labels is not None and len(category_labels) == n_samples:
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                hue=category_labels,
                style=category_labels,
                s=100,
                ax=ax
            )
        else:
            sns.scatterplot(
                x=coords[:, 0],
                y=coords[:, 1],
                s=100,
                ax=ax
            )

        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        plt.tight_layout()

        session.pending_figure = fig

        coords_rounded = np.round(coords, 4).tolist()
        summary = {
            "result_id": session.next_result_id("reduction"),
            "method": method,
            "n_components": n_components,
            "coordinates": coords_rounded,
            "explained_variance_ratio": [round(v, 4) for v in explained_var] if explained_var else None
        }

        session.store_result("reduce_dimensions_tool", {
            "method": method,
            "n_components": n_components,
            "random_state": random_state,
            "perplexity": perplexity,
            "n_neighbors": n_neighbors
        }, summary)

        return summary

    @tool
    def binarize_labels_tool() -> Dict[str, Any]:
        """One-hot encodes the category labels using LabelBinarizer and stores them in session.artifacts['binarized_labels']."""
        from sklearn.preprocessing import LabelBinarizer

        labels = None
        if session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
        elif session.labels is not None:
            labels = session.labels

        if labels is None:
            return {"error": "No category labels found in session. Load a dataset first."}

        lb = LabelBinarizer()
        binarized_matrix = lb.fit_transform(labels)
        classes = lb.classes_.tolist()

        # Store in session.artifacts
        session.artifacts["binarized_labels"] = binarized_matrix

        summary = {
            "result_id": session.next_result_id("binarize_labels"),
            "shape": list(binarized_matrix.shape),
            "classes": classes,
            "binarized_matrix": binarized_matrix.tolist()
        }

        session.store_result("binarize_labels_tool", {}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
