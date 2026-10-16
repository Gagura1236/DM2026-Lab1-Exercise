"""Pattern mining tools for frequent itemset analysis using PAMI."""

import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool


def make_tools(session):
    """Factory function to instantiate tools_patterns tools with shared session state."""

    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str = "variance",
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 10,
    ) -> dict:
        """Mines frequent itemsets / terms for a specific category using PAMI algorithms.

        Args:
            category_name: Category to filter session.dataframe by.
            filtering_method: One of 'variance', 'tfidf', or 'term_frequency'.
            algorithm: One of 'fpgrowth', 'topk', or 'maxfpgrowth'.
            min_sup: Minimum support threshold (used by 'fpgrowth' and 'maxfpgrowth').
            k: Top-k count (used by 'topk').

        Returns:
            Dict containing mined patterns and summary metadata.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"status": "error", "message": "session.dataframe is missing or empty."}

        # 1. Filter dataframe to specific category
        cat_df = session.dataframe[session.dataframe["category_name"] == category_name]
        if cat_df.empty:
            return {
                "status": "error",
                "message": f"No documents found for category_name '{category_name}'.",
            }

        texts = cat_df["text"].tolist()

        # 2. Build category-local DTM
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            # Handle empty vocabulary case (e.g., all words are stop words or short)
            X = None

        if X is None or X.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived vectorization",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        feature_names = np.array(vectorizer.get_feature_names_out())
        X_dense = X.toarray()

        # 3. Apply feature filtering
        if filtering_method == "variance":
            variances = np.var(X_dense, axis=0)
            p5, p95 = np.percentile(variances, [5, 95])
            keep_mask = (variances >= p5) & (variances <= p95)
        elif filtering_method == "term_frequency":
            counts = np.sum(X_dense, axis=0)
            p5, p95 = np.percentile(counts, [5, 95])
            keep_mask = (counts >= p5) & (counts <= p95)
        elif filtering_method == "tfidf":
            tfidf = TfidfTransformer().fit_transform(X).toarray()
            mean_tfidf = np.mean(tfidf, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= p20
        else:
            return {
                "status": "error",
                "message": f"Unsupported filtering_method '{filtering_method}'.",
            }

        surviving_features = feature_names[keep_mask]
        surviving_counts = X_dense[:, keep_mask]

        # 4. Handle zero terms surviving edge case
        if len(surviving_features) == 0 or surviving_counts.shape[1] == 0:
            result_id = session.next_result_id("patterns")
            summary = {
                "result_id": result_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "n_patterns": 0,
                "patterns": [],
                "note": "no terms survived filtering",
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 5. Convert to transactional database DataFrame (binary presence/absence or counts > 0)
        binary_df = pd.DataFrame(surviving_counts > 0, columns=surviving_features)

        # Create temporary file for transactional DB output
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".db") as tmp_db:
            db_file = tmp_db.name

        try:
            from PAMI.extras.convert import DF2DB

            converter = DF2DB.DF2DB(binary_df)
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            patterns_list = []

            # 6. Mine patterns with chosen PAMI algorithm
            if algorithm == "fpgrowth":
                from PAMI.frequentPattern.basic import FPGrowth

                miner = FPGrowth.FPGrowth(db_file, minSup=min_sup, sep="\t")
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns_list.append({"pattern": row["Patterns"], "support": int(row["Support"])})

            elif algorithm == "topk":
                from PAMI.frequentPattern.topk import FAE

                miner = FAE.FAE(db_file, k=k, sep="\t")
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns_list.append({"pattern": row["Patterns"], "support": int(row["Support"])})

            elif algorithm == "maxfpgrowth":
                from PAMI.frequentPattern.maximal import MaxFPGrowth

                miner = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup, sep="\t")
                miner.mine()
                df_res = miner.getPatternsAsDataFrame()
                if df_res is not None and not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns_list.append({"pattern": row["Patterns"], "support": int(row["Support"])})
            else:
                return {"status": "error", "message": f"Unsupported algorithm '{algorithm}'."}

        finally:
            if os.path.exists(db_file):
                os.remove(db_file)

        # Store artifacts and result
        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns_list

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "patterns": patterns_list,
        }
        session.store_result("mine_patterns_tool", {"category_name": category_name, "algorithm": algorithm}, summary)
        return summary

    return [mine_patterns_tool]
