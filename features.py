"""Train-only ingredient presence features for supplied-label modelling.

The encoder deliberately learns its ingredient vocabulary during ``fit``.  This
keeps validation and held-out formulas from contributing feature names or
frequencies.  It accepts the cleaned Python ingredient lists used for training,
a safe literal-list representation, and explicit comma/semicolon-separated
input for local inference.  Comma/semicolon input is a convenience parser, not
an ingredient-labeling or regulatory parser.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .cleaning import canonical_text, parse_ingredients


class IngredientPresenceEncoder(BaseEstimator, TransformerMixin):
    """Encode train-vocabulary ingredient presence plus list length.

    Each selected ingredient contributes a binary (0/1) sparse feature.  The
    final numeric column is ``log1p`` of the number of parsed ingredients,
    including ingredients outside the fitted vocabulary.  Vocabulary selection
    uses training document frequency, with lexical ordering used only to break
    frequency ties deterministically.

    Raw local input is accepted only as a real list, a safe literal list, a
    comma-separated list, a semicolon-separated list, or a single ingredient.
    Free prose should not be treated as a validated INCI ingredient list.
    """

    def __init__(self, max_ingredients: int = 1500):
        self.max_ingredients = max_ingredients

    @staticmethod
    def normalize_input(value) -> list[str]:
        """Safely normalize one ingredient-list input without using ``eval``.

        ``parse_ingredients`` remains the canonical parser for persisted
        literal lists.  Delimiter parsing is intentionally explicit for local
        application input; it does not attempt to infer ingredients from prose.
        """
        if isinstance(value, (list, tuple, np.ndarray)):
            return [canonical_text(item) for item in value if isinstance(item, str) and canonical_text(item)]
        if not isinstance(value, str):
            return []
        stripped = value.strip()
        if not stripped:
            return []
        # A bracketed value is a persisted literal list.  Do not fall back to
        # delimiter parsing when it is malformed.
        if stripped.startswith("[") or stripped.startswith("("):
            return parse_ingredients(stripped)
        if "," in stripped or ";" in stripped:
            return [canonical_text(item) for item in stripped.replace(";", ",").split(",") if canonical_text(item)]
        # Single-item local inference (e.g. ``Phenoxyethanol``) is useful and
        # unambiguous; lengthy prose remains one unsupported token.
        item = canonical_text(stripped)
        return [item] if item else []

    @classmethod
    def _rows(cls, X) -> list[list[str]]:
        if hasattr(X, "tolist"):
            values = X.tolist()
        elif isinstance(X, (str, bytes)):
            values = [X]
        else:
            values = list(X)
        return [cls.normalize_input(value) for value in values]

    def fit(self, X, y=None):
        if not isinstance(self.max_ingredients, int) or self.max_ingredients < 1:
            raise ValueError("max_ingredients must be a positive integer.")
        counts: Counter[str] = Counter()
        for ingredients in self._rows(X):
            counts.update(set(ingredients))
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        self.vocabulary_ = {ingredient: index for index, (ingredient, _) in enumerate(ordered[: self.max_ingredients])}
        self.document_frequency_ = {ingredient: int(counts[ingredient]) for ingredient in self.vocabulary_}
        self.n_features_in_ = 1
        self.feature_names_out_ = np.asarray(
            [f"ingredient={ingredient}" for ingredient in self.vocabulary_] + ["log1p_ingredient_count"],
            dtype=object,
        )
        return self

    def transform(self, X):
        check_is_fitted(self, ["vocabulary_", "feature_names_out_"])
        rows = self._rows(X)
        row_indices: list[int] = []
        col_indices: list[int] = []
        values: list[float] = []
        length_col = len(self.vocabulary_)
        for row_index, ingredients in enumerate(rows):
            for ingredient in set(ingredients):
                column = self.vocabulary_.get(ingredient)
                if column is not None:
                    row_indices.append(row_index)
                    col_indices.append(column)
                    values.append(1.0)
            row_indices.append(row_index)
            col_indices.append(length_col)
            values.append(float(np.log1p(len(ingredients))))
        return sparse.csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(rows), len(self.vocabulary_) + 1),
            dtype=np.float64,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def normalize_ingredient_input(value) -> list[str]:
    """Public convenience wrapper shared by inference validation and tests."""
    return IngredientPresenceEncoder.normalize_input(value)
