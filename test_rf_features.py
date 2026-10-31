from __future__ import annotations

import pickle

import numpy as np
from scipy import sparse
from sklearn.base import clone

from src.features import IngredientPresenceEncoder, normalize_ingredient_input


def test_encoder_safe_literal_and_explicit_raw_list_parsing():
    assert normalize_ingredient_input("[' AQUA ', 'Glycerin']") == ["aqua", "glycerin"]
    assert normalize_ingredient_input("Aqua; Glycerin, Phenoxyethanol") == ["aqua", "glycerin", "phenoxyethanol"]
    assert normalize_ingredient_input("[__import__('os').system('unsafe')]") == []
    assert normalize_ingredient_input("['unterminated'") == []


def test_encoder_uses_train_only_vocabulary_and_unknowns_do_not_expand_it():
    encoder = IngredientPresenceEncoder(max_ingredients=2).fit([["aqua", "glycerin"], ["aqua", "parfum"]])
    assert list(encoder.vocabulary_) == ["aqua", "glycerin"]
    before = dict(encoder.vocabulary_)
    transformed = encoder.transform([["aqua", "previously unseen"]])
    assert dict(encoder.vocabulary_) == before
    assert transformed.shape == (1, 3)
    assert transformed.toarray()[0, before["aqua"]] == 1
    assert transformed.toarray()[0, -1] == np.log1p(2)


def test_encoder_binary_presence_dimension_count_and_feature_names():
    encoder = IngredientPresenceEncoder(max_ingredients=3).fit([["aqua", "aqua", "glycerin"], ["parfum"]])
    matrix = encoder.transform([["aqua", "aqua", "glycerin", "unknown"], []])
    assert sparse.isspmatrix_csr(matrix)
    assert matrix.shape == (2, 4)
    assert matrix.toarray()[0, encoder.vocabulary_["aqua"]] == 1
    assert matrix.toarray()[0, encoder.vocabulary_["glycerin"]] == 1
    assert matrix.toarray()[0, -1] == np.log1p(4)
    assert matrix.toarray()[1, -1] == 0
    assert encoder.get_feature_names_out()[-1] == "log1p_ingredient_count"


def test_encoder_clone_and_pickle_are_compatible():
    encoder = IngredientPresenceEncoder(max_ingredients=7)
    assert clone(encoder).max_ingredients == 7
    fitted = encoder.fit([["aqua"], ["glycerin"]])
    restored = pickle.loads(pickle.dumps(fitted))
    assert restored.get_feature_names_out().tolist() == fitted.get_feature_names_out().tolist()
    assert np.array_equal(restored.transform([["aqua"]]).toarray(), fitted.transform([["aqua"]]).toarray())
