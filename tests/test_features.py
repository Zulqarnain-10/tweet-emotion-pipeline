"""Unit tests for tweet_emotion.features: the vectoriser factory, the npz layout, and the stage.

The stage tests write small processed CSVs to a temporary directory and read back every
output the contract names.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from tweet_emotion import features, settings
from tweet_emotion.features import (
    fit_vectorizer,
    load_npz,
    make_vectorizer,
    save_npz,
    select_vocabulary,
)

MANIFEST_KEYS = {
    "method",
    "max_features",
    "ngram_range",
    "min_df",
    "sublinear_tf",
    "vocabulary_size",
    "n_train",
    "n_test",
    "nnz_train",
    "density_train",
}
TRAIN_TEXTS = [
    "happy sunshine great day",
    "sad tears lonely night",
    "happy happy joy",
    "",
    "lost broken sad",
    "great joy weekend",
]
TEST_TEXTS = ["happy weekend", "sad night zzzquux", ""]


def block(**overrides) -> dict:
    base = {
        "method": "tfidf",
        "max_features": 5000,
        "ngram_range": [1, 2],
        "min_df": 2,
        "sublinear_tf": True,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# make_vectorizer
# ---------------------------------------------------------------------------


def test_tfidf_vectorizer_carries_the_params():
    vectorizer = make_vectorizer(block())
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.max_features == 5000
    assert vectorizer.ngram_range == (1, 2)
    assert isinstance(vectorizer.ngram_range, tuple)
    assert vectorizer.min_df == 2
    assert vectorizer.sublinear_tf is True
    assert vectorizer.lowercase is False
    assert vectorizer.token_pattern == r"(?u)\b\w+\b"
    assert vectorizer.dtype == np.float32


def test_bow_vectorizer_is_a_count_vectorizer():
    vectorizer = make_vectorizer(block(method="bow", sublinear_tf=True))
    assert isinstance(vectorizer, CountVectorizer)
    assert not isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.max_features == 5000
    assert vectorizer.ngram_range == (1, 2)
    assert vectorizer.min_df == 2
    assert vectorizer.lowercase is False
    assert vectorizer.token_pattern == r"(?u)\b\w+\b"
    assert vectorizer.dtype == np.float32


def test_tfidf_without_sublinear():
    vectorizer = make_vectorizer(block(sublinear_tf=False))
    assert vectorizer.sublinear_tf is False


def test_unknown_method_raises():
    with pytest.raises(ValueError):
        make_vectorizer(block(method="word2vec"))


def test_single_letter_tokens_survive():
    vectorizer = make_vectorizer(block(ngram_range=[1, 1], min_df=1))
    vectorizer.fit(["i a b", "u r"])
    assert {"i", "a", "b", "u", "r"} <= set(vectorizer.vocabulary_)


def test_case_is_not_folded_by_the_vectorizer():
    vectorizer = make_vectorizer(block(ngram_range=[1, 1], min_df=1))
    vectorizer.fit(["Hello hello"])
    assert {"Hello", "hello"} <= set(vectorizer.vocabulary_)


def test_transform_output_is_float32_sparse():
    vectorizer = make_vectorizer(block(ngram_range=[1, 2], min_df=1))
    X = vectorizer.fit_transform(TRAIN_TEXTS)
    assert issparse(X)
    assert X.dtype == np.float32
    assert X.shape[0] == len(TRAIN_TEXTS)
    assert X.getrow(3).nnz == 0


def test_bigrams_are_in_the_vocabulary():
    vectorizer = make_vectorizer(block(ngram_range=[1, 2], min_df=1))
    vectorizer.fit(["happy happy joy"])
    assert "happy happy" in vectorizer.vocabulary_
    assert "happy joy" in vectorizer.vocabulary_


def test_max_features_caps_the_vocabulary():
    vectorizer = make_vectorizer(block(ngram_range=[1, 1], min_df=1, max_features=3))
    vectorizer.fit(TRAIN_TEXTS)
    assert len(vectorizer.vocabulary_) == 3


def test_make_vectorizer_accepts_a_fixed_vocabulary():
    vocabulary = {"great": 0, "happy": 1, "joy": 2}
    vectorizer = make_vectorizer(block(ngram_range=[1, 1]), vocabulary=vocabulary)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.vocabulary == vocabulary
    assert vectorizer.vocabulary is not vocabulary
    # max_features and min_df are implied by the vocabulary but stay readable on the object.
    assert vectorizer.max_features == 5000
    assert vectorizer.min_df == 2
    assert vectorizer.sublinear_tf is True
    assert vectorizer.dtype == np.float32
    X = vectorizer.fit_transform(TRAIN_TEXTS)
    assert vectorizer.vocabulary_ == vocabulary
    assert X.shape == (len(TRAIN_TEXTS), 3)
    assert not hasattr(vectorizer, "stop_words_")


# ---------------------------------------------------------------------------
# select_vocabulary / fit_vectorizer
# ---------------------------------------------------------------------------


def test_select_vocabulary_ranks_by_document_frequency_then_term():
    # Document frequency 2: great, happy, joy, sad; every other term appears in one text.
    # "happy" has the highest term frequency (3), which must not matter.
    vocabulary = select_vocabulary(block(ngram_range=[1, 1], min_df=1, max_features=3), TRAIN_TEXTS)
    assert vocabulary == {"great": 0, "happy": 1, "joy": 2}


def test_select_vocabulary_indices_are_alphabetical():
    vocabulary = select_vocabulary(block(ngram_range=[1, 2], min_df=1), TRAIN_TEXTS)
    terms = list(vocabulary)
    assert terms == sorted(terms)
    assert list(vocabulary.values()) == list(range(len(terms)))
    assert "happy happy" in vocabulary
    assert "zzzquux" not in vocabulary


def test_select_vocabulary_honours_min_df():
    vocabulary = select_vocabulary(block(ngram_range=[1, 1], min_df=2), TRAIN_TEXTS)
    assert vocabulary == {"great": 0, "happy": 1, "joy": 2, "sad": 3}


def test_select_vocabulary_without_a_cap_keeps_every_term():
    cfg = block(ngram_range=[1, 1], min_df=1, max_features=None)
    vocabulary = select_vocabulary(cfg, TRAIN_TEXTS)
    plain = CountVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False).fit(TRAIN_TEXTS)
    assert set(vocabulary) == set(plain.vocabulary_)


def test_select_vocabulary_refuses_an_empty_corpus():
    with pytest.raises(ValueError):
        select_vocabulary(block(min_df=1), ["", "   "])


def test_fit_vectorizer_uses_the_selected_vocabulary():
    cfg = block(ngram_range=[1, 2], min_df=1, max_features=8)
    vectorizer = fit_vectorizer(cfg, TRAIN_TEXTS)
    assert isinstance(vectorizer, TfidfVectorizer)
    assert vectorizer.vocabulary_ == select_vocabulary(cfg, TRAIN_TEXTS)
    assert len(vectorizer.vocabulary_) == 8
    assert list(features.feature_names(vectorizer)) == sorted(vectorizer.vocabulary_)
    assert not hasattr(vectorizer, "stop_words_")


def test_fit_vectorizer_bow():
    vectorizer = fit_vectorizer(block(method="bow", ngram_range=[1, 1], min_df=1), TRAIN_TEXTS)
    assert isinstance(vectorizer, CountVectorizer)
    assert not isinstance(vectorizer, TfidfVectorizer)
    X = vectorizer.transform(["happy happy joy"])
    assert X.max() == 2


def test_fit_vectorizer_is_the_same_on_shuffled_copies_of_the_corpus(corpus):
    cfg = block(ngram_range=[1, 2], min_df=1, max_features=60)
    texts = corpus["text"].tolist()
    first = fit_vectorizer(cfg, pd.Series(texts).sample(frac=1.0, random_state=1).tolist())
    second = fit_vectorizer(cfg, pd.Series(texts).sample(frac=1.0, random_state=2).tolist())
    assert first.vocabulary_ == second.vocabulary_
    assert len(first.vocabulary_) == 60
    np.testing.assert_array_equal(first.idf_, second.idf_)
    np.testing.assert_array_equal(
        first.transform(texts[:10]).toarray(), second.transform(texts[:10]).toarray()
    )


def test_fit_vectorizer_matches_the_manual_two_step_fit():
    cfg = block(ngram_range=[1, 2], min_df=1, max_features=8)
    manual = make_vectorizer(cfg, vocabulary=select_vocabulary(cfg, TRAIN_TEXTS)).fit(TRAIN_TEXTS)
    helper = fit_vectorizer(cfg, TRAIN_TEXTS)
    np.testing.assert_allclose(
        helper.transform(TEST_TEXTS).toarray(), manual.transform(TEST_TEXTS).toarray()
    )


# ---------------------------------------------------------------------------
# save_npz / load_npz
# ---------------------------------------------------------------------------


def test_npz_round_trip(tmp_path: Path, tiny_csr):
    X, y = tiny_csr
    path = tmp_path / "features" / "train.npz"
    save_npz(path, X, y)
    assert path.is_file()
    X2, y2 = load_npz(path)
    assert isinstance(X2, csr_matrix)
    assert X2.shape == X.shape
    assert X2.dtype == np.float32
    np.testing.assert_array_equal(X2.toarray(), X.toarray())
    np.testing.assert_array_equal(y2, y)


def test_npz_uses_exactly_the_contract_keys(tmp_path: Path, tiny_csr):
    X, y = tiny_csr
    path = tmp_path / "train.npz"
    save_npz(path, X, y)
    with np.load(path, allow_pickle=False) as archive:
        assert set(archive.files) == set(settings.NPZ_KEYS)
        np.testing.assert_array_equal(archive["data"], X.data)
        np.testing.assert_array_equal(archive["indices"], X.indices)
        np.testing.assert_array_equal(archive["indptr"], X.indptr)
        assert tuple(int(v) for v in archive["shape"]) == X.shape
        np.testing.assert_array_equal(archive["y"], y)


def test_npz_keeps_empty_rows_and_trailing_empty_columns(tmp_path: Path):
    X = csr_matrix(np.zeros((3, 4), dtype=np.float32))
    y = np.array([0, 1, 0])
    save_npz(tmp_path / "z.npz", X, y)
    X2, y2 = load_npz(tmp_path / "z.npz")
    assert X2.shape == (3, 4)
    assert X2.nnz == 0
    np.testing.assert_array_equal(y2, y)


def test_load_npz_reads_the_layout_written_by_hand(tmp_path: Path, tiny_csr, helpers):
    X, y = tiny_csr
    helpers["write_npz"](tmp_path / "hand.npz", X, y)
    X2, y2 = load_npz(tmp_path / "hand.npz")
    np.testing.assert_array_equal(X2.toarray(), X.toarray())
    np.testing.assert_array_equal(y2, y)


def test_save_npz_accepts_a_pandas_label_series(tmp_path: Path, tiny_csr):
    X, y = tiny_csr
    save_npz(tmp_path / "s.npz", X, pd.Series(y))
    _, y2 = load_npz(tmp_path / "s.npz")
    np.testing.assert_array_equal(y2, y)


def test_save_npz_writes_identical_bytes_twice(tmp_path: Path, tiny_csr):
    X, y = tiny_csr
    save_npz(tmp_path / "first.npz", X, y)
    save_npz(tmp_path / "second.npz", X.copy(), y.copy())
    first = (tmp_path / "first.npz").read_bytes()
    assert first == (tmp_path / "second.npz").read_bytes()
    assert len(first) > 0


def test_save_npz_pins_the_zip_member_metadata(tmp_path: Path, tiny_csr):
    """The creating system and file mode differ between Windows and Linux by default, which
    makes identical matrices hash differently; both are fixed along with the timestamp."""
    X, y = tiny_csr
    save_npz(tmp_path / "pinned.npz", X, y)
    with zipfile.ZipFile(tmp_path / "pinned.npz") as archive:
        members = archive.infolist()
        assert [info.filename for info in members] == [f"{k}.npy" for k in settings.NPZ_KEYS]
        for info in members:
            assert info.date_time == (1980, 1, 1, 0, 0, 0), info.filename
            assert info.create_system == 3, info.filename
            assert info.external_attr == 0o100644 << 16, info.filename
            assert info.compress_type == zipfile.ZIP_DEFLATED, info.filename


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


@pytest.fixture
def staged(tmp_path: Path, monkeypatch, toy_params, helpers) -> dict:
    processed = tmp_path / "processed"
    train = pd.DataFrame(
        {"tweet_id": range(1, 7), "text": TRAIN_TEXTS, "label": [1, 0, 1, 0, 0, 1]}
    )
    test = pd.DataFrame({"tweet_id": [11, 12, 13], "text": TEST_TEXTS, "label": [1, 0, 1]})
    helpers["write_csv"](processed / "train.csv", train)
    helpers["write_csv"](processed / "test.csv", test)
    stage_params = {**toy_params, "features": block(min_df=1)}
    features_dir = tmp_path / "features"
    models_dir = tmp_path / "models"
    monkeypatch.setattr(settings, "load_params", lambda path=None: stage_params)
    monkeypatch.setattr(settings, "PROCESSED_DIR", processed)
    monkeypatch.setattr(settings, "PROCESSED_TRAIN_PATH", processed / "train.csv")
    monkeypatch.setattr(settings, "PROCESSED_TEST_PATH", processed / "test.csv")
    monkeypatch.setattr(settings, "FEATURES_DIR", features_dir)
    monkeypatch.setattr(settings, "FEATURES_TRAIN_PATH", features_dir / "train.npz")
    monkeypatch.setattr(settings, "FEATURES_TEST_PATH", features_dir / "test.npz")
    monkeypatch.setattr(settings, "FEATURE_MANIFEST_PATH", features_dir / "feature_manifest.json")
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    monkeypatch.setattr(settings, "VECTORIZER_PATH", models_dir / "vectorizer.joblib")
    return {
        "params": stage_params,
        "features_dir": features_dir,
        "models_dir": models_dir,
        "train": train,
        "test": test,
    }


def test_main_writes_both_matrices_and_the_vectorizer(staged):
    assert features.main([]) == 0
    X_train, y_train = load_npz(staged["features_dir"] / "train.npz")
    X_test, y_test = load_npz(staged["features_dir"] / "test.npz")
    assert X_train.shape[0] == 6
    assert X_test.shape[0] == 3
    assert X_train.shape[1] == X_test.shape[1]
    np.testing.assert_array_equal(y_train, staged["train"]["label"].to_numpy())
    np.testing.assert_array_equal(y_test, staged["test"]["label"].to_numpy())
    assert X_train.dtype == np.float32
    assert (staged["models_dir"] / "vectorizer.joblib").is_file()


def test_main_fits_on_train_only(staged):
    assert features.main([]) == 0
    vectorizer = joblib.load(staged["models_dir"] / "vectorizer.joblib")
    assert "zzzquux" not in vectorizer.vocabulary_
    assert "happy" in vectorizer.vocabulary_


def test_saved_vectorizer_reproduces_the_test_matrix(staged):
    assert features.main([]) == 0
    vectorizer = joblib.load(staged["models_dir"] / "vectorizer.joblib")
    X_test, _ = load_npz(staged["features_dir"] / "test.npz")
    again = vectorizer.transform(TEST_TEXTS)
    np.testing.assert_allclose(again.toarray(), X_test.toarray(), atol=1e-7)


def test_empty_texts_become_empty_rows(staged):
    assert features.main([]) == 0
    X_train, _ = load_npz(staged["features_dir"] / "train.npz")
    X_test, _ = load_npz(staged["features_dir"] / "test.npz")
    assert X_train.getrow(3).nnz == 0
    assert X_test.getrow(2).nnz == 0


def test_feature_manifest_keys_and_values(staged, helpers):
    assert features.main([]) == 0
    manifest = helpers["read_json"](staged["features_dir"] / "feature_manifest.json")
    assert set(manifest) == MANIFEST_KEYS
    vectorizer = joblib.load(staged["models_dir"] / "vectorizer.joblib")
    X_train, _ = load_npz(staged["features_dir"] / "train.npz")
    cfg = staged["params"]["features"]
    assert manifest["method"] == "tfidf"
    assert manifest["max_features"] == cfg["max_features"]
    assert list(manifest["ngram_range"]) == list(cfg["ngram_range"])
    assert manifest["min_df"] == cfg["min_df"]
    assert manifest["sublinear_tf"] is True
    assert manifest["vocabulary_size"] == len(vectorizer.vocabulary_) == X_train.shape[1]
    assert manifest["n_train"] == 6
    assert manifest["n_test"] == 3
    assert manifest["nnz_train"] == X_train.nnz
    density = X_train.nnz / (X_train.shape[0] * X_train.shape[1])
    assert manifest["density_train"] == round(density, 4)


def test_main_is_deterministic(staged):
    assert features.main([]) == 0
    first, _ = load_npz(staged["features_dir"] / "train.npz")
    first_bytes = (staged["features_dir"] / "train.npz").read_bytes()
    assert features.main([]) == 0
    second, _ = load_npz(staged["features_dir"] / "train.npz")
    np.testing.assert_array_equal(first.toarray(), second.toarray())
    assert (staged["features_dir"] / "train.npz").read_bytes() == first_bytes


def test_main_vocabulary_is_alphabetical_and_selected_up_front(staged):
    assert features.main([]) == 0
    vectorizer = joblib.load(staged["models_dir"] / "vectorizer.joblib")
    terms = list(vectorizer.vocabulary_)
    assert terms == sorted(terms)
    assert list(vectorizer.vocabulary_.values()) == list(range(len(terms)))
    cfg = staged["params"]["features"]
    assert vectorizer.vocabulary_ == select_vocabulary(cfg, TRAIN_TEXTS)
    assert vectorizer.vocabulary == vectorizer.vocabulary_
    assert vectorizer.max_features == cfg["max_features"]
    assert not hasattr(vectorizer, "stop_words_")


def test_main_with_bow(staged, monkeypatch, helpers):
    bow = {**staged["params"], "features": block(method="bow", min_df=1, ngram_range=[1, 1])}
    monkeypatch.setattr(settings, "load_params", lambda path=None: bow)
    assert features.main([]) == 0
    vectorizer = joblib.load(staged["models_dir"] / "vectorizer.joblib")
    assert isinstance(vectorizer, CountVectorizer)
    assert not isinstance(vectorizer, TfidfVectorizer)
    X_train, _ = load_npz(staged["features_dir"] / "train.npz")
    assert X_train.max() == 2  # "happy happy joy" counts happy twice
    manifest = helpers["read_json"](staged["features_dir"] / "feature_manifest.json")
    assert manifest["method"] == "bow"
    assert not manifest["sublinear_tf"]


def test_main_fails_without_the_processed_split(tmp_path: Path, monkeypatch, toy_params):
    monkeypatch.setattr(settings, "load_params", lambda path=None: toy_params)
    monkeypatch.setattr(settings, "PROCESSED_TRAIN_PATH", tmp_path / "nope" / "train.csv")
    monkeypatch.setattr(settings, "PROCESSED_TEST_PATH", tmp_path / "nope" / "test.csv")
    try:
        code = features.main([])
    except (OSError, ValueError):
        return
    assert code != 0
