"""Predictor, feature-assembly and embedding tests.

Two kinds of assertion here.  The recovery tests check that the fitting machinery
finds signal that is genuinely present.  The *negative* tests are the more important
half: they check that the machinery finds nothing when there is nothing, because a
ranker that reports accuracy on noise is worse than one that reports none.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
from adml.embeddings import EmbeddingSet, EmbeddingSpec, hash_embeddings, missing_blocks
from adml.featureset import (
    BlockPCA,
    FeatureGroup,
    FeaturePipeline,
    FeatureTable,
    Standardiser,
    build_table,
    context_features,
    parameter_budget,
)
from adml.predictor import (
    L2_GRID,
    PairwiseRanker,
    StrengthRegressor,
    TrainConfig,
    encode,
    fit_with_selection,
    random_scores,
    single_feature_scores,
)
from adml.split import Observation, cross_validate_folds
from adschema import AssetRef, CameraAngle, Lighting, Platform, Tier, Vertical
from adschema.annotation import CorpusItem, ItemKind, PairKind


def item(item_id: str, set_id: str, **kwargs) -> CorpusItem:
    return CorpusItem(
        item_id=item_id,
        set_id=set_id,
        kind=ItemKind.IMAGE,
        asset=AssetRef(key=f"k/{item_id}.png", content_type="image/png"),
        tier=Tier.MOCK,
        provider="mock",
        model="m",
        vertical=Vertical.BEAUTY,
        platform=Platform.INSTAGRAM_REELS,
        **kwargs,
    )


def synthetic(n_items: int = 180, d: int = 12, seed: int = 0, noise: float = 1.0):
    """Items whose true quality is a known linear function of their features."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i // 3:03d}-i{i % 3}" for i in range(n_items)]
    set_of = {i: i.split("-")[0] for i in ids}
    x = rng.normal(size=(n_items, d))
    weights = np.zeros(d)
    weights[:4] = [1.6, -1.1, 0.8, 0.5]
    truth = x @ weights
    row = {item_id: i for i, item_id in enumerate(ids)}

    obs: list[Observation] = []
    for _ in range(6):
        order = list(ids)
        rng.shuffle(order)
        for k in range(0, len(order) - 1, 2):
            a, b = order[k], order[k + 1]
            margin = (truth[row[a]] - truth[row[b]]) / noise
            kind = PairKind.WITHIN_SET if set_of[a] == set_of[b] else PairKind.CROSS_SET
            if rng.random() < 1.0 / (1.0 + math.exp(-margin)):
                obs.append(Observation(a, b, kind))
            else:
                obs.append(Observation(b, a, kind))
    return ids, set_of, x, weights, row, obs


# --- Encoding ---------------------------------------------------------------


def test_a_tie_becomes_two_half_weight_observations_in_both_directions():
    pairs = encode([Observation("a", "b", is_tie=True)], {"a": 0, "b": 1})
    assert len(pairs) == 2
    assert pairs.total_weight == pytest.approx(1.0)
    assert sorted(zip(pairs.winner.tolist(), pairs.loser.tolist(), strict=True)) == [
        (0, 1),
        (1, 0),
    ]


def test_observations_naming_unknown_items_are_skipped_not_raised():
    pairs = encode([Observation("a", "ghost")], {"a": 0, "b": 1})
    assert len(pairs) == 0


# --- Recovery ---------------------------------------------------------------


def test_the_ranker_recovers_a_known_weight_direction():
    _, _, x, weights, row, obs = synthetic()
    model = PairwiseRanker.fit(x, encode(obs, row), config=TrainConfig(max_steps=1500))
    cosine = float(model.w1 @ weights / (np.linalg.norm(model.w1) * np.linalg.norm(weights)))
    assert cosine > 0.9


def test_the_ranker_generalises_to_held_out_sets():
    ids, set_of, x, _, row, obs = synthetic()
    accuracies = [
        PairwiseRanker.fit(x, encode(f.train, row), config=TrainConfig(max_steps=1200)).accuracy(
            x, encode(f.test, row)
        )
        for f in cross_validate_folds(obs, set_of, k=5)
    ]
    assert float(np.mean(accuracies)) > 0.70
    del ids


def test_the_ranker_finds_nothing_in_features_that_carry_nothing():
    """The control that makes every positive result above mean something."""
    ids, set_of, _, _, row, obs = synthetic()
    noise = np.random.default_rng(99).normal(size=(len(ids), 12))
    accuracies = [
        PairwiseRanker.fit(noise, encode(f.train, row), config=TrainConfig(max_steps=800)).accuracy(
            noise, encode(f.test, row)
        )
        for f in cross_validate_folds(obs, set_of, k=5)
    ]
    assert float(np.mean(accuracies)) == pytest.approx(0.5, abs=0.06)


def test_a_stand_in_embedding_produces_no_held_out_signal():
    """Hash embeddings must be useless, or they are not a safe placeholder."""
    ids, set_of, _, _, row, obs = synthetic(n_items=120)
    block = hash_embeddings(ids, dim=24).l2_normalised()
    x, coverage = block.matrix(ids)
    assert not coverage.missing
    accuracies = [
        PairwiseRanker.fit(x, encode(f.train, row), config=TrainConfig(max_steps=600)).accuracy(
            x, encode(f.test, row)
        )
        for f in cross_validate_folds(obs, set_of, k=5)
    ]
    assert float(np.mean(accuracies)) == pytest.approx(0.5, abs=0.08)


# --- Numerics and gauge freedom ---------------------------------------------


def test_the_loss_survives_extreme_margins():
    """A confident model must not produce inf or nan in the loss or the gradient."""
    x = np.array([[40.0], [-40.0]])
    model = PairwiseRanker(config=TrainConfig(), w1=np.array([25.0]), b1=None, w2=None)
    pairs = encode([Observation("a", "b"), Observation("b", "a")], {"a": 0, "b": 1})
    loss, grad = model._loss_and_grad_scores(model.scores(x), pairs)
    assert math.isfinite(loss)
    assert np.isfinite(grad).all()


def test_scores_are_only_defined_up_to_a_constant():
    """No output bias exists, because pairwise data cannot identify one."""
    _, _, x, _, row, obs = synthetic(n_items=60, d=6)
    model = PairwiseRanker.fit(x, encode(obs, row), config=TrainConfig(max_steps=400))
    shifted = model.scores(x) + 7.0
    original = model.scores(x)
    assert np.allclose(np.diff(shifted), np.diff(original))


def test_a_linear_fit_is_reproducible_from_its_seed():
    _, _, x, _, row, obs = synthetic(n_items=90, d=8)
    config = TrainConfig(max_steps=300, seed=5)
    a = PairwiseRanker.fit(x, encode(obs, row), config=config)
    b = PairwiseRanker.fit(x, encode(obs, row), config=config)
    assert np.allclose(a.w1, b.w1)


def test_coefficients_are_refused_for_the_nonlinear_model():
    _, _, x, _, row, obs = synthetic(n_items=60, d=6)
    model = PairwiseRanker.fit(x, encode(obs, row), config=TrainConfig(hidden=4, max_steps=100))
    assert model.n_parameters > x.shape[1]
    with pytest.raises(ValueError, match="only interpretable"):
        model.coefficients()


# --- Selection ---------------------------------------------------------------


def test_selection_averages_over_inner_folds_and_picks_from_the_grid():
    ids, set_of, x, _, row, obs = synthetic(n_items=150, d=10)
    outer = cross_validate_folds(obs, set_of, k=5)[0]
    train_items = [i for i in ids if set_of[i] in set(outer.train_sets)]
    inner = [
        (encode(f.train, row), encode(f.test, row))
        for f in cross_validate_folds(outer.train, {i: set_of[i] for i in train_items}, k=3)
    ]
    model, selection = fit_with_selection(
        x, encode(outer.train, row), inner, base=TrainConfig(max_steps=400)
    )
    assert selection.best.l2 in L2_GRID
    assert len(selection.scores) == len(L2_GRID)
    assert model.config.l2 == selection.best.l2


def test_a_heavier_penalty_shrinks_the_weights():
    _, _, x, _, row, obs = synthetic(n_items=120, d=10)
    light = PairwiseRanker.fit(x, encode(obs, row), config=TrainConfig(l2=1e-4, max_steps=800))
    heavy = PairwiseRanker.fit(x, encode(obs, row), config=TrainConfig(l2=10.0, max_steps=800))
    assert np.linalg.norm(heavy.w1) < np.linalg.norm(light.w1)


# --- Baselines --------------------------------------------------------------


def test_random_scores_are_seeded_and_cover_every_item():
    ids = [f"i{i}" for i in range(20)]
    assert random_scores(ids, seed=3) == random_scores(ids, seed=3)
    assert set(random_scores(ids, seed=3)) == set(ids)


def test_single_feature_scoring_rejects_an_unknown_column():
    with pytest.raises(KeyError, match="not a column"):
        single_feature_scores(["a", "b"], np.zeros((2, 2)), ["x", "y"], "missing")


def test_the_strength_regressor_learns_the_same_direction_as_the_ranker():
    ids, _, x, weights, row, obs = synthetic(n_items=150, d=10)
    regressor = StrengthRegressor.fit(x, ids, obs)
    cosine = float(
        regressor.weights @ weights / (np.linalg.norm(regressor.weights) * np.linalg.norm(weights))
    )
    assert cosine > 0.8
    del row


# --- Feature assembly -------------------------------------------------------


def test_every_registered_column_has_a_group():
    ctx = context_features(item("a", "s0", angle=CameraAngle.LOW_ANGLE, lighting=Lighting.HIGH_KEY))
    table = build_table([item("a", "s0")], {"a": {"contrast": 0.2, "sharpness": 0.7}})
    assert set(table.groups) <= set(FeatureGroup)
    assert all(v in (0.0, 1.0) for v in ctx.values())


def test_one_hot_encoding_marks_exactly_one_level_per_axis():
    ctx = context_features(item("a", "s0", angle=CameraAngle.PROFILE))
    angle_columns = {k: v for k, v in ctx.items() if k.startswith("angle=")}
    assert sum(angle_columns.values()) == 1.0
    assert angle_columns["angle=profile"] == 1.0


def test_an_absent_design_axis_leaves_every_level_at_zero():
    """A missing axis must not silently become the first level."""
    ctx = context_features(item("a", "s0"))
    assert sum(v for k, v in ctx.items() if k.startswith("angle=")) == 0.0


def test_items_without_measurements_are_dropped_rather_than_zero_filled():
    items = [item("a", "s0"), item("b", "s0")]
    table = build_table(items, {"a": {"contrast": 0.3}})
    assert table.item_ids == ["a"]


def test_items_missing_from_an_embedding_block_are_dropped():
    items = [item("a", "s0"), item("b", "s0")]
    measured = {"a": {"contrast": 0.3}, "b": {"contrast": 0.4}}
    block = hash_embeddings(["a"], dim=4)
    table = build_table(items, measured, embeddings={"siglip": block})
    assert table.item_ids == ["a"]


def test_decoys_never_enter_the_feature_table():
    items = [item("a", "s0"), item("d", "s0", degraded_from="a", degradation="blur")]
    table = build_table(items, {"a": {"contrast": 0.3}, "d": {"contrast": 0.1}})
    assert table.item_ids == ["a"]


def test_stand_in_provenance_propagates_into_the_table():
    items = [item("a", "s0")]
    table = build_table(
        items, {"a": {"contrast": 0.3}}, embeddings={"siglip": hash_embeddings(["a"])}
    )
    assert table.is_real is False


def test_a_real_block_keeps_the_table_real():
    spec = EmbeddingSpec(name="siglip", dim=2, source="google/siglip-base-patch16-224")
    block = EmbeddingSet(spec, {"a": np.array([0.3, 0.4])})
    table = build_table([item("a", "s0")], {"a": {"contrast": 0.3}}, embeddings={"siglip": block})
    assert table.is_real is True


def test_ablation_removes_exactly_one_group():
    items = [item(f"i{i}", f"s{i // 3}") for i in range(9)]
    measured = {
        f"i{i}": {"contrast": 0.1 * i, "thirds_alignment": 0.2 * i, "sharpness": 0.05 * i}
        for i in range(9)
    }
    table = build_table(items, measured)
    before = table.group_sizes()
    reduced = table.without(FeatureGroup.PHOTOMETRIC)
    assert FeatureGroup.PHOTOMETRIC not in reduced.group_sizes()
    assert reduced.matrix.shape[1] == table.matrix.shape[1] - before[FeatureGroup.PHOTOMETRIC]
    assert reduced.item_ids == table.item_ids


# --- Transforms -------------------------------------------------------------


def test_constant_columns_are_dropped_and_named():
    matrix = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
    std = Standardiser.fit(matrix, ["varies", "constant"])
    assert std.dropped_names == ["constant"]
    assert std.transform(matrix).shape == (3, 1)


def test_indicator_columns_are_not_rescaled_by_their_rarity():
    """A one-hot on for 1 of 20 items must not arrive with 4x the leverage."""
    rare = np.zeros((20, 1))
    rare[0, 0] = 1.0
    common = np.zeros((20, 1))
    common[:10, 0] = 1.0
    matrix = np.hstack([rare, common])
    out = Standardiser.fit(matrix, ["rare", "common"]).transform(matrix)
    assert out.max() == pytest.approx(1.0)
    assert abs(out).max() == pytest.approx(1.0)


def test_continuous_columns_are_scaled_to_unit_variance():
    matrix = np.array([[1.0], [2.0], [3.0], [4.0]])
    out = Standardiser.fit(matrix, ["x"]).transform(matrix)
    assert out.std() == pytest.approx(1.0)


def test_pca_is_fitted_on_training_rows_only():
    rng = np.random.default_rng(4)
    train = rng.normal(size=(40, 12))
    pca = BlockPCA.fit(train, k=3)
    before = pca.components.copy()
    pca.transform(rng.normal(size=(10, 12)) * 100.0)
    assert np.array_equal(pca.components, before)


def test_pca_keeps_the_leading_variance():
    rng = np.random.default_rng(5)
    latent = rng.normal(size=(80, 2))
    mixing = rng.normal(size=(2, 20))
    data = latent @ mixing + rng.normal(size=(80, 20)) * 0.01
    pca = BlockPCA.fit(data, k=2)
    assert pca.explained.sum() > 0.98


def test_the_pipeline_never_reads_a_test_row():
    items = [item(f"i{i}", f"s{i // 3}") for i in range(30)]
    rng = np.random.default_rng(6)
    measured = {
        it.item_id: {"contrast": float(rng.random()), "sharpness": float(rng.random())}
        for it in items
    }
    table = build_table(items, measured)
    train = table.item_ids[:21]
    fitted = FeaturePipeline.fit(table, train)
    expected = Standardiser.fit(table.rows(train), fitted.names)
    assert np.allclose(fitted.standardiser.std, expected.std)


def test_the_parameter_budget_scales_with_the_labels():
    assert parameter_budget(1200) == 80
    assert parameter_budget(300) == 20
    assert parameter_budget(0) >= 1


# --- Embedding round-trip ---------------------------------------------------


def test_an_embedding_set_round_trips_through_npz(tmp_path):
    spec = EmbeddingSpec(name="dinov2", dim=3, source="facebook/dinov2-base")
    block = EmbeddingSet(spec, {"a": np.array([0.1, 0.2, 0.3]), "b": np.array([1.0, 0.0, -1.0])})
    path = block.save_npz(tmp_path / "dinov2.npz")
    loaded = EmbeddingSet.load_npz(path)
    assert loaded.spec == spec
    assert np.allclose(loaded.vectors["b"], block.vectors["b"], atol=1e-6)


def test_the_notebook_write_format_loads_without_the_repo(tmp_path):
    """The Colab boundary contract, written the way the notebook writes it.

    ``notebooks/colab_embeddings.ipynb`` cannot import this package — Colab has the
    images, not the checkout — so it constructs the npz by hand.  This reproduces
    that construction exactly.  If the two ever drift, an afternoon of GPU time
    produces files the laptop refuses, and the failure would appear far from its
    cause.
    """
    spec = json.dumps(
        {
            "name": "siglip",
            "dim": 3,
            "source": "google/siglip-base-patch16-224",
            "is_real": True,
            "l2_normalised": False,
        }
    )
    np.savez_compressed(
        tmp_path / "siglip.npz",
        **{
            "item-a": np.array([0.1, 0.2, 0.3], dtype=np.float32),
            "item-b": np.array([0.4, 0.5, 0.6], dtype=np.float32),
            "__spec__": np.frombuffer(spec.encode(), dtype=np.uint8),
        },
    )
    loaded = EmbeddingSet.load_npz(tmp_path / "siglip.npz")
    assert loaded.spec.name == "siglip"
    assert loaded.spec.dim == 3
    assert loaded.spec.is_real is True
    assert sorted(loaded.vectors) == ["item-a", "item-b"]


def test_an_npz_without_provenance_is_refused(tmp_path):
    path = tmp_path / "bare.npz"
    np.savez(path, a=np.zeros(4))
    with pytest.raises(ValueError, match="provenance"):
        EmbeddingSet.load_npz(path)


def test_a_wrong_width_vector_is_refused_at_construction():
    spec = EmbeddingSpec(name="siglip", dim=4, source="x")
    with pytest.raises(ValueError, match="expected"):
        EmbeddingSet(spec, {"a": np.zeros(3)})


def test_coverage_reports_missing_items_instead_of_filling_them():
    block = hash_embeddings(["a", "b"], dim=4)
    rows, coverage = block.matrix(["a", "b", "c"])
    assert coverage.missing == ["c"]
    assert rows.shape == (2, 4)
    assert coverage.fraction == pytest.approx(2 / 3)


def test_l2_normalisation_makes_every_vector_unit_length():
    block = hash_embeddings(["a", "b", "c"], dim=8).l2_normalised()
    for vec in block.vectors.values():
        assert float(np.linalg.norm(vec)) == pytest.approx(1.0)
    assert block.spec.l2_normalised is True


def test_hash_embeddings_are_deterministic_and_flagged_unreal():
    a = hash_embeddings(["x"], dim=6)
    b = hash_embeddings(["x"], dim=6)
    assert np.array_equal(a.vectors["x"], b.vectors["x"])
    assert a.spec.is_real is False


def test_missing_blocks_lists_the_encoders_that_have_not_run():
    assert "dinov2" in missing_blocks({})
    assert "siglip" not in missing_blocks({"siglip": hash_embeddings(["a"])})


def test_an_empty_table_is_shaped_not_crashed():
    table = build_table([], {})
    assert isinstance(table, FeatureTable)
    assert table.matrix.shape[0] == 0
