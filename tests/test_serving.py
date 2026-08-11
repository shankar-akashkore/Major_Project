"""Persisting a fitted model and serving it in the pipeline.

The three properties worth testing are the three that would fail silently.  A model
must refuse features it was not fitted on, rather than multiplying weights by
whichever numbers arrive.  A model measured at chance must be labelled a stub, not
shipped as a validated predictor.  And a model the serving path cannot compute
features for must fall back loudly, not fail the job or quietly pretend.
"""

from __future__ import annotations

import numpy as np
import pytest
from adml import featureset as FS
from adml import predictor as PR
from adml import serving as S
from adml.split import Observation
from adschema import CameraAngle, Composition, ItemKind, Lighting, MotionIntent, Platform, Vertical
from adschema.annotation import CorpusItem
from adschema.request import AssetRef

# --- Fixtures ----------------------------------------------------------------


def _item(index: int) -> CorpusItem:
    angles = list(CameraAngle)
    return CorpusItem(
        item_id=f"i{index}",
        set_id=f"s{index // 3}",
        kind=ItemKind.IMAGE,
        asset=AssetRef(key=f"x/{index}.png", mime_type="image/png"),
        vertical=Vertical.BEAUTY,
        platform=Platform.INSTAGRAM_REELS,
        seed=index,
        angle=angles[index % len(angles)],
        lighting=list(Lighting)[index % len(Lighting)],
        composition=list(Composition)[index % len(Composition)],
        motion=list(MotionIntent)[index % len(MotionIntent)],
    )


@pytest.fixture
def fitted() -> S.ServedModel:
    """A small real model: measured features, real fit, real card."""
    rng = np.random.default_rng(0)
    items = [_item(i) for i in range(30)]
    measured = {
        it.item_id: {
            "luminance": float(rng.uniform(0.2, 0.8)),
            "contrast": float(rng.uniform(0.1, 0.4)),
            "colorfulness": float(rng.uniform(0.2, 0.7)),
            "sharpness": float(rng.uniform(0.3, 0.9)),
            "focal_concentration": float(rng.uniform(0.05, 0.4)),
        }
        for it in items
    }
    table = FS.build_table(items, measured, include_context=False)
    pipeline = FS.FeaturePipeline.fit(table, list(table.item_ids))
    x = pipeline.transform(table, list(table.item_ids))
    row_of = {item: i for i, item in enumerate(table.item_ids)}

    # Comparisons ordered by one real feature, so the fit has something to learn.
    column = pipeline.kept_names().index("sharpness")
    observations = []
    ids = list(table.item_ids)
    for a in range(0, len(ids) - 1, 2):
        left, right = ids[a], ids[a + 1]
        if x[row_of[left], column] >= x[row_of[right], column]:
            observations.append(Observation(winner=left, loser=right))
        else:
            observations.append(Observation(winner=right, loser=left))

    pairs = PR.encode(observations, row_of)
    ranker = PR.PairwiseRanker.fit(
        x, pairs, config=PR.TrainConfig(max_steps=200), names=pipeline.kept_names()
    )
    card = S.build_card(
        pipeline=pipeline,
        ranker=ranker,
        n_train_observations=len(pairs),
        holdout_accuracy=0.68,
        ceiling_accuracy=0.72,
        holdout_ci_low=0.61,
    )
    return S.ServedModel(pipeline=pipeline, ranker=ranker, card=card), table


# --- Round trip --------------------------------------------------------------


def test_a_saved_model_round_trips_to_identical_scores(fitted, tmp_path):
    model, table = fitted
    path = S.save_model(model, tmp_path / "ranker.npz")
    reloaded = S.load_model(path)

    ids = list(table.item_ids)
    before = model.score_items(table, ids)
    after = reloaded.score_items(table, ids)
    for item in ids:
        assert after[item] == pytest.approx(before[item], abs=1e-10)


def test_the_transform_travels_with_the_weights(fitted, tmp_path):
    """Standardisation statistics are part of the model, not of the caller.

    Refitting the transform at serving time would standardise against whatever three
    candidates happened to be in front of it, so a feature's value would depend on
    its peers rather than on itself.
    """
    model, _ = fitted
    reloaded = S.load_model(S.save_model(model, tmp_path / "r.npz"))
    np.testing.assert_allclose(
        reloaded.pipeline.standardiser.mean, model.pipeline.standardiser.mean
    )
    np.testing.assert_allclose(reloaded.pipeline.standardiser.std, model.pipeline.standardiser.std)
    assert reloaded.pipeline.kept_names() == model.pipeline.kept_names()


def test_a_file_without_provenance_is_refused(tmp_path):
    """A bare npz of weights has unknown feature order, so it cannot be served."""
    path = tmp_path / "bare.npz"
    np.savez(path, **{"head.w1": np.zeros(4)})
    with pytest.raises(S.ModelMismatch, match="provenance|__meta__"):
        S.load_model(path)


def test_a_missing_model_is_absence_not_failure(tmp_path):
    assert S.try_load(tmp_path / "nope.npz") is None
    with pytest.raises(FileNotFoundError, match="train_predictor"):
        S.load_model(tmp_path / "nope.npz")


# --- The mismatch guard ------------------------------------------------------


def test_a_model_refuses_features_it_was_not_fitted_on(fitted):
    """The silent, total failure this check exists to prevent.

    Add a column and every weight after the insertion point applies to the wrong
    feature, while the model goes on returning perfectly plausible numbers.
    """
    model, table = fitted
    extra = FS.FeatureTable(
        item_ids=list(table.item_ids),
        names=[*table.names, "sharpness"],
        matrix=np.column_stack([table.matrix, table.matrix[:, :1]]),
        groups=[*table.groups, FS.FeatureGroup.PHOTOMETRIC],
    )
    with pytest.raises(S.ModelMismatch, match="unexpected|columns"):
        model.score_items(extra, list(table.item_ids))


def test_reordered_columns_are_as_wrong_as_missing_ones(fitted):
    """A reordering is harder to notice and exactly as damaging.

    Every weight still finds a number to multiply and the output stays in range.
    """
    model, table = fitted
    order = [2, 0, 1, 3, 4][: len(table.names)]
    shuffled = FS.FeatureTable(
        item_ids=list(table.item_ids),
        names=[table.names[i] for i in order],
        matrix=table.matrix[:, order],
        groups=[table.groups[i] for i in order],
    )
    with pytest.raises(S.ModelMismatch, match="different order"):
        model.score_items(shuffled, list(table.item_ids))


# --- The stub label ----------------------------------------------------------


def test_a_model_measured_at_chance_is_a_stub():
    """Evaluated and found not to work is a stronger reason to label than never
    evaluated at all. The first model this project saved sat at 0.496."""
    card = S.ModelCard(
        model_version="v",
        trained_at="",
        n_train_observations=100,
        n_features=10,
        n_parameters=10,
        holdout_accuracy=0.496,
        holdout_ci_low=0.441,
    )
    assert not card.beats_chance
    assert card.is_stub
    assert "DOES NOT BEAT CHANCE" in card.summary()


def test_an_unresolved_accuracy_above_chance_is_still_a_stub():
    """The interval's lower bound decides, not the point estimate.

    0.53 over 100 comparisons and 0.53 over 3,000 are different claims, and this
    project does not treat an unresolved difference as a result anywhere else.
    """
    card = S.ModelCard(
        model_version="v",
        trained_at="",
        n_train_observations=100,
        n_features=10,
        n_parameters=10,
        holdout_accuracy=0.58,
        holdout_ci_low=0.47,
    )
    assert card.is_stub


def test_a_resolved_model_on_real_features_is_not_a_stub(fitted):
    model, _ = fitted
    assert model.card.beats_chance
    assert not model.card.is_stub
    # 0.68 against a 0.72 ceiling, both measured from the 0.5 coin-flip floor:
    # (0.68 - 0.5) / (0.72 - 0.5) = 0.82. Reported as a share of achievable so a
    # 0.68 is not read as mediocre when it is most of what is reachable.
    assert "82% of achievable" in model.card.summary()
    assert "DOES NOT BEAT CHANCE" not in model.card.summary()


def test_stand_in_features_make_a_model_a_stub_however_well_it_scores():
    card = S.ModelCard(
        model_version="v",
        trained_at="",
        n_train_observations=1000,
        n_features=40,
        n_parameters=40,
        holdout_accuracy=0.80,
        holdout_ci_low=0.75,
        is_real_features=False,
    )
    assert card.is_stub
    assert "STAND-IN FEATURES" in card.summary()


def test_a_stand_in_cannot_be_relabelled_real_by_the_caller(fitted):
    """``is_real_features`` comes off the fitted objects, never off an argument."""
    model, table = fitted
    stand_in = FS.FeaturePipeline.fit(
        FS.FeatureTable(
            item_ids=list(table.item_ids),
            names=list(table.names),
            matrix=table.matrix,
            groups=list(table.groups),
            is_real=False,
        ),
        list(table.item_ids),
    )
    card = S.build_card(pipeline=stand_in, ranker=model.ranker, n_train_observations=100)
    assert not card.is_real_features


# --- Set-relative scores -----------------------------------------------------


def test_scores_are_reported_as_a_within_set_position(fitted):
    """The pairwise objective fixes no origin, so an absolute reading is a fiction."""
    model, table = fitted
    ids = list(table.item_ids)[:3]
    scores = S.rank_within_set(model, table, ids)

    relative = scores.relative()
    assert min(relative.values()) == pytest.approx(0.0)
    assert max(relative.values()) == pytest.approx(1.0)
    assert scores.order[0] == max(relative, key=lambda k: relative[k])


def test_one_candidate_maps_to_the_middle_not_to_an_extreme(fitted):
    """A single item carries no ranking information, so 1.0 would be a claim."""
    model, table = fitted
    scores = S.rank_within_set(model, table, [table.item_ids[0]])
    assert scores.relative() == {table.item_ids[0]: 0.5}


def test_an_empty_set_scores_to_nothing(fitted):
    model, table = fitted
    assert S.rank_within_set(model, table, []).relative() == {}


# --- Serving inside the pipeline ---------------------------------------------


async def test_the_pipeline_uses_a_trained_model_when_one_matches(
    storage, governor, make_request, tmp_path
):
    """A model fitted on the features the serving path computes is actually served."""
    import adproviders as P
    from adworker import Pipeline
    from adworker.scoring import MODEL_VERSION, score_image_set

    request = make_request()
    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
        ranker=None,
    )
    record = await pipeline.run(request)
    candidates = [c for c in record.result.images if c.passed_gate]

    # Fit on exactly the columns the serving path produces, which is what makes a
    # model servable at all: there is no torch here to compute embedding blocks.
    model = _fit_on_serving_features(candidates, request, storage)

    served = score_image_set(candidates, request, storage, model=model)
    assert served.fallback_reason is None
    assert served.scored_by == model.card.model_version
    for breakdown in served.breakdowns.values():
        assert breakdown.model_version == model.card.model_version
        assert breakdown.model_version != MODEL_VERSION
    # Set-relative, so exactly one candidate sits at each extreme.
    overalls = sorted(b.overall for b in served.breakdowns.values())
    assert overalls[0] == pytest.approx(0.0) and overalls[-1] == pytest.approx(1.0)


async def test_a_model_needing_colab_features_falls_back_loudly(
    storage, governor, make_request, fitted
):
    """The permanent, legitimate mismatch: trained with embeddings, served without.

    There is no torch on the serving machine, so a model fitted on SigLIP columns can
    never be served here. That must not fail the job — and must not pass silently
    either, since a ranking produced by the baseline while a model was configured is
    exactly what goes unnoticed until the write-up.
    """
    import adproviders as P
    from adworker import Pipeline
    from adworker.scoring import MODEL_VERSION

    model, _ = fitted  # fitted on five synthetic columns, not the pipeline's set
    pipeline = Pipeline(
        storage=storage,
        governor=governor,
        image_provider=P.MockImageProvider(storage),
        video_provider=P.MockVideoProvider(storage),
        llm_provider=P.MockLLMProvider(),
        ranker=model,
    )
    record = await pipeline.run(make_request())

    from adschema import JobState

    assert record.state is JobState.COMPLETED, record.error
    warnings = [e for e in record.events if e.state == "warning"]
    assert any("fell back to the heuristic" in e.message for e in warnings), [
        e.message for e in record.events
    ]
    for candidate in record.result.images:
        if candidate.passed_gate:
            assert candidate.score.model_version == MODEL_VERSION
            assert candidate.score.is_stub


def _fit_on_serving_features(candidates, request, storage) -> S.ServedModel:
    """Fit a model on exactly the columns ``score_image_set`` will compute."""
    from adworker.scoring import _corpus_item

    items = [_corpus_item(c, request, ItemKind.IMAGE) for c in candidates]
    measured = {
        it.item_id: FS.image_features(it, storage.get_bytes(it.asset.key), request.theme.palette)
        for it in items
    }
    table = FS.build_table(items, measured)
    pipeline = FS.FeaturePipeline.fit(table, list(table.item_ids))
    x = pipeline.transform(table, list(table.item_ids))
    row_of = {item: i for i, item in enumerate(table.item_ids)}
    ids = list(table.item_ids)
    observations = [Observation(winner=ids[0], loser=ids[-1])]
    ranker = PR.PairwiseRanker.fit(
        x,
        PR.encode(observations, row_of),
        config=PR.TrainConfig(max_steps=50),
        names=pipeline.kept_names(),
    )
    card = S.build_card(
        pipeline=pipeline,
        ranker=ranker,
        n_train_observations=1,
        holdout_accuracy=0.66,
        ceiling_accuracy=0.72,
        holdout_ci_low=0.55,
    )
    return S.ServedModel(pipeline=pipeline, ranker=ranker, card=card)
