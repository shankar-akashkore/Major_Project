"""Train and evaluate the image-stage performance predictor.

    # the real thing, once annotations exist
    .venv/bin/python scripts/train_predictor.py

    # with the Colab embedding blocks
    .venv/bin/python scripts/train_predictor.py --embeddings fixtures/corpus/embeddings

    # exercise the whole path on deterministic stand-in embeddings (not a result)
    .venv/bin/python scripts/train_predictor.py --hash-embeddings

Everything is cross-validated over *generation sets*, never over comparisons.  The
reason is measured rather than asserted: on features that are uninformative by
construction, a random split over comparisons reports 0.659 pairwise accuracy and a
set-wise split reports 0.465.  Nineteen points of a completely fictional result
(see ``docs/prediction-protocol.md``).

The output has four parts, in the order the report needs them:

1. **Data and split** — how many comparisons survive an honest split, and the
   interval that sample size implies.  If the interval is wider than the gaps
   between models, nothing below can be resolved and that is the finding.
2. **Ceiling** — what accuracy a perfect predictor could reach against single
   noisy human judgements.  Every accuracy is printed as a fraction of it.
3. **Models** — the trained head against every baseline, each as a *paired*
   comparison on the same held-out comparisons.
4. **Ablation** — one row per feature group, each paired against the full model.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for package in ("schema", "providers", "ml"):
    sys.path.insert(0, str(ROOT / "packages" / package))
sys.path.insert(0, str(ROOT / "services" / "api"))

import adproviders as P  # noqa: E402
from adapi.annotation_store import AnnotationStore  # noqa: E402
from adml import embeddings as EM  # noqa: E402
from adml import evaluate as EV  # noqa: E402
from adml import featureset as FS  # noqa: E402
from adml import predictor as PR  # noqa: E402
from adml.split import Observation, cross_validate_folds, split_report  # noqa: E402
from adschema import Choice, ItemKind, PairKind  # noqa: E402

DEFAULT_FEATURES = ROOT / "fixtures" / "corpus" / "features.json"
DEFAULT_EMBEDDINGS = ROOT / "fixtures" / "corpus" / "embeddings"


# --- Loading -----------------------------------------------------------------


async def _load_labels(store: AnnotationStore) -> tuple[list[Observation], EV.Ceiling]:
    """Observations for training, and the ceiling from the repeats.

    Only ``showing == 0`` judgements become training observations.  A deliberate
    repeat is the *same* person answering the *same* question a second time, partly
    from memory; counting it as an independent observation would double-weight that
    pair and overstate how much data exists.  The repeats are worth more as the
    measurement of how noisy a single judgement is, which is what bounds every
    accuracy reported below.
    """
    pairs = {p.pair_id: p for p in await store.list_pairs()}
    judgements = await store.list_judgements()

    observations: list[Observation] = []
    first_choice: dict[tuple[str, str], str] = {}
    repeat_agreements: list[float] = []
    by_pair: dict[str, dict[str, str]] = defaultdict(dict)

    for j in judgements:
        pair = pairs.get(j.pair_id)
        if pair is None or pair.kind is PairKind.CATCH:
            continue
        label = "tie" if j.choice is Choice.TIE else (j.winner(pair) or "tie")
        key = (j.annotator_id, j.pair_id)

        if j.showing == 0:
            first_choice[key] = label
            by_pair[j.pair_id][j.annotator_id] = label
            if j.choice is Choice.TIE:
                observations.append(
                    Observation(
                        winner=pair.item_a,
                        loser=pair.item_b,
                        kind=pair.kind,
                        annotator_id=j.annotator_id,
                        is_tie=True,
                    )
                )
            else:
                winner = j.winner(pair)
                assert winner is not None
                observations.append(
                    Observation(
                        winner=winner,
                        loser=pair.other(winner),
                        kind=pair.kind,
                        annotator_id=j.annotator_id,
                    )
                )
        elif key in first_choice:
            repeat_agreements.append(1.0 if first_choice[key] == label else 0.0)

    multi = [list(v.values()) for v in by_pair.values() if len(v) >= 2]
    return observations, EV.measure_ceiling(repeat_agreements, multi)


def _load_features(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        raise SystemExit(
            f"{path} not found. Measure the corpus first:\n"
            "  .venv/bin/python scripts/extract_features.py"
        )
    with path.open() as handle:
        return json.load(handle)["features"]


def _load_embeddings(
    args: argparse.Namespace, item_ids: Sequence[str]
) -> dict[str, EM.EmbeddingSet]:
    if args.hash_embeddings:
        return {"siglip": EM.hash_embeddings(item_ids, dim=args.hash_dim)}
    directory = Path(args.embeddings)
    if not directory.exists():
        return {}
    return EM.load_directory(directory)


# --- Scoring one fold --------------------------------------------------------


def _fold_scores(
    table: FS.FeatureTable,
    train_items: list[str],
    test_items: list[str],
    train_obs: list[Observation],
    set_of: dict[str, str],
    *,
    config: PR.TrainConfig,
    embedding_dim: int,
    seed: int,
    inner_k: int = 3,
) -> tuple[dict[str, dict[str, float]], PR.Selection | None, FS.FeaturePipeline]:
    """Fit every scorer on one fold's training sets and score its test items.

    All scorers are built inside the same fold so they see identical training data
    and identical feature scaling.  That is what makes the paired comparisons
    downstream legitimate: any difference between two rows of the results table is
    a difference between the scorers, not between the folds they happened to get.
    """
    pipeline = FS.FeaturePipeline.fit(table, train_items, embedding_dim=embedding_dim)
    x_train = pipeline.transform(table, train_items)
    x_test = pipeline.transform(table, test_items)
    names = pipeline.kept_names()

    train_row = {item: i for i, item in enumerate(train_items)}

    # Inner cross-validation over the *training* sets only, for choosing the
    # penalty. Several folds rather than one: a single inner holdout of ~5 sets
    # yields ~20 comparisons, which selects a hyper-parameter no better than a coin.
    inner_set_of = {i: set_of[i] for i in train_items}
    inner_folds = [
        (PR.encode(f.train, train_row), PR.encode(f.test, train_row))
        for f in cross_validate_folds(train_obs, inner_set_of, k=inner_k, seed=seed + 1)
    ]

    scores: dict[str, dict[str, float]] = {}
    selection: PR.Selection | None = None

    if any(len(a) and len(b) for a, b in inner_folds):
        model, selection = PR.fit_with_selection(
            x_train,
            PR.encode(train_obs, train_row),
            inner_folds,
            base=config,
            names=names,
        )
        scores["pairwise (trained)"] = model.score_items(x_test, test_items)

    scores["random"] = PR.random_scores(test_items, seed=seed)
    for feature, sign, label in (
        ("focal_concentration", 1.0, "salience-only"),
        ("sharpness", 1.0, "sharpness-only"),
        ("aesthetic[0]", 1.0, "aesthetic-only"),
    ):
        if feature in names:
            scores[label] = PR.single_feature_scores(names, x_test, test_items, feature, sign=sign)

    regressor = PR.StrengthRegressor.fit(x_train, train_items, train_obs, l2=1.0, names=names)
    scores["BT-target ridge"] = regressor.score_items(x_test, test_items)

    return scores, selection, pipeline


def _cross_validate(
    table: FS.FeatureTable,
    observations: list[Observation],
    set_of: dict[str, str],
    *,
    k: int,
    config: PR.TrainConfig,
    embedding_dim: int,
    seed: int,
    verbose: bool = True,
) -> tuple[dict[str, EV.Evaluation], int]:
    """Run every fold and pool the held-out predictions.

    Returns the pooled evaluations and the number of columns that actually reached
    the model.  That is not the table's column count: constant columns are dropped
    per fold, so a group whose every column is constant produces an ablation row
    identical to the full model, and reporting the table width there would imply a
    comparison that never happened.
    """
    folds = cross_validate_folds(observations, set_of, k=k, seed=seed)
    in_table = set(table.item_ids)

    pooled_obs: list[Observation] = []
    pooled_scores: dict[str, list[float]] = defaultdict(list)
    pooled_items: dict[str, dict[str, float]] = defaultdict(dict)
    fitted_features = 0

    for fold in folds:
        test_sets = set(fold.test_sets)
        train_items = [i for i in table.item_ids if set_of.get(i) not in test_sets]
        test_items = [i for i in table.item_ids if set_of.get(i) in test_sets]
        usable_test = [o for o in fold.test if o.winner in in_table and o.loser in in_table]
        if not train_items or not test_items or not usable_test:
            if verbose:
                print(f"  {fold.summary()}   SKIPPED (nothing testable in the table)")
            continue

        scores, selection, pipeline = _fold_scores(
            table,
            train_items,
            test_items,
            [o for o in fold.train if o.winner in in_table and o.loser in in_table],
            set_of,
            config=config,
            embedding_dim=embedding_dim,
            seed=seed + fold.index,
        )
        fitted_features = pipeline.n_features
        if verbose:
            note = selection.summary() if selection else "no inner split"
            print(f"  {fold.summary()}  {pipeline.n_features} features  {note}")
            if pipeline.standardiser.dropped_names:
                dropped = pipeline.standardiser.dropped_names
                print(f"      dropped {len(dropped)} constant columns: {', '.join(dropped[:4])}...")

        start = len(pooled_obs)
        pooled_obs.extend(usable_test)
        for name, per_item in scores.items():
            # Pad any scorer that did not run on this fold, so every scorer stays
            # index-aligned with the pooled observation list.
            while len(pooled_scores[name]) < start:
                pooled_scores[name].append(float("nan"))
            pooled_scores[name].extend(EV.credit(per_item, usable_test))
            pooled_items[name].update(per_item)

    total = len(pooled_obs)
    out: dict[str, EV.Evaluation] = {}
    for name, values in pooled_scores.items():
        padded = values + [float("nan")] * (total - len(values))
        out[name] = EV.Evaluation(
            name=name,
            observations=pooled_obs,
            credit=np.array(padded, dtype=float),
            item_scores=pooled_items[name],
            is_real=table.is_real,
        )
    return out, fitted_features


# --- Reporting ---------------------------------------------------------------


def _print_models(results: dict[str, EV.Evaluation], ceiling: float) -> None:
    order = sorted(results, key=lambda n: -results[n].accuracy)
    print("\n3. MODELS  (pooled over folds, paired against the trained model)")
    print("-" * 78)
    for name in order:
        print("  " + results[name].summary(ceiling))

    trained = results.get("pairwise (trained)")
    if trained is None:
        print("\n  The trained model did not run — no usable inner validation split.")
        return
    print("\n  paired differences vs the trained model:")
    for name in order:
        if name == trained.name:
            continue
        print("    " + EV.paired_difference(trained, results[name]).summary())


def _print_kinds(results: dict[str, EV.Evaluation]) -> None:
    print("\n   accuracy by comparison kind (within-set is the deployment task):")
    for name in sorted(results):
        parts = [
            f"{kind} {acc:.3f} (n={n})"
            for kind, (acc, n) in sorted(results[name].accuracy_by_kind().items())
        ]
        print(f"    {name:<22} {'  '.join(parts)}")


def _print_ablation(
    table: FS.FeatureTable,
    observations: list[Observation],
    set_of: dict[str, str],
    full: EV.Evaluation,
    full_features: int,
    *,
    k: int,
    config: PR.TrainConfig,
    embedding_dim: int,
    seed: int,
) -> None:
    print("\n4. ABLATION  (* = difference resolved at 95%)")
    print("-" * 78)
    variants: list[tuple[str, int, EV.Evaluation]] = []
    for group in table.present_groups():
        reduced = table.without(group)
        if reduced.matrix.shape[1] == 0:
            continue
        results, n_fitted = _cross_validate(
            reduced,
            observations,
            set_of,
            k=k,
            config=config,
            embedding_dim=embedding_dim,
            seed=seed,
            verbose=False,
        )
        variant = results.get("pairwise (trained)")
        if variant is not None:
            variants.append((f"minus {group.value}", n_fitted, variant))

    print(f"  {'variant':<26} {'cols':>4}  accuracy [95% CI]      delta")
    for row in EV.ablate(full, variants, n_features=full_features):
        print("  " + row.summary())
    print(
        "\n  A row without a * is not evidence that the group does not matter — at "
        "this\n  corpus size most single groups cannot be resolved either way."
    )


# --- Main --------------------------------------------------------------------


async def _run(args: argparse.Namespace) -> int:
    settings = P.Settings()
    store = AnnotationStore.from_url(settings.database_url)
    await store.create_all()

    items = await store.list_items(kind=ItemKind(args.kind), include_decoys=False)
    observations, ceiling = await _load_labels(store)
    await store.engine.dispose()

    if not items:
        print("No corpus items. Generate and build first:")
        print("  .venv/bin/python scripts/generate_corpus.py --sets 12")
        print("  .venv/bin/python scripts/build_corpus.py --commit")
        return 1
    if not observations:
        print(f"{len(items)} corpus items but no judgements yet.")
        print("Start the API and annotate:")
        print("  .venv/bin/uvicorn adapi.main:app --reload   ->  /annotate/ui")
        return 1

    measured = _load_features(Path(args.features))
    set_of = {it.item_id: it.set_id for it in items}
    embeddings = _load_embeddings(args, [it.item_id for it in items])
    table = FS.build_table(items, measured, embeddings=embeddings)

    print("=" * 78)
    print("IMAGE-STAGE PERFORMANCE PREDICTOR")
    print("=" * 78)
    if not table.is_real:
        print(
            "\n  !! STAND-IN EMBEDDINGS: the numbers below exercise the pipeline and\n"
            "     are NOT a result. The features carry no visual information, so the\n"
            "     trained model should land at chance. If it does not, something\n"
            "     downstream is leaking.\n"
        )

    print("\n1. DATA AND SPLIT")
    print("-" * 78)
    print(f"  features : {table.summary()}")
    missing = EM.missing_blocks(embeddings)
    if missing:
        print(f"  absent   : {', '.join(missing)} (needs notebooks/colab_embeddings.ipynb)")
    report = split_report(observations, set_of, k=args.folds)
    print("  " + report.summary().replace("\n", "\n  "))
    budget = FS.parameter_budget(report.trainable // args.folds)
    print(
        f"  budget   : ~{report.trainable // args.folds} training comparisons per fold "
        f"supports ~{budget} parameters"
    )

    print("\n2. CEILING")
    print("-" * 78)
    print("  " + ceiling.summary().replace("\n", "\n  "))
    best_ceiling = ceiling.inter_annotator_ceiling
    if np.isnan(best_ceiling):
        best_ceiling = ceiling.test_retest_ceiling
    if np.isnan(best_ceiling):
        print(
            "  No repeats and no multiply-judged pairs yet, so there is no ceiling to\n"
            "  read the accuracies against. Collect more judgements per pair before\n"
            "  quoting any of the numbers below."
        )

    config = PR.TrainConfig(
        hidden=args.hidden,
        max_steps=400 if args.quick else 3000,
        seed=args.seed,
    )
    print(f"\n  fitting {args.folds} folds, {config.describe()}")
    results, full_features = _cross_validate(
        table,
        observations,
        set_of,
        k=args.folds,
        config=config,
        embedding_dim=args.embedding_dim,
        seed=args.seed,
    )

    _print_models(results, best_ceiling)
    _print_kinds(results)

    trained = results.get("pairwise (trained)")
    if trained is not None:
        outcomes = EV.within_set_outcomes(observations, set_of)
        print("\n   deployment metrics (rank the 3 candidates of one job):")
        for name in ("pairwise (trained)", "random"):
            if name in results:
                metrics = EV.set_metrics(results[name].item_scores, outcomes)
                print(f"    {name:<22} {metrics.summary()}")
        print("    chance top-1 with 3 candidates is 0.333")

        if not args.no_ablation:
            _print_ablation(
                table,
                observations,
                set_of,
                trained,
                full_features,
                k=args.folds,
                config=config,
                embedding_dim=args.embedding_dim,
                seed=args.seed,
            )

    print("\n" + "=" * 78)
    if not table.is_real:
        print("Reminder: stand-in embeddings. Not a result.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", default=str(DEFAULT_FEATURES))
    parser.add_argument("--embeddings", default=str(DEFAULT_EMBEDDINGS))
    parser.add_argument(
        "--hash-embeddings",
        action="store_true",
        help="use deterministic stand-in vectors to exercise the path; not a result",
    )
    parser.add_argument("--hash-dim", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=8, help="PCA dims per block")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument(
        "--hidden",
        type=int,
        default=0,
        help="hidden units; 0 is linear. Non-zero exceeds the parameter budget on "
        "purpose, for the row of the table that shows what that costs.",
    )
    parser.add_argument("--kind", default=ItemKind.IMAGE.value, choices=[k.value for k in ItemKind])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="fewer steps, for iterating")
    parser.add_argument("--no-ablation", action="store_true")
    args = parser.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
