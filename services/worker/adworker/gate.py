"""The quality gate — hard pass/fail, no scoring.

This stage answers one question: *is this candidate usable at all?*  It is not a
ranking, and it deliberately shares no code path with the performance predictor.
A candidate whose product is unrecognisable is not "slightly worse", it is
unusable, and expressing that as a low score would let it be ranked first on a bad
day.

Some checks are real today (palette, safe area, exposure, focal clarity — all
numpy, all running now).  Face and product identity compute a real ArcFace or
DINOv2 cosine similarity when an embedding archive is available (see
``_identity_check``), but nothing in this synchronous worker can run either
encoder itself, so on a live job today neither ever is.  NSFW needs the safety
classifier and has no such archive to fall back on.  All three return
``implemented=False`` and pass by default, so nothing claims to have verified an
identity it never looked at — including, for the two that now compute a real
number, the identity it looked at but has no calibrated threshold for yet.

``product_scale`` is a fourth kind and the reason this docstring changed.  It is
real when it can be, and honestly absent when it cannot: it finds the product by
its own colours, so it works for a tan trainer and is undefined for a white one.
Rather than pass silently on the products it cannot see, it reports
``implemented=False`` with the reason, and joins the pending checks in the UI.

It is also the only *ceiling* here.  Every other threshold asks whether there is
enough of something; this one asks whether there is too much, because the failure
it exists for is a product drawn larger than the person holding it — which both
of the first two live jobs returned, and which nothing else in this file could
express.  ``focal_clarity`` in particular cannot: it is a floor on attention
concentration, and across those eleven real frames correct and oversized frames
are interleaved through its whole range, so no cut through it separates them.
"""

from __future__ import annotations

import numpy as np
from adml import features as F
from adml.embeddings import EmbeddingSet
from adproviders import Storage
from adschema import (
    AdJobRequest,
    AssetRef,
    GateCheck,
    GateResult,
    GateVerdict,
    ImageCandidate,
    Tier,
)

# Thresholds calibrated by ``scripts/calibrate_gate.py`` against the mock renderer
# (150 lighting x composition x angle combinations, matched vs deliberately
# mismatched palettes). Measured on-brand ranges are recorded beside each value so
# the week-6 recalibration against real generations has a baseline to compare to.
#
# Each threshold sits below the observed minimum of the legitimate class, not at
# its median: the gate exists to catch genuine failures, and a filter tuned to
# reject a third of usable candidates costs more than it saves.

#: on-brand 0.912-0.972, off-brand 0.000 — classes separate with a 0.912 gap.
THRESH_PALETTE = 0.456
#: on-brand 0.680-0.914 once the sampler stops picking compositions that
#: structurally collide with platform chrome (see ``sampler.compositions_for``).
THRESH_SAFE_AREA = 0.55
#: on-brand 0.578-0.728. Held well below that as a smeared-attention floor,
#: because real generations will spread wider than the mock's synthetic frames.
THRESH_FOCUS = 0.25
#: on-brand luminance 0.531-0.643; these bounds only catch black or blown frames.
MIN_LUMINANCE = 0.06
MAX_LUMINANCE = 0.96
#: on-brand contrast 0.180-0.248; this only catches flat/empty frames.
MIN_CONTRAST = 0.03

#: Ceiling on the share of the frame the product may occupy.  The only threshold
#: here calibrated against **real** generations rather than the mock renderer, and
#: the only one that is a ceiling rather than a floor.
#:
#: It exists because the first two live jobs both returned a product drawn far
#: larger than the person holding it, and nothing in this gate could express that.
#: ``focal_clarity`` cannot: measured across those eleven frames it spans
#: 0.31-0.50 with correct and oversized frames interleaved, so no cut through it
#: separates the classes — see ``adml.features`` on the two other approaches that
#: were tried and failed.
#:
#: Measured with :func:`adml.features.product_area_share` over eleven frames from
#: two live Seedream jobs (an iPhone and a trainer):
#:
#:   visibly correct scale   0.011 - 0.094   (n=5)
#:   unlabelled              0.084 - 0.113   (n=3)
#:   visibly oversized       0.172 - 0.294   (n=3)
#:
#: The classes separate with a 0.059 gap and the threshold sits near the top of
#: it, deliberately: the same reasoning as every floor above, which is that a
#: false reject costs a paid retry on a usable frame while a miss costs a frame
#: that is merely ranked. Eleven frames is a calibration set, not a validation
#: set — this is the first threshold due for revision when the corpus grows.
THRESH_PRODUCT_AREA = 0.16

#: Above this share the reading stops being about the product at all.
#:
#: The measurement finds the product by colour, so it reports an upper bound: it
#: cannot distinguish a product from anything else wearing the product's colours.
#: Past a point that ambiguity dominates, and the honest answer is to decline
#: rather than to reject a frame on a number that is measuring a backdrop.
#:
#: The mock pipeline is where this is unmissable, and the mechanism is a closed
#: loop rather than a bad frame: intake extracts the brand palette *from the
#: product cutout*, and the mock renderer then paints the backdrop, the wardrobe
#: and the product from that same palette. The product's colours genuinely are
#: everywhere, and mock frames measure 0.48-0.57 with a product block covering
#: about 3% of the frame. Real generations do not close that loop — Seedream put
#: the trainer's tan on the trainer and painted the backdrop white.
#:
#: Placed in the gap between the two: real oversized frames reach 0.294, mock
#: frames start at 0.478. It is also a physical claim worth stating plainly — a
#: product covering more than 40% of a 9:16 frame alongside a human model is not
#: a mis-scaled ad, it is a pack shot, and this check is for composites.
UNDISCRIMINATIVE_SHARE = 0.40

#: A check that would need model weights this machine cannot host yet — see
#: ``adml.embeddings`` for why: torch does not belong on the 8 GB laptop this
#: worker runs on, so nothing here ever runs an encoder itself.
_PENDING = {
    "nsfw": "safety classifier — lands with the intake safety gate",
}


def _pending_check(name: str) -> GateCheck:
    return GateCheck(
        name=name,
        value=0.0,
        threshold=0.0,
        passed=True,
        detail=_PENDING[name],
        implemented=False,
    )


#: Which embedding block (as written by ``notebooks/colab_embeddings.ipynb`` and
#: read by ``adml.embeddings.load_directory``) each identity check compares
#: against, and the model that produces it — kept beside ``adml.embeddings.EXPECTED``
#: rather than duplicating those strings from scratch.
_IDENTITY_BLOCKS = {
    "face_identity": ("arcface", "insightface/buffalo_l"),
    "product_identity": ("dinov2", "facebook/dinov2-base"),
}


def _identity_key(asset: AssetRef) -> str:
    """The key an embedding archive would know an asset by.

    Content-addressed, matching the fingerprinting convention already used to
    cache generations (``adworker.pipeline``): the same reference photo used
    across many jobs gets one embedding, not one per job.
    """
    return asset.sha256 or asset.key


def _identity_check(
    name: str,
    candidate_asset: AssetRef,
    reference: AssetRef | None,
    embeddings: dict[str, EmbeddingSet] | None,
) -> GateCheck:
    """Cosine similarity between a candidate frame and its reference.

    This is the mechanism the module docstring used to describe as unbuilt. It is
    no longer unbuilt: given an embedding archive (``notebooks/colab_embeddings.ipynb``
    now writes an ``arcface`` block alongside ``dinov2``, both keyed by
    :func:`_identity_key`), this computes a real cosine similarity with plain numpy.

    It still reports ``implemented=False`` in every case, including the case where
    a real number came out. That is deliberate, not a placeholder left over from
    before: no calibration sweep has measured where same-identity and
    different-identity pairs actually separate for *this* generator — the exact
    measurement ``THRESH_PRODUCT_AREA`` above required eleven real frames before it
    was trusted as a gate. Inventing a threshold from a vendor's generic
    recommendation would be exactly the kind of unmeasured number this project does
    not ship, and would also make ``GateResult.verified_identity`` claim a
    verification this check cannot back. So the value is surfaced — genuinely
    useful for a human deciding where to put that threshold — and the verdict is
    not. Turning this into a real gate is one calibration script and one constant,
    not a rewrite.

    Also honestly absent, same as before: nothing in the live pipeline runs an
    encoder synchronously, because none of them can (see ``adml.embeddings``), so
    ``embeddings`` is ``None`` on every real job today and this still reports
    exactly what it always has. The parameter exists so an offline or future
    asynchronous re-score pass — the only place a Colab round-trip can actually
    happen — has something to call.
    """
    block_name, model_id = _IDENTITY_BLOCKS[name]

    def _absent(detail: str) -> GateCheck:
        return GateCheck(
            name=name, value=0.0, threshold=0.0, passed=True, detail=detail, implemented=False
        )

    if embeddings is None or block_name not in embeddings:
        return _absent(
            f"{block_name} cosine vs the reference — needs {model_id}; run "
            "notebooks/colab_embeddings.ipynb and pass the resulting archive in "
            "to compute it"
        )
    if reference is None:
        return _absent("no reference image was available to compare against")

    block = embeddings[block_name]
    ref_key, cand_key = _identity_key(reference), _identity_key(candidate_asset)
    missing = [k for k in (ref_key, cand_key) if k not in block]
    if missing:
        return _absent(
            f"{block_name} archive is missing {len(missing)} of 2 vectors needed "
            f"(reference {'present' if ref_key not in missing else 'absent'}, "
            f"candidate {'present' if cand_key not in missing else 'absent'})"
        )

    ref_vec, cand_vec = block.vectors[ref_key], block.vectors[cand_key]
    denom = float(np.linalg.norm(ref_vec) * np.linalg.norm(cand_vec))
    cosine = float(np.dot(ref_vec, cand_vec) / denom) if denom > 1e-12 else 0.0
    return GateCheck(
        name=name,
        value=round(cosine, 4),
        threshold=0.0,
        passed=True,
        detail=(
            f"{block_name} cosine similarity to the reference is {cosine:.3f} — "
            "reported for the record, not yet a gate. No calibration sweep has "
            "measured where same-identity and different-identity pairs separate "
            "for this generator, so this cannot pass or fail a candidate yet."
        ),
        implemented=False,
    )


def _product_scale_check(
    rgb: np.ndarray, product: AssetRef | None, storage: Storage, tier: Tier
) -> GateCheck:
    """Is the product drawn at a plausible size, or has it swallowed the frame?

    Reports ``implemented=False`` rather than guessing whenever the measurement is
    undefined — no cutout was kept, or the product has no chromatic signature to
    find it by (a white trainer, a black phone). That is a common case, not an
    edge one, and a check that quietly passed those would claim to have verified
    scale on exactly the products it cannot see.

    **The commonest undefined case is no cutout at all.** ``product_reference``
    falls back to the original upload when rembg produced nothing usable, and the
    dominant colours of a product *photograph* are its backdrop's as much as the
    product's — measured, such a signature matched 59% of a frame whose product
    covered 3% of it. :func:`~adml.features.product_colour_signature` refuses a
    fully opaque image for that reason, so the fallback arrives here as "could not
    measure" rather than as a spurious rejection.

    **Synthetic frames are excluded outright, and the reason is a closed loop.**
    Intake extracts the brand palette *from the product cutout*, and the mock
    renderer paints its backdrop, wardrobe and product from that same palette — so
    the product's colours genuinely cover the frame, and the reading is meaningless
    across its whole range rather than merely high. Which end of the range a given
    seed lands on is arbitrary, so leaving this to ``UNDISCRIMINATIVE_SHARE`` made
    the dev pipeline reject candidates by seed. Real generators do not close that
    loop: Seedream put the trainer's tan on the trainer and painted the backdrop
    white.

    The cost is real and belongs here rather than in a footnote — this check is
    exercised only against paid generations, so its calibration cannot be
    regression-tested by the default dev path.
    """
    if tier is Tier.MOCK:
        return GateCheck(
            name="product_scale",
            value=0.0,
            threshold=0.0,
            passed=True,
            detail=(
                "the mock renderer paints the whole frame from a palette extracted "
                "from the product, so the product's colours locate nothing"
            ),
            implemented=False,
        )

    unavailable = "no product reference was available to measure against"
    signature = None
    if product is not None:
        try:
            signature = F.product_colour_signature(F.load_rgba(storage.get_bytes(product.key)))
        except (KeyError, OSError, ValueError) as exc:
            signature = None
            unavailable = f"the product reference could not be read ({exc})"
        else:
            if signature is None:
                unavailable = (
                    "the product has no chromatic signature to locate it by — a white, "
                    "black or grey product cannot be told from a studio backdrop by "
                    "colour, and this check has no detector yet"
                )

    if signature is None:
        return GateCheck(
            name="product_scale",
            value=0.0,
            threshold=0.0,
            passed=True,
            detail=unavailable,
            implemented=False,
        )

    share = F.product_area_share(rgb, signature)
    if share > UNDISCRIMINATIVE_SHARE:
        return GateCheck(
            name="product_scale",
            value=round(share, 4),
            threshold=THRESH_PRODUCT_AREA,
            passed=True,
            higher_is_better=False,
            detail=(
                f"the product's colours cover {share:.1%} of the frame, which is too "
                "much to be the product — they are matching the backdrop or the "
                "wardrobe, so scale could not be measured here"
            ),
            implemented=False,
        )

    return GateCheck(
        name="product_scale",
        value=round(share, 4),
        threshold=THRESH_PRODUCT_AREA,
        passed=share <= THRESH_PRODUCT_AREA,
        higher_is_better=False,
        detail=(
            f"the product's colours cover {share:.1%} of the frame; above "
            f"{THRESH_PRODUCT_AREA:.0%} it is drawn larger than the scene supports. "
            "An upper bound — anything else wearing the product's colours counts too."
        ),
    )


def evaluate_image(
    candidate: ImageCandidate,
    request: AdJobRequest,
    storage: Storage,
    attempt: int = 1,
    product: AssetRef | None = None,
    palette_reliable: bool = True,
    human_reference: AssetRef | None = None,
    embeddings: dict[str, EmbeddingSet] | None = None,
) -> GateResult:
    """Run every hard filter against a generated frame.

    ``product`` is the reference the generator was given — the rembg cutout when
    intake kept one. Optional because the scale check degrades to "could not be
    measured" without it rather than failing the job. ``human_reference`` is its
    counterpart for the face-identity check. ``embeddings`` is the archive
    :func:`_identity_check` reads from; ``None`` on every live job today, because
    nothing in this synchronous worker can run an encoder — see that function's
    docstring for what it does once one exists.

    ``palette_reliable`` is False when intake could not cut the product out and read
    the palette off the whole photograph instead. The measurement still runs and is
    still reported; it just stops being grounds for rejection. This is not leniency.
    A palette taken from a product shot on a pale studio sweep comes back ~79% off-
    white, and the gate then demands that every advertisement be 79% off-white —
    which no brief in this system would ever ask a generator to produce. The first
    live job to hit it rejected all five candidates, retried all five, and refused
    to spend on video, for $0.20 and no usable output. The UI already called that
    palette "a materially weaker constraint"; this makes the gate agree with it.
    """
    rgb = F.load_image(storage.get_bytes(candidate.asset.key))
    sal = F.saliency_map(rgb)
    safe = request.platform.safe_area

    checks: list[GateCheck] = []

    palette_score, mean_de = F.palette_adherence(rgb, request.theme.palette)
    checks.append(
        GateCheck(
            name="palette_adherence",
            value=round(palette_score, 4),
            threshold=THRESH_PALETTE,
            passed=palette_score >= THRESH_PALETTE,
            advisory=not palette_reliable,
            detail=(
                f"weighted mean ΔE76 {mean_de:.1f} against the brand palette"
                + (
                    ""
                    if palette_reliable
                    else " — advisory only: the palette was read off the whole product "
                    "photograph, so it describes that photo's backdrop rather than the brand"
                )
            ),
        )
    )

    safe_share = F.region_saliency_share(sal, safe.top, safe.bottom)
    checks.append(
        GateCheck(
            name="safe_area",
            value=round(safe_share, 4),
            threshold=THRESH_SAFE_AREA,
            passed=safe_share >= THRESH_SAFE_AREA,
            detail=(
                f"{safe_share:.0%} of attention sits clear of the top {safe.top:.0%} / "
                f"bottom {safe.bottom:.0%} platform chrome"
            ),
        )
    )

    focus = F.focal_concentration(sal)
    checks.append(
        GateCheck(
            name="focal_clarity",
            value=round(focus, 4),
            threshold=THRESH_FOCUS,
            passed=focus >= THRESH_FOCUS,
            detail="attention is concentrated on a clear subject rather than smeared",
        )
    )

    checks.append(_product_scale_check(rgb, product, storage, candidate.tier))

    lum = F.mean_luminance(rgb)
    checks.append(
        GateCheck(
            name="exposure",
            value=round(lum, 4),
            threshold=MIN_LUMINANCE,
            passed=MIN_LUMINANCE <= lum <= MAX_LUMINANCE,
            detail=f"mean luminance {lum:.2f}; rejects fully black or blown-out frames",
        )
    )

    contrast = F.rms_contrast(rgb)
    checks.append(
        GateCheck(
            name="contrast",
            value=round(contrast, 4),
            threshold=MIN_CONTRAST,
            passed=contrast >= MIN_CONTRAST,
            detail=f"RMS contrast {contrast:.3f}; rejects flat or empty frames",
        )
    )

    checks.append(_identity_check("face_identity", candidate.asset, human_reference, embeddings))
    checks.append(_identity_check("product_identity", candidate.asset, product, embeddings))
    checks.extend(_pending_check(name) for name in _PENDING)

    # Blocking failures, not every failure: an advisory check reports but cannot veto.
    failed = [c for c in checks if not c.passed and not c.advisory]
    if not failed:
        verdict = GateVerdict.PASS
    elif attempt == 1:
        # One retry with a stricter prompt, then give up. The retry allowance is
        # enforced by the cost governor, not here — this only expresses intent.
        verdict = GateVerdict.RETRY
    else:
        verdict = GateVerdict.REJECT

    return GateResult(verdict=verdict, checks=checks, attempt=attempt)


def stricter_prompt(original: str, result: GateResult) -> str:
    """Amend a prompt in response to the specific checks that failed.

    Retrying with the identical prompt is a waste of a paid generation, so the
    retry says something new about what went wrong.
    """
    failed = {c.name for c in result.failures}
    additions = []
    if "palette_adherence" in failed:
        additions.append(
            "Adhere strictly to the specified colour palette — the backdrop and "
            "wardrobe must come from those colours."
        )
    if "safe_area" in failed:
        additions.append(
            "Move the product and the model's face toward the vertical centre of the "
            "frame, well clear of the top and bottom edges."
        )
    if "product_scale" in failed:
        additions.append(
            "Draw the product at its true physical size next to the model — it must "
            "read as an object a person could pick up, never enlarged to fill the "
            "frame. Show the model's hand or body near it so the scale is legible."
        )
    if "focal_clarity" in failed:
        additions.append(
            "Make the product the single unmistakable focal point; simplify the "
            "background and remove competing detail."
        )
    if "exposure" in failed or "contrast" in failed:
        additions.append(
            "Use balanced commercial exposure with clear tonal separation between "
            "the subject and the background."
        )
    if not additions:
        return original
    return (
        original
        + "\n\nCORRECTIONS (previous attempt failed quality control):\n"
        + "\n".join(f"- {a}" for a in additions)
    )
