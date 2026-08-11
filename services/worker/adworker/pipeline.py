"""The eight-stage pipeline.

Ordering here is not arbitrary — it is the cost-aware cascade the whole project
is built around:

    intake → brief → images → gate → image rank → videos → video rank → delivery

The gate and the image-stage ranking both sit *before* video generation, which is
where essentially all the money is.  A candidate that fails quality control never
gets animated, and the image-stage ranking is recorded before any video exists so
that its agreement with the final ranking can be measured honestly rather than
reconstructed after the fact.

Runs inline on asyncio today.  Nothing here assumes an event loop it owns, so
moving it behind ARQ later is a matter of calling ``run_job`` from a task.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

from adproviders import (
    BudgetExceeded,
    CostGovernor,
    ImageGenRequest,
    ImageProvider,
    LLMProvider,
    RetryBudgetExceeded,
    Storage,
    VideoGenRequest,
    VideoProvider,
    estimate_job_cost,
)
from adschema import (
    AdJobRequest,
    GateVerdict,
    ImageCandidate,
    JobRecord,
    JobResult,
    JobState,
    Stage,
    StageEvent,
    VideoCandidate,
)

from .briefs import compile_briefs
from .gate import evaluate_image, stricter_prompt
from .intake import preprocess
from .scoring import rank_videos, score_image, score_video

#: Called with each progress event.  The API turns this into an SSE stream.
ProgressHook = Callable[[StageEvent], Awaitable[None]] | None


class ConsentRefused(RuntimeError):
    """Intake refused the job because the rights attestation was incomplete."""


class UnusableReference(RuntimeError):
    """An upload cannot produce a usable ad, so the job is refused before any spend."""


class NoViableCandidates(RuntimeError):
    """Every candidate failed the quality gate, so there is nothing to animate."""


class Pipeline:
    def __init__(
        self,
        *,
        storage: Storage,
        governor: CostGovernor,
        image_provider: ImageProvider,
        video_provider: VideoProvider,
        llm_provider: LLMProvider,
        on_progress: ProgressHook = None,
    ):
        self.storage = storage
        self.governor = governor
        self.images = image_provider
        self.videos = video_provider
        self.llm = llm_provider
        self.on_progress = on_progress

    # --- Progress -----------------------------------------------------------

    async def _emit(
        self,
        record: JobRecord,
        stage: Stage,
        state: str,
        message: str = "",
        progress: float = 0.0,
    ) -> None:
        event = StageEvent(
            job_id=record.job_id, stage=stage, state=state, message=message, progress=progress
        )
        record.events.append(event)
        if self.on_progress is not None:
            await self.on_progress(event)

    # --- Stage 1: intake ----------------------------------------------------

    async def _intake(self, record: JobRecord, result: JobResult) -> None:
        request = record.request
        await self._emit(record, Stage.INTAKE, "started", "validating inputs and rights")

        # Rights first. Everything after this reads the uploads' pixels, and there
        # is no reason to process a likeness the user has not attested to.
        if not request.consent.is_valid:
            raise ConsentRefused(
                "the human-model image needs an affirmative rights attestation "
                "(model release held, and not a public figure) before generation can start"
            )

        report = preprocess(request, self.storage)
        result.intake = report

        blocking = report.blocking_reason
        if blocking is not None:
            raise UnusableReference(
                f"refusing before any spend — {blocking}. "
                "No generator can recover detail an upload does not contain."
            )

        await self._emit(
            record,
            Stage.INTAKE,
            "progress",
            f"product cutout: {report.cutout.detail}",
            0.4,
        )
        await self._emit(record, Stage.INTAKE, "progress", f"face: {report.face.detail}", 0.6)

        # The extracted palette is written back into the request rather than
        # threaded separately, because the brief compiler, the gate and the scorer
        # all read `request.theme.palette`. `palette_auto_extracted` keeps the
        # record honest about the fact that the user did not supply it, and
        # `IntakeReport.palette_source` records where it came from.
        if report.palette and not request.theme.palette:
            request.theme.palette = report.palette
            request.theme.palette_auto_extracted = True
            await self._emit(
                record,
                Stage.INTAKE,
                "progress",
                f"brand palette derived from the {report.palette_source.replace('-', ' ')}: "
                + " ".join(report.palette),
                0.8,
            )
        elif not request.theme.palette:
            await self._emit(
                record,
                Stage.INTAKE,
                "progress",
                "no brand palette supplied or extractable; palette adherence will not "
                "constrain this job",
                0.8,
            )

        for note in report.all_advisories:
            await self._emit(record, Stage.INTAKE, "progress", f"note — {note}", 0.9)

        await self._emit(
            record,
            Stage.INTAKE,
            "completed",
            f"{len(report.references)} reference(s) accepted",
            1.0,
        )

    # --- Stage 2: briefs ----------------------------------------------------

    async def _briefs(self, record: JobRecord, result: JobResult) -> None:
        await self._emit(record, Stage.BRIEF, "started", "sampling the design space")
        brief_set = await compile_briefs(record.request, self.llm)
        result.briefs = brief_set
        await self._emit(
            record,
            Stage.BRIEF,
            "completed",
            f"{len(brief_set)} briefs, min pairwise distance {brief_set.min_pairwise_distance}/4",
            1.0,
        )

    # --- Stage 3 + 4: images with the gate in the retry loop ----------------

    async def _generate_one_image(
        self,
        record: JobRecord,
        result: JobResult,
        brief,
        attempt: int,
        prompt_override: str | None,
    ) -> ImageCandidate:
        request = record.request
        seed = request.candidate_seed(brief.index)
        suffix = "" if attempt == 1 else f"_r{attempt}"
        key = f"generations/{request.job_id}/img_{brief.index}{suffix}.png"

        effective = (
            brief
            if prompt_override is None
            else brief.model_copy(update={"image_prompt": prompt_override})
        )

        # The cutout when intake produced a trustworthy one, the original otherwise.
        # A product on transparency holds its identity through multi-reference
        # composition markedly better than one still attached to its backdrop.
        product = (
            result.intake.product_reference(request.product_image)
            if result.intake is not None
            else request.product_image
        )
        references = [request.human_model_image, product]
        if request.logo_image is not None:
            references.append(request.logo_image)

        gen_request = ImageGenRequest(
            brief=effective,
            references=references,
            aspect_ratio=request.aspect_ratio,
            seed=seed,
            output_key=key,
            reference_roles=["human model", "product", "logo"][: len(references)],
            palette=request.theme.palette,
        )

        fingerprint = self.governor.fingerprint(
            "image",
            self.images.model,
            {
                "prompt": effective.image_prompt,
                "negative": effective.negative_prompt,
                "aspect": request.aspect_ratio.value,
                "seed": seed,
                "refs": [r.sha256 or r.key for r in references],
            },
        )
        cached = await self.governor.cached_asset(fingerprint)
        if cached is not None:
            return ImageCandidate(
                index=brief.index,
                brief=effective,
                asset=cached,
                tier=self.images.tier,
                provider=self.images.name,
                seed=seed,
                cost_usd=0.0,
            )

        result = await self.governor.guarded_call(
            job_id=request.job_id,
            provider=self.images.name,
            model=self.images.model,
            operation="image",
            estimated_usd=self.images.estimate_cost(1),
            quantity=1.0,
            call=lambda: self.images.generate(gen_request),
            actual_cost_of=lambda r: r.cost_usd,
            note=f"candidate {brief.index} attempt {attempt}",
        )
        await self.governor.remember_asset(fingerprint, "image", result.asset)

        return ImageCandidate(
            index=brief.index,
            brief=effective,
            asset=result.asset,
            tier=result.tier,
            provider=result.model,
            seed=result.seed,
            cost_usd=result.cost_usd,
            latency_ms=result.latency_ms,
        )

    async def _images_and_gate(self, record: JobRecord, result: JobResult) -> None:
        request = record.request
        briefs = result.briefs.briefs if result.briefs else []
        total = len(briefs)

        await self._emit(record, Stage.IMAGE_GEN, "started", f"generating {total} candidates")

        for brief in briefs:
            slot_key = f"{request.job_id}:slot{brief.index}"
            prompt_override: str | None = None
            candidate: ImageCandidate | None = None

            while True:
                try:
                    attempt = self.governor.register_attempt(slot_key)
                except RetryBudgetExceeded:
                    # Allowance spent: keep the last attempt as a rejected candidate
                    # so the UI can show what happened rather than silently dropping it.
                    break

                candidate = await self._generate_one_image(
                    record, result, brief, attempt, prompt_override
                )
                gate = evaluate_image(candidate, request, self.storage, attempt=attempt)
                candidate.gate = gate

                if gate.verdict is GateVerdict.PASS:
                    break
                if gate.verdict is GateVerdict.REJECT:
                    break

                # RETRY: say something new about what failed, or stop.
                prompt_override = stricter_prompt(brief.image_prompt, gate)
                await self._emit(
                    record,
                    Stage.QUALITY_GATE,
                    "progress",
                    f"candidate {brief.index} failed {gate.reason}; retrying once",
                    (brief.index + 0.5) / max(1, total),
                )

            if candidate is not None:
                result.images.append(candidate)
                result.total_cost_usd += candidate.cost_usd

            await self._emit(
                record,
                Stage.IMAGE_GEN,
                "progress",
                f"candidate {brief.index + 1}/{total} done",
                (brief.index + 1) / max(1, total),
            )

        generated = len(result.images)
        await self._emit(
            record,
            Stage.IMAGE_GEN,
            "completed",
            f"{generated} candidate frame(s) generated",
            1.0,
        )

        passed = [c for c in result.images if c.passed_gate]
        await self._emit(
            record,
            Stage.QUALITY_GATE,
            "completed",
            f"{len(passed)}/{len(result.images)} candidates passed quality control",
            1.0,
        )
        if not passed:
            raise NoViableCandidates(
                "every candidate failed the quality gate; refusing to spend on video "
                f"({'; '.join(c.gate.reason for c in result.images if c.gate)})"
            )

    # --- Stage 5: image-stage ranking (before any video spend) --------------

    async def _rank_images(self, record: JobRecord, result: JobResult) -> None:
        await self._emit(
            record, Stage.IMAGE_RANK, "started", "predicting performance from the still frames"
        )
        for candidate in result.images:
            if candidate.passed_gate:
                candidate.score = score_image(candidate, record.request, self.storage)

        order = result.image_stage_order
        await self._emit(
            record,
            Stage.IMAGE_RANK,
            "completed",
            "predicted order before animation: " + " > ".join(chr(ord("A") + i) for i in order),
            1.0,
        )

    # --- Stage 6: video ----------------------------------------------------

    async def _videos(self, record: JobRecord, result: JobResult) -> None:
        request = record.request
        promote = [c for c in result.images if c.passed_gate]
        total = len(promote)

        native = self.videos.supports_duration(request.duration_seconds)
        delivered = self.videos.deliverable_duration(request.duration_seconds)
        if not native:
            note = "chained (provider caps below the requested duration)"
        elif delivered != request.duration_seconds:
            # Kling offers {5, 10} and nothing between, so a 9 s ask becomes 10 s.
            note = (
                f"native, delivered at {delivered:.0f}s — "
                f"{self.videos.model} offers fixed durations only"
            )
        else:
            note = "native"
        await self._emit(
            record,
            Stage.VIDEO_GEN,
            "started",
            f"animating {total} candidates at {request.duration_seconds:.0f}s — {note}",
        )

        for position, source in enumerate(promote):
            seed = request.candidate_seed(source.index)
            key = f"generations/{request.job_id}/vid_{source.index}.mp4"
            gen_request = VideoGenRequest(
                brief=source.brief,
                start_image=source.asset,
                duration_seconds=request.duration_seconds,
                aspect_ratio=request.aspect_ratio,
                seed=seed,
                output_key=key,
            )

            fingerprint = self.governor.fingerprint(
                "video",
                self.videos.model,
                {
                    "motion": source.brief.motion_prompt,
                    "start": source.asset.sha256 or source.asset.key,
                    # The delivered duration, not the requested one: on a provider
                    # with a {5, 10} enum an 8 s and a 9 s request produce the same
                    # 10 s clip, so they should share a cache entry rather than
                    # paying twice for identical output.
                    "duration": delivered,
                    "aspect": request.aspect_ratio.value,
                    "seed": seed,
                },
            )
            cached = await self.governor.cached_asset(fingerprint)
            if cached is not None:
                video = VideoCandidate(
                    index=position,
                    source_image_index=source.index,
                    brief=source.brief,
                    asset=cached,
                    duration_seconds=delivered,
                    requested_duration_seconds=request.duration_seconds,
                    tier=self.videos.tier,
                    provider=self.videos.name,
                    seed=seed,
                    seed_honoured=self.videos.honours_seed,
                    cost_usd=0.0,
                )
            else:
                gen = await self.governor.guarded_call(
                    job_id=request.job_id,
                    provider=self.videos.name,
                    model=self.videos.model,
                    operation="video",
                    estimated_usd=self.videos.estimate_cost(request.duration_seconds, 1),
                    quantity=request.duration_seconds,
                    call=lambda r=gen_request: self.videos.generate(r),
                    actual_cost_of=lambda r: r.cost_usd,
                    note=f"candidate {source.index}",
                )
                await self.governor.remember_asset(fingerprint, "video", gen.asset)
                video = VideoCandidate(
                    index=position,
                    source_image_index=source.index,
                    brief=source.brief,
                    asset=gen.asset,
                    duration_seconds=gen.duration_seconds,
                    requested_duration_seconds=request.duration_seconds,
                    fps=gen.fps,
                    tier=gen.tier,
                    provider=gen.model,
                    seed=gen.seed,
                    seed_honoured=gen.seed_honoured,
                    cost_usd=gen.cost_usd,
                    latency_ms=gen.latency_ms,
                    was_chained=gen.was_chained,
                    seam_consistency=gen.seam_consistency,
                )

            video.thumbnail = source.asset
            result.videos.append(video)
            result.total_cost_usd += video.cost_usd

            await self._emit(
                record,
                Stage.VIDEO_GEN,
                "progress",
                f"video {position + 1}/{total} done",
                (position + 1) / max(1, total),
            )

        await self._emit(record, Stage.VIDEO_GEN, "completed", f"{total} videos generated", 1.0)

    # --- Stage 7: final ranking --------------------------------------------

    async def _rank_videos(self, record: JobRecord, result: JobResult) -> None:
        await self._emit(record, Stage.VIDEO_RANK, "started", "scoring the animated candidates")
        for video in result.videos:
            video.score = score_video(video, record.request, self.storage)

        result.ranking = rank_videos(result.videos, result.image_stage_order)
        winner = result.winner
        await self._emit(
            record,
            Stage.VIDEO_RANK,
            "completed",
            f"winner: candidate {chr(ord('A') + winner.video.source_image_index)}"
            if winner
            else "no ranking produced",
            1.0,
        )

    # --- Stage 8: delivery -------------------------------------------------

    async def _delivery(self, record: JobRecord, result: JobResult) -> None:
        await self._emit(record, Stage.DELIVERY, "started", "preparing platform renders")
        # Smart crop, caption burn-in and the ffmpeg audio mix land in week 13.
        # The native aspect ratio is registered now so the UI has something to show.
        ratio = record.request.aspect_ratio.value
        for video in result.videos:
            video.platform_renders[ratio] = video.asset
        await self._emit(
            record,
            Stage.DELIVERY,
            "completed",
            f"{len(result.videos)} videos ready at {ratio}",
            1.0,
        )

    # --- Entry point -------------------------------------------------------

    async def run(self, request: AdJobRequest) -> JobRecord:
        record = JobRecord(request=request, state=JobState.RUNNING)
        record.started_at = datetime.now(UTC)
        result = JobResult(job_id=request.job_id)
        record.result = result

        estimate = estimate_job_cost(
            self.images.model,
            self.videos.model,
            self.llm.model,
            request.candidate_count,
            request.duration_seconds,
        )

        try:
            # Refuse up front rather than halfway through: a job that dies at the
            # video stage has already paid for images that buy nothing.
            await self.governor.preflight(request.job_id, estimate)

            await self._intake(record, result)
            await self._briefs(record, result)
            await self._images_and_gate(record, result)
            await self._rank_images(record, result)
            await self._videos(record, result)
            await self._rank_videos(record, result)
            await self._delivery(record, result)

            record.state = JobState.COMPLETED

        except BudgetExceeded as exc:
            record.state = JobState.REFUSED_OVER_BUDGET
            record.refusal_reason = str(exc)
            await self._emit(record, Stage.INTAKE, "failed", str(exc))
        except (ConsentRefused, UnusableReference, NoViableCandidates) as exc:
            record.state = JobState.FAILED
            record.error = str(exc)
            await self._emit(record, record.current_stage or Stage.INTAKE, "failed", str(exc))
        except Exception as exc:  # pragma: no cover - surfaced to the UI verbatim
            record.state = JobState.FAILED
            record.error = f"{type(exc).__name__}: {exc}"
            await self._emit(record, record.current_stage or Stage.INTAKE, "failed", record.error)
        finally:
            record.finished_at = datetime.now(UTC)
            self.governor.reset_attempts(f"{request.job_id}:")

        return record
