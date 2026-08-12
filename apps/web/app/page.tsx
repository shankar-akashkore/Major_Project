"use client";

/**
 * The wizard.
 *
 * Three decisions here come from the schema rather than from taste:
 *
 * **Camera angle is not asked for.** It is the axis the design-space sampler varies
 * across the three candidates, which is what makes them genuinely different rather
 * than three attempts at one idea. Offering it as a control would quietly turn the
 * diversity guarantee off, so it appears only as an explicit "lock all three",
 * under Advanced, where choosing it is a decision rather than a default.
 *
 * **Aspect ratio is not asked for either.** It follows from the platform, and the
 * resolved value is shown so the derivation is visible rather than hidden.
 *
 * **Consent is a gate, not a checkbox.** `ConsentAttestation.is_valid` requires
 * both attestations and the API returns 422 without them. The submit button stays
 * disabled and says which one is missing, because a form that lets you submit and
 * then explains the refusal has taught you nothing about why it exists.
 */

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { useConfig } from "@/components/Shell.tsx";
import {
  Badge,
  Button,
  Card,
  Caveat,
  Checkbox,
  ErrorNote,
  Field,
  SectionTitle,
  Select,
  TextArea,
  TextInput,
} from "@/components/ui.tsx";
import { createDemoJob, createJob } from "@/lib/api.ts";
import {
  BACKGROUND_TREATMENT_VALUES,
  CAMERA_ANGLE_VALUES,
  DEFAULT_CANDIDATE_COUNT,
  MAX_DURATION_S,
  MIN_DURATION_S,
  MOOD_VALUES,
  PLATFORM_VALUES,
  VERTICAL_VALUES,
  type BackgroundTreatment,
  type CameraAngle,
  type Mood,
  type Platform,
  type Vertical,
} from "@/lib/contract.ts";

export default function NewJobPage() {
  const router = useRouter();
  const config = useConfig();

  const [human, setHuman] = useState<File | null>(null);
  const [product, setProduct] = useState<File | null>(null);
  const [logo, setLogo] = useState<File | null>(null);

  const [productName, setProductName] = useState("");
  const [caption, setCaption] = useState("");
  const [cta, setCta] = useState("Shop now");
  const [additional, setAdditional] = useState("");
  const [negative, setNegative] = useState("");

  const [vertical, setVertical] = useState<Vertical>("other");
  const [platform, setPlatform] = useState<Platform>("instagram_reels");
  const [mood, setMood] = useState<Mood>("warm_lifestyle");
  const [background, setBackground] = useState<BackgroundTreatment>("soft_gradient");
  const [palette, setPalette] = useState("");
  const [duration, setDuration] = useState(9);
  const [seed, setSeed] = useState(0);
  const [lockAngle, setLockAngle] = useState(false);
  const [angle, setAngle] = useState<CameraAngle>("eye_level");
  const [advanced, setAdvanced] = useState(false);

  const [hasRelease, setHasRelease] = useState(false);
  const [notPublicFigure, setNotPublicFigure] = useState(false);

  const [busy, setBusy] = useState<null | "job" | "demo">(null);
  const [error, setError] = useState<string | null>(null);

  const geometry = config?.platforms?.[platform];

  /** Everything standing between this form and a job, most fixable first. */
  const blockers = useMemo(() => {
    const out: string[] = [];
    if (!human) out.push("a human-model image");
    if (!product) out.push("a product image");
    if (!productName.trim()) out.push("the product name");
    if (!hasRelease) out.push("the model-release attestation");
    if (!notPublicFigure) out.push("the public-figure attestation");
    return out;
  }, [human, product, productName, hasRelease, notPublicFigure]);

  async function submit() {
    if (blockers.length > 0) return;
    setBusy("job");
    setError(null);
    const form = new FormData();
    form.set("human_model_image", human!);
    form.set("product_image", product!);
    if (logo) form.set("logo_image", logo);
    form.set("product_name", productName);
    form.set("caption", caption);
    form.set("cta_text", cta);
    form.set("additional_prompt", additional);
    form.set("negative_constraints", negative);
    form.set("vertical", vertical);
    form.set("platform", platform);
    form.set("mood", mood);
    form.set("palette", palette);
    form.set("background", background);
    form.set("duration_seconds", String(duration));
    form.set("candidate_count", String(DEFAULT_CANDIDATE_COUNT));
    form.set("seed", String(seed));
    if (lockAngle) form.set("locked_angle", angle);
    form.set("has_model_release", String(hasRelease));
    form.set("not_a_public_figure", String(notPublicFigure));
    try {
      const launched = await createJob(form);
      router.push(`/jobs/${launched.job_id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setBusy(null);
    }
  }

  async function runDemo() {
    setBusy("demo");
    setError(null);
    try {
      const launched = await createDemoJob({ platform, duration_seconds: duration, seed: seed || 7 });
      router.push(`/jobs/${launched.job_id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setBusy(null);
    }
  }

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-zinc-100">New job</h1>
          <p className="mt-1 max-w-2xl text-sm text-zinc-400">
            Three image candidates from three well-separated design points, ranked as images, then
            each animated into an {MIN_DURATION_S}–{MAX_DURATION_S} second video and ranked again.
          </p>
        </div>
        <Button variant="ghost" onClick={runDemo} disabled={busy !== null}>
          {busy === "demo" ? "Starting…" : "Run with synthetic references"}
        </Button>
      </div>

      {error ? <ErrorNote>{error}</ErrorNote> : null}

      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <div className="space-y-6">
          <Card className="p-4">
            <SectionTitle hint="PNG or JPEG">References</SectionTitle>
            <div className="grid gap-4 sm:grid-cols-3">
              <FilePicker label="Human model" required file={human} onPick={setHuman} />
              <FilePicker
                label="Product"
                required
                file={product}
                onPick={setProduct}
                hint="Cut out at intake; a clean background helps."
              />
              <FilePicker label="Logo" file={logo} onPick={setLogo} hint="Optional." />
            </div>
          </Card>

          <Card className="p-4">
            <SectionTitle>Copy</SectionTitle>
            <div className="space-y-4">
              <Field label="Product name" required>
                <TextInput
                  value={productName}
                  maxLength={120}
                  placeholder="Aurora Serum"
                  onChange={(event) => setProductName(event.target.value)}
                />
              </Field>
              <div className="grid gap-4 sm:grid-cols-2">
                <Field label="Caption">
                  <TextInput
                    value={caption}
                    maxLength={300}
                    placeholder="Glow that lasts"
                    onChange={(event) => setCaption(event.target.value)}
                  />
                </Field>
                <Field label="Call to action">
                  <TextInput
                    value={cta}
                    maxLength={40}
                    onChange={(event) => setCta(event.target.value)}
                  />
                </Field>
              </div>
              <Field label="Additional direction" hint="Folded into each candidate's shot brief.">
                <TextArea
                  value={additional}
                  maxLength={1000}
                  onChange={(event) => setAdditional(event.target.value)}
                />
              </Field>
              <Field
                label="Negative constraints"
                hint="e.g. no hands covering the label, no visible text on the product."
              >
                <TextArea
                  value={negative}
                  maxLength={500}
                  onChange={(event) => setNegative(event.target.value)}
                />
              </Field>
            </div>
          </Card>

          <Card className="p-4">
            <SectionTitle>Targeting and look</SectionTitle>
            <div className="grid gap-4 sm:grid-cols-2">
              <Field label="Product vertical" hint="A predictor feature, not a label.">
                <Select value={vertical} onChange={setVertical} options={VERTICAL_VALUES} />
              </Field>
              <Field
                label="Platform"
                hint={
                  geometry
                    ? `${geometry.aspect_ratio} · safe area ${Math.round(
                        geometry.safe_area.top * 100,
                      )}% top, ${Math.round(geometry.safe_area.bottom * 100)}% bottom`
                    : "Aspect ratio and safe areas follow from this."
                }
              >
                <Select value={platform} onChange={setPlatform} options={PLATFORM_VALUES} />
              </Field>
              <Field label="Mood" hint="Drives lighting and motion energy.">
                <Select value={mood} onChange={setMood} options={MOOD_VALUES} />
              </Field>
              <Field label="Background">
                <Select
                  value={background}
                  onChange={setBackground}
                  options={BACKGROUND_TREATMENT_VALUES}
                />
              </Field>
              <Field
                label="Brand palette"
                hint="Comma-separated hex. Left empty, intake extracts it from the product cutout."
              >
                <TextInput
                  value={palette}
                  placeholder="#2b3a55, #ce7777"
                  onChange={(event) => setPalette(event.target.value)}
                />
              </Field>
              <Field label={`Duration (${MIN_DURATION_S}–${MAX_DURATION_S}s)`}>
                <TextInput
                  type="number"
                  min={MIN_DURATION_S}
                  max={MAX_DURATION_S}
                  step={0.5}
                  value={duration}
                  onChange={(event) => setDuration(Number(event.target.value))}
                />
              </Field>
            </div>

            <button
              type="button"
              onClick={() => setAdvanced((open) => !open)}
              className="mt-4 text-xs text-zinc-400 underline decoration-dotted hover:text-zinc-200"
            >
              {advanced ? "Hide" : "Show"} advanced
            </button>

            {advanced ? (
              <div className="mt-4 space-y-4 border-t border-zinc-800 pt-4">
                <Field
                  label="Seed"
                  hint={`Candidate i uses seed + i, so runs are reproducible. ${DEFAULT_CANDIDATE_COUNT} candidates.`}
                >
                  <TextInput
                    type="number"
                    min={0}
                    value={seed}
                    onChange={(event) => setSeed(Number(event.target.value))}
                  />
                </Field>
                <div className="space-y-2">
                  <Checkbox checked={lockAngle} onChange={setLockAngle}>
                    Lock the camera angle across all three candidates
                  </Checkbox>
                  {lockAngle ? (
                    <>
                      <Select value={angle} onChange={setAngle} options={CAMERA_ANGLE_VALUES} />
                      <Caveat>
                        Angle is the sampler&apos;s main diversity axis. Locking it makes the three
                        candidates variations on one shot, which is a weaker comparison.
                      </Caveat>
                    </>
                  ) : (
                    <p className="text-[11px] text-zinc-500">
                      Left unlocked, the sampler varies angle across the three candidates and
                      reports the diversity it achieved.
                    </p>
                  )}
                </div>
              </div>
            ) : null}
          </Card>
        </div>

        <div className="space-y-6">
          <Card className="border-warn-500/30 p-4">
            <SectionTitle>Rights</SectionTitle>
            <p className="mb-3 text-xs leading-relaxed text-zinc-400">
              This uploads a person&apos;s likeness to a generative model. Both attestations are
              required and the job is refused without them.
            </p>
            <div className="space-y-3">
              <Checkbox checked={hasRelease} onChange={setHasRelease}>
                I hold the rights to use this person&apos;s likeness in an advertisement.
              </Checkbox>
              <Checkbox checked={notPublicFigure} onChange={setNotPublicFigure}>
                This image is not of a public figure or celebrity.
              </Checkbox>
            </div>
          </Card>

          <Card className="p-4">
            <SectionTitle>Submit</SectionTitle>
            {blockers.length > 0 ? (
              <p className="mb-3 text-xs leading-relaxed text-zinc-400">
                Still needed: {blockers.join(", ")}.
              </p>
            ) : (
              <p className="mb-3 text-xs text-zinc-400">
                {DEFAULT_CANDIDATE_COUNT} images → {DEFAULT_CANDIDATE_COUNT} videos at {duration}s,{" "}
                {geometry?.aspect_ratio ?? "platform ratio"}.
              </p>
            )}
            <Button type="button" onClick={submit} disabled={blockers.length > 0 || busy !== null}>
              {busy === "job" ? "Starting…" : "Generate candidates"}
            </Button>
            <div className="mt-4 flex flex-wrap gap-1.5">
              <Badge>{DEFAULT_CANDIDATE_COUNT} images</Badge>
              <Badge>{DEFAULT_CANDIDATE_COUNT} videos</Badge>
              <Badge>ranked twice</Badge>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}

function FilePicker({
  label,
  file,
  onPick,
  hint,
  required,
}: {
  label: string;
  file: File | null;
  onPick: (file: File | null) => void;
  hint?: string;
  required?: boolean;
}) {
  const preview = useMemo(() => (file ? URL.createObjectURL(file) : null), [file]);
  return (
    <Field label={label} hint={hint} required={required}>
      <div className="space-y-2">
        <div className="flex aspect-4/5 items-center justify-center overflow-hidden rounded-md border border-dashed border-zinc-700 bg-zinc-950">
          {preview ? (
            // eslint-disable-next-line @next/next/no-img-element
            <img src={preview} alt={label} className="size-full object-cover" />
          ) : (
            <span className="text-[11px] text-zinc-600">none</span>
          )}
        </div>
        <input
          type="file"
          accept="image/png,image/jpeg,image/webp"
          onChange={(event) => onPick(event.target.files?.[0] ?? null)}
          className="w-full text-[11px] text-zinc-400 file:mr-2 file:rounded file:border-0 file:bg-zinc-800 file:px-2 file:py-1 file:text-zinc-200"
        />
      </div>
    </Field>
  );
}
