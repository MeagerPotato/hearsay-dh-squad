"""HEARSAY command line.

    python -m hearsay.cli predict DIR --out submissions/x.tsv --meta submissions/<ver>/fusion_meta.json \
        --template data/.../HGT_Hearsay_score_template.csv [--reports reports/]   # the submission pipeline
    python -m hearsay.cli score  DIR --out submissions/x.tsv --template data/.../HGT_Hearsay_score_template.csv
    python -m hearsay.cli explain FILE --html out.html
    python -m hearsay.cli report DIR --out reports/
    python -m hearsay.cli compute DIR --limit 20        # agentic vs full compute on a sample
    python -m hearsay.cli experts                       # what is installed
    python -m hearsay.cli policy                        # the routing policy in plain text

Shared options: --mode agentic|full, --deep-cache work/aasist_scores.json (reference deep scorer
backed by cached AASIST scores), --deep pkg.mod:callable (a live deep_scorer(ctx) -> Evidence),
--fusion pkg.mod:callable (fusion(features, evidence) -> P(synthetic)), --transcripts file.json.
"""
from __future__ import annotations

import json
import math
import textwrap
from enum import Enum
from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

from hearsay import orchestrator as orch

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="HEARSAY agentic audio-deepfake forensics: score, explain and report.")

DEFAULT_TEMPLATE = "data/HackGTHearsayTesting/HGT_Hearsay_score_template.csv"


class Mode(str, Enum):
    agentic = "agentic"
    full = "full"


# ---- shared option definitions -------------------------------------------------------------------
MODE_OPT = typer.Option(Mode.agentic, "--mode", help="agentic = policy-driven routing; full = run every expert.")
DEEP_CACHE_OPT = typer.Option(None, "--deep-cache", help="JSON of cached deep-detector log-odds "
                              "({model: {filename: score}}), e.g. work/aasist_scores.json.")
DEEP_OPT = typer.Option(None, "--deep", help="Import path 'pkg.module:callable' of a deep_scorer(ctx) -> Evidence.")
FUSION_OPT = typer.Option(None, "--fusion", help="Import path 'pkg.module:callable' of fusion(features, evidence) -> p.")
TRANSCRIPTS_OPT = typer.Option(None, "--transcripts", help="JSON mapping filename -> transcript (skips ASR).")
LIMIT_OPT = typer.Option(None, "--limit", help="Only analyse the first N files (for quick tests).")
SKIP_OPT = typer.Option(None, "--skip", help="Expert name to never run (repeatable), e.g. --skip transcript "
                        "to avoid running Whisper on files without a cached transcript.")


def _cfg(skip: Optional[List[str]]) -> orch.RouterConfig:
    return orch.RouterConfig(disabled_experts=tuple(skip or ()))


def _deep(deep_cache: Optional[Path], deep: Optional[str]):
    if deep:
        obj = orch.load_callable(deep)
        # Accept either a scorer or a zero-argument factory returning one.
        try:
            import inspect
            if inspect.isclass(obj) or (callable(obj) and not inspect.signature(obj).parameters):
                obj = obj()
        except (TypeError, ValueError):
            pass
        return obj
    if deep_cache:
        return orch.CachedDeepScorer(str(deep_cache))
    return None


def _fusion(spec: Optional[str]):
    return orch.load_callable(spec) if spec else None


def _echo_accounting(acc: dict) -> None:
    n = max(1, acc["n_files"])
    tiers = ", ".join(f"{t}={c} ({100 * c / n:.0f}%)" for t, c in acc["files_per_tier"].items() if c)
    typer.echo(f"files per tier: {tiers}")
    frac = acc.get("agentic_fraction_of_full")
    if frac is not None:
        unpriced = acc.get("unpriced_experts") or []
        typer.echo(f"analysis compute: {acc['analysis_total_ms'] / 1000:.1f} s of an estimated "
                   f"{acc['full_estimate_analysis_ms'] / 1000:.1f} s for running every expert "
                   f"({100 * frac:.0f}%" + (f"; upper bound, never-run experts are not priced: {', '.join(unpriced)}; "
                                            "use the 'compute' command for a measured ratio" if unpriced else "") + ")")


# ---- commands ------------------------------------------------------------------------------------

@app.command()
def predict(directory: Path = typer.Argument(..., help="Folder of audio files (searched recursively) or one file."),
            out: Path = typer.Option(Path("submissions/hearsay_predictions.tsv"), "--out", help="Output TSV."),
            meta: Path = typer.Option(..., "--meta", help="fusion_meta.json from scripts/make_submission.py: which "
                                      "detectors run, their weights and the calibration."),
            template: Optional[Path] = typer.Option(None, "--template", help="Organizer template (fixes the file "
                                                    "order; every file it lists must be in DIRECTORY)."),
            reports: Optional[Path] = typer.Option(None, "--reports", help="Also write per-file HTML forensic "
                                                   "reports + dashboard here (agentic orchestrator)."),
            self_train: Optional[bool] = typer.Option(None, "--self-train/--no-self-train",
                                                      help="Transductive self-training of the WavLM head "
                                                      "(default: on when the fusion config uses it)."),
            device: str = typer.Option("auto", "--device", help="auto (cuda, else mps, else cpu), cuda, mps or cpu."),
            cache_dir: Path = typer.Option(Path("work/cache_predict"), "--cache-dir",
                                           help="Per-model score cache; re-runs only compute what changed."),
            no_cache: bool = typer.Option(False, "--no-cache", help="Neither read nor write the score cache."),
            threads: Optional[int] = typer.Option(None, "--threads", help="CPU threads per model "
                                                  "(default: OMP_NUM_THREADS, else all cores up to 8)."),
            workers: Optional[int] = typer.Option(None, "--workers", help="Processes for the forensic features."),
            transcripts: Optional[Path] = TRANSCRIPTS_OPT, skip: Optional[List[str]] = SKIP_OPT,
            asr: bool = typer.Option(False, "--asr", help="Reports: allow Whisper ASR for files without a cached "
                                     "transcript (otherwise the transcript expert is skipped)."),
            limit: Optional[int] = LIMIT_OPT, quiet: bool = typer.Option(False, "--quiet")):
    """End-to-end inference: run the detectors named in the fusion config, fuse and calibrate exactly as
    scripts/make_submission.py, and write the TSV plus a JSON of every score (and optional reports)."""
    from hearsay import pipeline

    res = pipeline.predict(directory, out, meta, template=template, reports_dir=reports, self_train=self_train,
                           device=device, cache_dir=None if no_cache else cache_dir, threads=threads,
                           workers=workers, transcripts=transcripts, skip=skip, asr=asr, limit=limit, quiet=quiet)
    p = np.array(list(res["probabilities"].values()))
    typer.echo(f"wrote {res['tsv']} ({len(p)} files; {100 * np.mean(p > 0.5):.1f}% with cm-score > 0.5) "
               f"and {res['scores_json']}")
    for key, t in res["timing"].items():
        per = f", {t['s_per_file']:.3f} s/file" if t.get("s_per_file") is not None else ""
        typer.echo(f"  {key:<16} {t['n_computed']:>5} computed, {t['n_cached']:>5} cached "
                   f"(load {t['load_s']:.1f} s, scoring {t['score_s']:.1f} s{per})")
    if res.get("reports_index"):
        typer.echo(f"reports: {res['reports_index']}")
        _echo_accounting(res["accounting"])
    typer.echo(f"total {res['wall_s']:.1f} s")


@app.command()
def score(directory: Path = typer.Argument(..., help="Folder of audio files (searched recursively)."),
          out: Path = typer.Option(Path("submissions/hearsay_predictions.tsv"), "--out", help="Output TSV."),
          template: Optional[Path] = typer.Option(Path(DEFAULT_TEMPLATE), "--template",
                                                  help="Organizer template (fixes filename order)."),
          mode: Mode = MODE_OPT, deep_cache: Optional[Path] = DEEP_CACHE_OPT, deep: Optional[str] = DEEP_OPT,
          fusion: Optional[str] = FUSION_OPT, transcripts: Optional[Path] = TRANSCRIPTS_OPT,
          json_out: Optional[Path] = typer.Option(None, "--json-out", help="Also export all evidence as JSON."),
          limit: Optional[int] = LIMIT_OPT, skip: Optional[List[str]] = SKIP_OPT,
          quiet: bool = typer.Option(False, "--quiet")):
    """Score every file and write the HEARSAY submission TSV (filename, cm-score; 1.0 = synthetic)."""
    from hearsay import submission

    results = orch.run_batch(directory, deep_scorer=_deep(deep_cache, deep), fusion=_fusion(fusion),
                             mode=mode.value, config=_cfg(skip), transcripts=str(transcripts) if transcripts else None,
                             json_out=str(json_out) if json_out else None, progress=not quiet, limit=limit)
    names = [r.name for r in results]
    # minDCF is rank-based: map cm-scores through a monotone transform that also breaks exact ties.
    logits = np.array([math.log(max(1e-9, r.cm_score) / max(1e-9, 1 - r.cm_score)) for r in results])
    probs = dict(zip(names, submission.to_probability(logits))) if len(results) > 1 else \
        {n: float(r.cm_score) for n, r in zip(names, results)}
    tmpl_names = submission.read_template(template) if template and template.exists() else None
    if tmpl_names is not None and set(tmpl_names) <= set(probs):
        path = submission.write(probs, template, out)
        typer.echo(f"wrote {path} ({len(tmpl_names)} files, validated against the template)")
    else:
        if tmpl_names is not None:
            typer.echo(f"warning: {len(set(tmpl_names) - set(probs))} template files were not scored; "
                       "writing an unvalidated TSV of the scored files", err=True)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as fh:
            fh.write("filename\tcm-score\n")
            for n in sorted(probs):
                fh.write(f"{n}\t{probs[n]:.12f}\n")
        typer.echo(f"wrote {out} ({len(probs)} files)")
    _echo_accounting(orch.compute_accounting(results))


@app.command()
def explain(file: Path = typer.Argument(..., help="One audio file (any format ffmpeg reads)."),
            html: Optional[Path] = typer.Option(None, "--html", help="Write the forensic HTML report here."),
            mode: Mode = MODE_OPT, deep_cache: Optional[Path] = DEEP_CACHE_OPT, deep: Optional[str] = DEEP_OPT,
            fusion: Optional[str] = FUSION_OPT, transcripts: Optional[Path] = TRANSCRIPTS_OPT,
            skip: Optional[List[str]] = SKIP_OPT,
            as_json: bool = typer.Option(False, "--json", help="Print the full result as JSON instead.")):
    """Analyse one file and explain the verdict (optionally as an HTML report)."""
    trans = orch.load_transcripts(str(transcripts)) if transcripts else {}
    # With a transcript collection, one file can still be compared with the batch (repeated scripts).
    corpus = orch.build_corpus([str(file)], trans)[0] if trans else None
    res = orch.run(str(file), deep_scorer=_deep(deep_cache, deep), fusion=_fusion(fusion), mode=mode.value,
                   config=_cfg(skip), transcript=trans.get(file.name), corpus=corpus)
    if as_json:
        typer.echo(json.dumps(res.to_dict(), indent=1))
    else:
        width = 100
        typer.echo(f"{res.name}: cm-score {res.cm_score:.3f}  |  {res.manipulation_type}  |  source: {res.score_source}"
                   f"  |  {res.runtime_ms:.0f} ms")
        typer.echo(textwrap.fill(res.summary, width))
        typer.echo("\nFindings (strongest first):")
        rows = sorted(((e, f) for e in res.evidence for f in e.findings), key=lambda t: -float(t[1].strength or 0))
        for e, f in rows:
            arrow = "+syn " if f.direction > 0 else "-real" if f.direction < 0 else " neu "
            typer.echo(textwrap.fill(f"[{arrow} {float(f.strength or 0):.2f}] {e.expert}: {f.text}", width,
                                     subsequent_indent="             "))
        for e in res.evidence:
            if e.error:
                typer.echo(f"[error] {e.expert}: {e.error}")
        typer.echo("\nRouting trace:")
        for t in res.trace:
            ran = ", ".join(t.get("experts_run") or []) or "-"
            typer.echo(textwrap.fill(f"{t['step']:>2} {t['tier']:<11} {t['rule']:<20} "
                                     f"{'RAN ' if t['fired'] else 'skip'}  {t['reason']}  -> {ran} "
                                     f"({t.get('ms', 0):.0f} ms)", width,
                                     subsequent_indent=" " * 40))
    if html:
        from hearsay.report import render_file_report
        render_file_report(res, str(html))
        typer.echo(f"\nwrote {html}", err=as_json)


@app.command()
def report(directory: Path = typer.Argument(..., help="Folder of audio files."),
           out: Path = typer.Option(Path("reports"), "--out", help="Output folder (index.html + files/)."),
           mode: Mode = MODE_OPT, deep_cache: Optional[Path] = DEEP_CACHE_OPT, deep: Optional[str] = DEEP_OPT,
           fusion: Optional[str] = FUSION_OPT, transcripts: Optional[Path] = TRANSCRIPTS_OPT,
           limit: Optional[int] = LIMIT_OPT, skip: Optional[List[str]] = SKIP_OPT,
           quiet: bool = typer.Option(False, "--quiet")):
    """Per-file HTML forensic reports plus a batch dashboard (index.html) and evidence.json."""
    from hearsay.report import make_report_writer, render_index

    cfg = _cfg(skip)
    results = orch.run_batch(directory, deep_scorer=_deep(deep_cache, deep), fusion=_fusion(fusion),
                             mode=mode.value, config=cfg, transcripts=str(transcripts) if transcripts else None,
                             on_result=make_report_writer(str(out), cfg), progress=not quiet, limit=limit)
    acc = orch.compute_accounting(results)
    orch.export_json(results, str(out / "evidence.json"))
    index = render_index(results, str(out), acc, config=cfg)
    typer.echo(f"wrote {index} and {len(results)} file reports under {out / 'files'}")
    _echo_accounting(acc)


@app.command()
def compute(directory: Path = typer.Argument(..., help="Folder of audio files."),
            limit: int = typer.Option(20, "--limit", help="Sample size (both modes run on it)."),
            deep_cache: Optional[Path] = DEEP_CACHE_OPT, deep: Optional[str] = DEEP_OPT,
            transcripts: Optional[Path] = TRANSCRIPTS_OPT, skip: Optional[List[str]] = SKIP_OPT,
            json_out: Optional[Path] = typer.Option(None, "--json-out", help="Save both accountings as JSON.")):
    """Measure agentic vs full compute on a sample (for the README 'agentic runs X% of full' chart)."""
    scorer = _deep(deep_cache, deep)
    tr = str(transcripts) if transcripts else None
    cfg = _cfg(skip)
    # Untimed warm-up (imports, numba JIT, model loads, cached corpus statistics): otherwise these one-off
    # costs are charged to whichever mode runs first and bias the agentic/full ratio.
    first = orch.list_audio_files(directory)[:1]
    if first:
        orch.run(first[0], deep_scorer=scorer, mode="full", config=cfg,
                 transcript=orch.load_transcripts(tr).get(Path(first[0]).name))
    full = orch.run_batch(directory, deep_scorer=scorer, mode="full", config=cfg, transcripts=tr, limit=limit,
                          progress=False)
    acc_full = orch.compute_accounting(full)
    agentic = orch.run_batch(directory, deep_scorer=scorer, mode="agentic", config=cfg, transcripts=tr,
                             limit=limit, progress=False)
    acc_ag = orch.compute_accounting(agentic, reference_ms=acc_full["expert_mean_ms"])
    frac = acc_ag["analysis_total_ms"] / max(1e-9, acc_full["analysis_total_ms"])
    typer.echo(f"{len(full)} files | full mode: {acc_full['analysis_total_ms'] / 1000:.1f} s of expert analysis | "
               f"agentic mode: {acc_ag['analysis_total_ms'] / 1000:.1f} s  ->  agentic runs {100 * frac:.0f}% "
               f"of the compute of full mode")
    _echo_accounting(acc_ag)
    same = np.mean([abs(a.cm_score - b.cm_score) < 1e-9 for a, b in zip(full, agentic)]) if full else float("nan")
    typer.echo(f"identical cm-scores in both modes: {100 * same:.0f}% of files "
               "(only differs when a fusion or the vote uses escalation-only evidence)")
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(orch._jsonable({"full": acc_full, "agentic": acc_ag,
                                                       "agentic_fraction_measured": frac}), indent=1))


@app.command()
def experts():
    """List the registered forensic experts (and any modules that failed to import)."""
    errors = orch.ensure_experts_loaded()
    from hearsay.evidence import REGISTRY

    for spec in sorted(REGISTRY.values(), key=lambda s: (orch.TIER_ORDER.index(s.cost) if s.cost in orch.TIER_ORDER
                                                          else 9, s.category, s.name)):
        desc = " ".join((spec.description or "").split())
        typer.echo(f"{spec.name:<22} {spec.category:<12} {spec.cost:<9} {desc[:110]}")
    for mod, err in errors.items():
        typer.echo(f"[import failed] {mod}: {err}", err=True)
    if not REGISTRY:
        typer.echo("no experts registered", err=True)


@app.command()
def policy():
    """Print the agentic routing policy, suppression rules and attribution rules."""
    typer.echo(orch.describe_policy())


def main():  # console-script entry point
    app()


if __name__ == "__main__":
    main()
