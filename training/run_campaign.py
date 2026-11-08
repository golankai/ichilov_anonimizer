"""Overnight CAMPAIGN: several training recipes x several seeds, each model scored on the base-model sets AND, right after
training, on the full inference pipeline (rules + chunking + post-processing) with the fake pipeline test set.

    python training/run_campaign.py                       # smoke test first, then the whole campaign
    python training/run_campaign.py --smoke-only          # just the quick end-to-end check (a few minutes)
    python training/run_campaign.py --only A_real B_mix_full --seeds 42
    python training/run_campaign.py --campaign night-20261109-2200     # RESUME: finished seeds/pipeline runs are skipped

Design goals: it must not die half way through the night and it must be easy to read in the morning.
  * Preflight checks (paths, GPU, disk) and a smoke test of every training mode (mixed, staged, real only) with tiny
    data BEFORE the long part. A failing smoke test stops everything with a clear message while you are still there.
  * Every experiment/seed runs in its own try/except: one crash (CUDA out of memory, bad file) is recorded in the
    registry and the campaign goes on. Re-running with ``--campaign <id>`` redoes only what is missing.
  * Model folders are named by the campaign and the recipe:  training/models/<campaign>/<experiment>/seed_<n>/
    (git-ignored). ``registry.csv/json/md`` in training/runs/<campaign>/ maps every (experiment, seed) to its model
    folder, its run record, its metrics and its pipeline scores. That table is what to bring back and analyse.
  * Only counts, metrics and fake-data scores go to training/runs/ (committed by training\\push_results.bat).

The pipeline evaluation uses inference/dummy_data (invented documents, ground truth in *.truth.jsonl), once with the
deterministic rules on (``rules``) and, for the first seed of each recipe, once with them off (``norules``) to show what
the rules add for that model.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("WANDB_MODE", "disabled")      # no login prompts at night
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import run_record  # noqa: E402
import train as T  # noqa: E402
from aggregate_runs import comparison_md, summarize_experiment  # noqa: E402

MODELS_ROOT = HERE / "models"
RUNS_ROOT = HERE / "runs"
DUMMY_DIR = ROOT / "inference" / "dummy_data"
DE_ID = ROOT / "inference" / "de_identify.py"
SCORER = ROOT / "inference" / "score_dummy.py"
SYNTH = str(T.SYNTH_TRAIN_FILE)

# The recipes. `extra`: synthetic training file; `fraction`: share of it used; `staged`: synthetic first, then real.
# Order = importance: if the night is cut short, the first rows are the ones that matter.
EXPERIMENTS = [
    {"id": "A_real", "desc": "real data only (baseline)", "extra": None},
    {"id": "B_mix_full", "desc": "real + all 3000 synthetic, shuffled together", "extra": SYNTH, "fraction": 1.0},
    {"id": "D_staged_1ep", "desc": "1 epoch on synthetic first, then real", "extra": SYNTH, "staged": True, "stage1_epochs": 1},
    {"id": "C_mix_half", "desc": "real + half of the synthetic (1500), shuffled", "extra": SYNTH, "fraction": 0.5},
    {"id": "E_mix_quarter", "desc": "real + a quarter of the synthetic (750), shuffled", "extra": SYNTH, "fraction": 0.25},
    {"id": "F_staged_2ep", "desc": "2 epochs on synthetic first, then real", "extra": SYNTH, "staged": True, "stage1_epochs": 2},
]
SMOKE_EXPERIMENTS = ["A_real", "B_mix_full", "D_staged_1ep"]       # one per training mode
TIERS = ["t1_simple", "t2_formats", "t3_traps", "t4_notes", "t5_long"]


# ------------------------------------------------------------------------------------------------ utilities
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        return len(data)

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False


def say(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def free_memory():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(k)
    return d


# ------------------------------------------------------------------------------------------------ registry
REG_COLUMNS = ["experiment", "seed", "status", "model_dir", "run_record", "description", "extra_train", "best_epoch",
               "epochs_completed", "val_f1", "test_f1", "test_f2", "rare_hard_f1", "long_f1", "ambiguous_f1",
               "ambig_seen_f1", "ambig_heldout_f1", "pipe_rules_masked", "pipe_rules_exact", "pipe_rules_leak_spans",
               "pipe_rules_traps_hit", "pipe_rules_extra_preds", "pipe_norules_masked", "pipe_norules_exact",
               "train_minutes", "error"]


class Registry:
    def __init__(self, campaign, run_dir):
        self.campaign, self.dir = campaign, Path(run_dir)
        self.path = self.dir / "registry.json"
        self.rows = {}
        if self.path.exists():
            try:
                for r in json.loads(self.path.read_text(encoding="utf-8"))["rows"]:
                    self.rows[(r["experiment"], r["seed"])] = r
            except Exception:
                say("registry.json unreadable; starting a fresh table")

    def row(self, exp, seed):
        return self.rows.setdefault((exp["id"], seed), {"experiment": exp["id"], "seed": seed, "status": "pending",
                                                        "description": exp["desc"], "extra_train": exp.get("extra")})

    def save(self):
        rows = sorted(self.rows.values(), key=lambda r: ([e["id"] for e in EXPERIMENTS].index(r["experiment"])
                                                          if r["experiment"] in [e["id"] for e in EXPERIMENTS] else 99,
                                                          r["seed"]))
        atomic_write(self.path, json.dumps({"campaign": self.campaign, "updated": datetime.now().isoformat(timespec="seconds"),
                                            "rows": rows}, indent=1, ensure_ascii=False))
        csv_path = self.dir / "registry.csv"
        tmp = csv_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=REG_COLUMNS, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        tmp.replace(csv_path)
        L = [f"# Campaign {self.campaign}: models and results", "",
             "Model folders are under `training/models/` (git-ignored, on the GPU machine). `model_dir` is relative to the repo.", "",
             "| experiment | seed | status | model_dir | best ep | val F1 | test F1 | ambig seen | ambig held-out | pipe masked | pipe exact | leaks |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        f3 = lambda x: "-" if x is None else f"{x:.3f}"
        for r in rows:
            L.append(f"| {r['experiment']} | {r['seed']} | {r['status']} | {r.get('model_dir', '-')} | {r.get('best_epoch', '-')} | "
                     f"{f3(r.get('val_f1'))} | {f3(r.get('test_f1'))} | {f3(r.get('ambig_seen_f1'))} | "
                     f"{f3(r.get('ambig_heldout_f1'))} | {f3(r.get('pipe_rules_masked'))} | {f3(r.get('pipe_rules_exact'))} | "
                     f"{r.get('pipe_rules_leak_spans', '-')} |")
        errs = [r for r in rows if r.get("error")]
        if errs:
            L += ["", "## Errors", ""] + [f"- {r['experiment']} seed {r['seed']}: {r['error']}" for r in errs]
        atomic_write(self.dir / "registry.md", "\n".join(L) + "\n")


def fill_training_metrics(row, record_path):
    rec = json.loads(Path(record_path).read_text(encoding="utf-8"))
    res = rec.get("results") or {}
    row.update({"run_record": str(Path(record_path).relative_to(ROOT)) if ROOT in Path(record_path).parents else str(record_path),
                "best_epoch": res.get("best_epoch"), "epochs_completed": res.get("epochs_completed"),
                "val_f1": res.get("best_metric_f1"), "test_f1": get(res, "test", "overall", "f1"),
                "test_f2": get(res, "test", "overall", "f2"), "train_minutes": round((rec.get("duration_seconds") or 0) / 60, 1)})
    for name in ("rare_hard", "long", "ambiguous", "ambig_seen", "ambig_heldout"):
        row[f"{name}_f1"] = get(res, "synthetic", name, "overall", "f1")


# ------------------------------------------------------------------------------------------------ pipeline evaluation
def run_cmd(cmd, log_path):
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(map(str, cmd)) + "\n")
        log.flush()
        return subprocess.run([str(c) for c in cmd], cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, env=env).returncode


def pipeline_eval(model_dir, out_dir, variant, tiers):
    """Run inference/de_identify.py on the fake tiers and score them. Returns the score summary dict.
    Raises RuntimeError when a step fails (the caller records it and goes on)."""
    out_dir = Path(out_dir)
    res_dir = out_dir / f"outputs_{variant}"
    if res_dir.exists():
        shutil.rmtree(res_dir, ignore_errors=True)
    log = out_dir / f"pipeline_{variant}.log"
    for tier in tiers:
        cmd = [sys.executable, "-X", "utf8", DE_ID, "--input", DUMMY_DIR / f"{tier}.csv", "--output-dir", res_dir,
               "--model", model_dir, "--output-format", "csv"]
        if variant == "norules":
            cmd.append("--no-rules")
        if run_cmd(cmd, log) != 0:
            raise RuntimeError(f"de_identify failed on {tier} ({variant}); see {log}")
    json_out = out_dir / f"pipeline_{variant}.json"
    cmd = [sys.executable, "-X", "utf8", SCORER, "--results-dir", res_dir, "--json-out", json_out]
    if len(tiers) == 1:
        cmd += ["--only", tiers[0]]
    if run_cmd(cmd, out_dir / f"pipeline_{variant}_scores.txt") != 0 or not json_out.exists():
        raise RuntimeError(f"score_dummy failed ({variant}); see {out_dir / f'pipeline_{variant}_scores.txt'}")
    return json.loads(json_out.read_text(encoding="utf-8"))


def fill_pipeline_metrics(row, variant, summ):
    o = summ["overall"]
    row[f"pipe_{variant}_masked"], row[f"pipe_{variant}_exact"] = o["masked"], o["exact"]
    if variant == "rules":
        row["pipe_rules_leak_spans"], row["pipe_rules_traps_hit"] = summ["leak_spans"], summ["traps_hit"]
        row["pipe_rules_extra_preds"] = summ["extra_predictions"]


# ------------------------------------------------------------------------------------------------ the campaign
def preflight(a):
    problems, warnings = [], []
    for name in ("train", "validation"):
        if T.find_split_file(a.data_dir, name) is None:
            problems.append(f"no {name} split in {a.data_dir}")
    if not (Path(a.model) / "config.json").is_file():
        problems.append(f"base model not found: {a.model}")
    for f in (T.SYNTH_TRAIN_FILE,):
        if not f.is_file():
            problems.append(f"missing {f} (python training/make_synth_train.py)")
    if not all((DUMMY_DIR / f"{t}.csv").is_file() and (DUMMY_DIR / f"{t}.truth.jsonl").is_file() for t in TIERS):
        problems.append(f"fake pipeline set incomplete in {DUMMY_DIR}")
    try:
        import torch
        if not torch.cuda.is_available():
            (warnings if a.allow_cpu else problems).append("no CUDA GPU visible to torch (use --allow-cpu to override)")
    except Exception as exc:
        problems.append(f"torch import failed: {exc}")
    MODELS_ROOT.mkdir(exist_ok=True)
    free_gb = shutil.disk_usage(MODELS_ROOT).free / 1024 ** 3
    need = 1.0 * len(a.experiments) * len(a.seeds)  # ~0.5 GB per model after checkpoint cleanup
    if free_gb < need + 5:
        problems.append(f"free disk {free_gb:.0f} GB, about {need:.0f} GB needed")
    elif free_gb < 3 * need:
        warnings.append(f"free disk {free_gb:.0f} GB, about {need:.0f} GB needed")
    for w in warnings:
        say(f"WARNING: {w}")
    if problems:
        say("PREFLIGHT FAILED:\n  - " + "\n  - ".join(problems))
        return False
    say(f"preflight ok (free disk {free_gb:.0f} GB)")
    return True


def run_one(a, exp, seed, reg, models_dir, run_dir, campaign, pipeline_variants, tiers, epochs, limit):
    row = reg.row(exp, seed)
    seed_dir = models_dir / exp["id"] / f"seed_{seed}"
    record = seed_dir / "training_run.json"
    row["model_dir"] = str(seed_dir.relative_to(ROOT)) if ROOT in seed_dir.parents else str(seed_dir)
    done_training = False
    if record.exists():
        try:
            done_training = json.loads(record.read_text(encoding="utf-8")).get("status") == "completed"
        except Exception:
            pass
    if not done_training:
        row["status"], row["error"] = "training", None
        reg.save()
        say(f"=== {exp['id']} seed {seed}: training ({exp['desc']})")
        try:
            T.train_ner(a.model, a.data_dir, str(models_dir / exp["id"]), seeds=[seed], epochs=epochs,
                        patience=T.DEFAULT_PATIENCE, extra_train=[exp["extra"]] if exp.get("extra") else None,
                        extra_fraction=exp.get("fraction", 1.0), staged=exp.get("staged", False),
                        stage1_epochs=exp.get("stage1_epochs", 1), limit=limit, run_group=campaign)
        except BaseException as exc:  # noqa: BLE001  (KeyboardInterrupt too: record it, then re-raise below)
            row["status"], row["error"] = "train_failed", f"{type(exc).__name__}: {str(exc)[:300]}"
            (run_dir / "errors.txt").open("a", encoding="utf-8").write(
                f"\n--- {exp['id']} seed {seed} ---\n{traceback.format_exc()}\n")
            reg.save()
            free_memory()
            if isinstance(exc, KeyboardInterrupt):
                raise
            say(f"!!! {exp['id']} seed {seed} training failed: {row['error']}")
            return
        free_memory()
    try:
        fill_training_metrics(row, record)
    except Exception as exc:
        row["status"], row["error"] = "train_failed", f"no readable run record: {exc}"
        reg.save()
        return
    row["status"] = "trained"
    reg.save()

    pipe_dir = run_dir / "pipeline" / exp["id"] / f"seed_{seed}"
    pipe_dir.mkdir(parents=True, exist_ok=True)
    row["error"] = None
    for variant in pipeline_variants:
        if variant == "norules" and seed != a.seeds[0]:
            continue
        done = pipe_dir / f"pipeline_{variant}.json"
        try:
            if done.exists():
                summ = json.loads(done.read_text(encoding="utf-8"))
            else:
                say(f"    pipeline evaluation ({variant}) of {exp['id']} seed {seed}")
                summ = pipeline_eval(seed_dir, pipe_dir, variant, tiers)
            fill_pipeline_metrics(row, variant, summ)
        except Exception as exc:
            row["error"] = f"pipeline {variant}: {str(exc)[:300]}"
            say(f"!!! {row['error']}")
            (run_dir / "errors.txt").open("a", encoding="utf-8").write(
                f"\n--- {exp['id']} seed {seed} pipeline {variant} ---\n{traceback.format_exc()}\n")
        free_memory()
    row["status"] = "done" if not row["error"] else "pipeline_failed"
    # the pipeline output files hold fake text only, but they are bulky: keep the scores, drop the outputs
    for d in pipe_dir.glob("outputs_*"):
        shutil.rmtree(d, ignore_errors=True)
    reg.save()
    say(f"    {exp['id']} seed {seed}: test F1 {row.get('test_f1')}, ambiguous held-out F1 {row.get('ambig_heldout_f1')}, "
        f"pipeline masked {row.get('pipe_rules_masked')}")


def write_report(reg, run_dir, campaign, experiments):
    named = []
    for exp in experiments:
        d = MODELS_ROOT / campaign / exp["id"]
        if d.exists():
            try:
                summ = summarize_experiment(d, mirror_name=f"{campaign}/{exp['id']}")
            except Exception as exc:
                say(f"summary of {exp['id']} failed: {exc}")
                summ = None
            if summ:
                named.append((exp["id"], summ))
    L = [f"# Campaign {campaign}", "", "## Base model (mean ± std over seeds)", ""]
    L.append(comparison_md(named) if named else "no completed runs\n")
    L += ["", "## Full pipeline on the fake test set (mean over seeds, rules on)", "",
          "| experiment | seeds | masked | exact | leak spans | traps hit | extra predictions |", "|---|---|---|---|---|---|---|"]
    for exp in experiments:
        rows = [r for r in reg.rows.values() if r["experiment"] == exp["id"] and r.get("pipe_rules_masked") is not None]
        if not rows:
            continue
        m = lambda k: sum(r[k] for r in rows if r.get(k) is not None) / max(1, len([r for r in rows if r.get(k) is not None]))
        L.append(f"| {exp['id']} | {len(rows)} | {m('pipe_rules_masked'):.3f} | {m('pipe_rules_exact'):.3f} | "
                 f"{m('pipe_rules_leak_spans'):.1f} | {m('pipe_rules_traps_hit'):.1f} | {m('pipe_rules_extra_preds'):.1f} |")
    L += ["", "Recipes: " + "; ".join(f"**{e['id']}**: {e['desc']}" for e in experiments), "",
          "Model folders and per-seed numbers: `registry.md` / `registry.csv`."]
    atomic_write(run_dir / "report.md", "\n".join(L) + "\n")


def run_campaign(a, campaign, experiments, seeds, epochs, limit, tiers, pipeline_variants, models_dir, run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    reg = Registry(campaign, run_dir)
    atomic_write(run_dir / "campaign.json", json.dumps(
        {"campaign": campaign, "started": datetime.now().isoformat(timespec="seconds"), "base_model": a.model,
         "data_dir": a.data_dir, "seeds": seeds, "epochs": epochs, "limit": limit, "pipeline_variants": pipeline_variants,
         "experiments": experiments, "code": run_record.code_version()},
        indent=1, ensure_ascii=False))
    started = time.time()
    try:
        for exp in experiments:
            for seed in seeds:
                if a.max_hours and (time.time() - started) / 3600 > a.max_hours:
                    say(f"time budget of {a.max_hours} h used up: not starting {exp['id']} seed {seed}")
                    reg.row(exp, seed)["status"] = "skipped_time"
                    continue
                run_one(a, exp, seed, reg, models_dir, run_dir, campaign, pipeline_variants, tiers, epochs, limit)
    finally:
        reg.save()
        try:
            write_report(reg, run_dir, campaign, experiments)
        except Exception as exc:
            say(f"report failed: {exc}")
    return reg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--campaign", default=None, help="campaign id (default night-<date>-<time>); pass an old id to RESUME it")
    ap.add_argument("--only", nargs="+", default=None, help="run just these experiment ids")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(T.DEFAULT_SEEDS))
    ap.add_argument("--epochs", type=int, default=T.DEFAULT_EPOCHS)
    ap.add_argument("--model", default=T.DEFAULT_MODEL)
    ap.add_argument("--data-dir", default=T.DEFAULT_DATA_DIR)
    ap.add_argument("--max-hours", type=float, default=0, help="stop starting new runs after this many hours (0 = no limit)")
    ap.add_argument("--no-pipeline", action="store_true", help="skip the full-pipeline evaluation")
    ap.add_argument("--skip-smoke", action="store_true", help="skip the smoke test (not recommended)")
    ap.add_argument("--smoke-only", action="store_true")
    ap.add_argument("--allow-cpu", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the recipes and exit")
    a = ap.parse_args(argv)

    if a.list:
        for e in EXPERIMENTS:
            print(f"{e['id']:16s} {e['desc']}")
        return 0
    by_id = {e["id"]: e for e in EXPERIMENTS}
    unknown = [x for x in (a.only or []) if x not in by_id]
    if unknown:
        print(f"unknown experiment(s) {unknown}; known: {list(by_id)}")
        return 2
    a.experiments = [by_id[x] for x in (a.only or by_id)]
    campaign = a.campaign or f"night-{time.strftime('%Y%m%d-%H%M')}"
    run_dir, models_dir = RUNS_ROOT / campaign, MODELS_ROOT / campaign
    models_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(models_dir / "campaign.log", "a", encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, log_file), Tee(sys.__stderr__, log_file)

    say(f"campaign {campaign}: {len(a.experiments)} recipes x seeds {a.seeds}, up to {a.epochs} epochs; log: {models_dir / 'campaign.log'}")
    if not preflight(a):
        return 2
    variants = [] if a.no_pipeline else ["rules", "norules"]

    if not a.skip_smoke:
        say("SMOKE TEST: every training mode on tiny data, with the pipeline evaluation (a few minutes)")
        smoke_name = f"_smoke-{campaign}"
        smoke_models, smoke_runs = MODELS_ROOT / smoke_name, RUNS_ROOT / smoke_name
        try:
            smoke_exps = [by_id[x] for x in SMOKE_EXPERIMENTS]
            reg = run_campaign(a, smoke_name, smoke_exps, [1], 1, 24, TIERS[:1], [] if a.no_pipeline else ["rules"],
                               smoke_models, smoke_runs)
            bad = [r for r in reg.rows.values() if r["status"] != "done"]
            if bad:
                say("SMOKE TEST FAILED, nothing else was started:")
                for r in bad:
                    say(f"  {r['experiment']}: {r['status']} {r.get('error')}")
                say(f"details: {smoke_runs / 'errors.txt'} (kept for inspection; delete training/models/{smoke_name} and training/runs/{smoke_name} afterwards)")
                return 3
            say("smoke test passed")
        except BaseException as exc:  # noqa: BLE001
            say(f"SMOKE TEST CRASHED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            return 3
        shutil.rmtree(smoke_models, ignore_errors=True)
        shutil.rmtree(smoke_runs, ignore_errors=True)
        if a.smoke_only:
            say("smoke-only: done. Nothing to push.")
            return 0

    say(f"MAIN CAMPAIGN {campaign}")
    reg = run_campaign(a, campaign, a.experiments, a.seeds, a.epochs, None, TIERS, variants, models_dir, run_dir)
    states = {}
    for r in reg.rows.values():
        states[r["status"]] = states.get(r["status"], 0) + 1
    say(f"FINISHED {campaign}: {states}")
    say(f"results: {run_dir / 'report.md'} and {run_dir / 'registry.md'}")
    say("NEXT: run training\\push_results.bat   (then tell Claude the campaign id: " + campaign + ")")
    return 0


if __name__ == "__main__":
    sys.exit(main())
