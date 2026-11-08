"""Aggregate the per-seed run records of one experiment into a summary table (mean, std, min, max, per seed).

``train.py train`` calls this at the end. It can also be run by hand, e.g. on the dev machine after a ``git pull``
(the committed copies under ``training/runs/<experiment>/seed_<n>/`` hold everything it needs)::

    python training/aggregate_runs.py training/runs/model_non_anon_20261108-1700
    python training/aggregate_runs.py training/runs/expA training/runs/expB     # also prints a comparison table

Writes ``summary.json`` and ``summary.md`` into the experiment folder (and a copy under ``training/runs/``).
Counts and metrics only, never document text. A folder that holds a single ``training_run.json`` (no seeds) works too;
standard deviations are then shown as "n=1".
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

RUNS_DIR = Path(__file__).resolve().parent / "runs"
RECORD_NAME = "training_run.json"
OVERALL = ("precision", "recall", "f1", "f2")
LABELS = ["FIRST_NAME", "LAST_NAME", "DATE", "AGE_ABOVE_89", "LOC", "ORG", "MED_ORG", "CONTACT", "ID", "INNER_ID"]


# ------------------------------------------------------------------------------------------------ loading
def _upgrade_v1(rec):
    """Schema 1 records (flat metric keys) -> the nested shape of schema 2."""
    from metrics import nest_metrics

    res = rec.get("results") or {}
    per_epoch = []
    for h in res.get("per_epoch_validation", []):
        n = nest_metrics(h, "eval")
        per_epoch.append({"epoch": h.get("epoch"), "step": h.get("step"), "eval_loss": n["other"].get("loss"),
                          "overall": n["overall"], "per_label": n["per_label"]})
    best = res.get("best_model_checkpoint") or ""
    tail = best.replace("\\", "/").rsplit("-", 1)[-1]
    best_epoch = next((e["epoch"] for e in per_epoch if str(e["step"]) == tail), None)
    res = dict(res, per_epoch_validation=per_epoch, best_epoch=best_epoch,
               test=nest_metrics(res["test"], "test") if res.get("test") else None, synthetic={})
    return dict(rec, results=res, seed=rec.get("seed", (rec.get("hyperparameters") or {}).get("seed")))


def load_experiment(exp_dir):
    """Completed run records of one experiment: ``seed_*/training_run.json``, or a lone ``training_run.json``."""
    exp_dir = Path(exp_dir)
    paths = sorted(exp_dir.glob(f"seed_*/{RECORD_NAME}"), key=lambda p: int(p.parent.name.split("_")[1])
                   if p.parent.name.split("_")[-1].isdigit() else 0)
    if not paths and (exp_dir / RECORD_NAME).exists():
        paths = [exp_dir / RECORD_NAME]
    runs = []
    for p in paths:
        rec = json.loads(p.read_text(encoding="utf-8"))
        if rec.get("status") != "completed" or not rec.get("results"):
            print(f"  skipped {p} (status {rec.get('status')})")
            continue
        if rec.get("schema_version", 1) < 2:
            rec = _upgrade_v1(rec)
        runs.append(rec)
    return runs


# ------------------------------------------------------------------------------------------------ statistics
def stats(values):
    """mean/std/min/max over the non-None values; std is the sample std (n-1), None for a single value."""
    v = [x for x in values if x is not None]
    if not v:
        return None
    n = len(v)
    mean = sum(v) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in v) / (n - 1)) if n > 1 else None
    return {"n": n, "mean": mean, "std": std, "min": min(v), "max": max(v)}


def _best_epoch_entry(rec):
    res = rec["results"]
    per_epoch = res.get("per_epoch_validation") or []
    if not per_epoch:
        return None
    be = res.get("best_epoch")
    hit = next((e for e in per_epoch if e["epoch"] == be), None)
    return hit or max(per_epoch, key=lambda e: e["overall"].get("f1") or 0)


def run_metrics(rec):
    """One run -> {"overall": {section: {metric: value}}, "per_label": {section: {label: {metric: value}}}, "info": {}}."""
    res = rec["results"]
    overall, per_label = {}, {}
    best = _best_epoch_entry(rec)
    if best:
        overall["validation (best epoch)"] = best["overall"]
        per_label["validation (best epoch)"] = best["per_label"]
    if res.get("test"):
        overall["test"] = res["test"]["overall"]
        per_label["test"] = res["test"]["per_label"]
    for name, nested in (res.get("synthetic") or {}).items():
        overall[f"synthetic: {name}"] = nested["overall"]
        per_label[f"synthetic: {name}"] = nested["per_label"]
    info = {"seed": rec.get("seed"), "best_epoch": res.get("best_epoch"), "epochs_completed": res.get("epochs_completed"),
            "stopped_early": res.get("stopped_early"), "minutes": round((rec.get("duration_seconds") or 0) / 60, 1),
            "commit": ((rec.get("code") or {}).get("git_commit") or "")[:7],
            "uncommitted": (rec.get("code") or {}).get("training_code_uncommitted_changes")}
    return {"overall": overall, "per_label": per_label, "info": info}


def aggregate(runs):
    per_run = [run_metrics(r) for r in runs]
    seeds = [r["info"]["seed"] for r in per_run]
    overall, per_label = {}, {}
    for section in per_run[0]["overall"]:
        overall[section] = {}
        for m in OVERALL:
            vals = {str(r["info"]["seed"]): r["overall"].get(section, {}).get(m) for r in per_run}
            overall[section][m] = {"stats": stats(vals.values()), "values": vals}
    for section in per_run[0]["per_label"]:
        per_label[section] = {}
        labels = sorted({lab for r in per_run for lab in r["per_label"].get(section, {})},
                        key=lambda x: (LABELS.index(x) if x in LABELS else 99, x))
        for lab in labels:
            row = {}
            for m in ("precision", "recall", "f1"):
                vals = {str(r["info"]["seed"]): r["per_label"].get(section, {}).get(lab, {}).get(m) for r in per_run}
                row[m] = {"stats": stats(vals.values()), "values": vals}
            sup = [r["per_label"].get(section, {}).get(lab, {}).get("support") for r in per_run]
            row["support"] = next((s for s in sup if s is not None), 0)
            per_label[section][lab] = row
    first = runs[0]
    hp = first.get("hyperparameters", {})
    return {
        "n_seeds": len(runs), "seeds": seeds,
        "base_model": (first.get("base_model") or {}).get("name_or_path"),
        "hyperparameters": {k: hp.get(k) for k in ("learning_rate", "per_device_train_batch_size", "num_train_epochs",
                                                   "weight_decay", "lr_scheduler_type", "warmup_ratio")},
        "early_stopping_patience": first.get("early_stopping_patience"),
        "data": {n: {"sha256": (s or {}).get("sha256"), "n_documents": (s or {}).get("n_documents"),
                     "n_entities": (s or {}).get("n_entities")}
                 for n, s in ((first.get("data") or {}).get("splits") or {}).items()},
        "synthetic_eval_sets": {n: {"sha256": s.get("sha256"), "n_documents": s.get("n_documents"),
                                    "n_entities": s.get("n_entities")}
                                for n, s in ((first.get("data") or {}).get("synthetic_eval_sets") or {}).items()},
        "per_seed": [r["info"] for r in per_run],
        "overall": overall, "per_label": per_label,
    }


# ------------------------------------------------------------------------------------------------ rendering
def fmt(st, digits=3):
    if not st:
        return "-"
    if st["std"] is None:
        return f"{st['mean']:.{digits}f} (n=1)"
    return f"{st['mean']:.{digits}f} ± {st['std']:.{digits}f}"


def rng(st, digits=3):
    return "-" if not st else f"{st['min']:.{digits}f}–{st['max']:.{digits}f}"


def render_md(name, summ):
    L = [f"# Training summary: {name}", ""]
    hp = summ["hyperparameters"]
    L.append(f"- seeds: {', '.join(str(s) for s in summ['seeds'])} ({summ['n_seeds']} runs); base model: `{summ['base_model']}`")
    L.append(f"- lr {hp['learning_rate']}, batch {hp['per_device_train_batch_size']}, max epochs {hp['num_train_epochs']}, "
             f"weight decay {hp['weight_decay']}, early stopping patience {summ['early_stopping_patience']}")
    L.append("- mean ± sample std over seeds (n-1). The synthetic sets never select the epoch.")
    L += ["", "## Overall (entity level)", "", "| set | metric | mean ± std | min–max | per seed |", "|---|---|---|---|---|"]
    for section, metrics in summ["overall"].items():
        for m in OVERALL:
            vals = metrics[m]["values"]
            per = ", ".join(f"{s}: {v:.3f}" for s, v in vals.items() if v is not None)
            L.append(f"| {section} | {m} | {fmt(metrics[m]['stats'])} | {rng(metrics[m]['stats'])} | {per} |")
    for section, labels in summ["per_label"].items():
        if not labels:  # e.g. records written before per-label metrics existed
            continue
        L += ["", f"## Per label: {section}", "", "| label | support | precision | recall | F1 | recall min–max |",
              "|---|---|---|---|---|---|"]
        for lab, row in labels.items():
            L.append(f"| {lab} | {row['support']} | {fmt(row['precision']['stats'])} | {fmt(row['recall']['stats'])} | "
                     f"{fmt(row['f1']['stats'])} | {rng(row['recall']['stats'])} |")
        if any(r["support"] == 0 for r in labels.values()):
            L.append("")
            L.append("support 0: the label does not occur in this set, so recall/F1 are undefined; "
                     "precision shows spurious predictions of it.")
    L += ["", "## Per seed", "", "| seed | best epoch | epochs run | early stop | minutes | commit |", "|---|---|---|---|---|---|"]
    for i in summ["per_seed"]:
        c = i["commit"] + ("+local changes" if i.get("uncommitted") else "")
        L.append(f"| {i['seed']} | {i['best_epoch']} | {i['epochs_completed']} | {i['stopped_early']} | {i['minutes']} | {c} |")
    sets = summ.get("synthetic_eval_sets") or {}
    if sets:
        L += ["", "Synthetic sets (sha256 prefix): " + ", ".join(
            f"{n} ({s['n_documents']} docs, {s['n_entities']} entities, {str(s['sha256'])[:10]})" for n, s in sets.items())]
    return "\n".join(L) + "\n"


KEY_ROWS = [("validation (best epoch)", "f1"), ("test", "precision"), ("test", "recall"), ("test", "f1"), ("test", "f2"),
            ("synthetic: rare_hard", "recall"), ("synthetic: rare_hard", "f1"),
            ("synthetic: long", "recall"), ("synthetic: long", "f1")]


def comparison_md(named):
    """One row per experiment, the headline metrics as mean ± std."""
    cols = [f"{s.replace('validation (best epoch)', 'val').replace('synthetic: ', 'syn ')} {m}" for s, m in KEY_ROWS]
    L = ["| experiment | seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, summ in named:
        cells = [fmt(summ["overall"].get(s, {}).get(m, {}).get("stats")) for s, m in KEY_ROWS]
        L.append(f"| {name} | {summ['n_seeds']} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------ entry points
def summarize_experiment(exp_dir, mirror_name=None):
    """Write summary.json / summary.md for the runs under `exp_dir` (and a copy under training/runs/<mirror_name>).
    Returns the summary dict, or None if there is no completed run."""
    exp_dir = Path(exp_dir)
    runs = load_experiment(exp_dir)
    if not runs:
        print(f"no completed runs under {exp_dir}")
        return None
    summ = aggregate(runs)
    md = render_md(exp_dir.name, summ)
    targets = [exp_dir]
    if mirror_name:
        targets.append(RUNS_DIR / mirror_name)
    for t in targets:
        try:
            t.mkdir(parents=True, exist_ok=True)
            (t / "summary.json").write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")
            (t / "summary.md").write_text(md, encoding="utf-8")
        except OSError as exc:
            print(f"could not write summary to {t}: {exc}")
    print("\n" + md)
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", help="experiment folder(s) holding seed_<n>/training_run.json")
    args = ap.parse_args(argv)
    named = []
    for e in args.experiments:
        d = Path(e)
        inside_runs = RUNS_DIR in d.resolve().parents or d.resolve() == RUNS_DIR
        summ = summarize_experiment(d, mirror_name=None if inside_runs else d.name)
        if summ:
            named.append((d.name, summ))
    if len(named) > 1:
        print("\n# Comparison\n\n" + comparison_md(named))


if __name__ == "__main__":
    main()
