"""Fine-tune a Hebrew BERT model for de-identification NER.

Run from anywhere; see ``training/README.md`` for the full workflow::

    python training/train.py train                      # defaults: model_non_anon + NER_anon_dataset/regular,
                                                        # 6 epochs, early stopping, 3 seeds (42 43 44), summary table
    python training/train.py train --seeds 42           # one seed only (quick check)
    python training/train.py train --model <hf-id-or-path> --data-dir <split folder> --output-dir models/X
    python training/train.py split --data data/annotated.jsonl --out data/splits
    python training/train.py cv    --data data/annotated.jsonl -k 5
    python training/train.py final --data data/annotated.jsonl
    python training/train.py train-all --epochs <best epoch>   # production: all splits merged

Every run writes ``training_run.json`` next to the saved model (see run_record.py). `train` writes one folder per
seed plus ``summary.json`` / ``summary.md`` (mean and std over seeds; see aggregate_runs.py).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import time
from pathlib import Path

from sklearn.model_selection import KFold, train_test_split
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from aggregate_runs import summarize_experiment
from dataset import (SPLIT_NAMES, find_split_file, load_extra_eval_sets, load_records, load_splits, prepare_dataset,
                     write_splits)
from labels import ALL_LABELS, ID_2_LABEL, LABEL_2_ID
from metrics import compute_metrics
from run_record import RunRecord, describe_records, describe_split

SEED = 42
DEFAULT_SEEDS = (42, 43, 44)   # `train` runs every model with all three and reports mean/std
DEFAULT_EPOCHS = 6
DEFAULT_PATIENCE = 2           # evaluations without a better validation F1 before stopping (0 = no early stopping)
SYNTH_EVAL_DIR = Path(__file__).resolve().parent / "synth_eval"
SYNTH_EVAL_AMBIGUOUS_DIR = Path(__file__).resolve().parent / "synth_eval_ambiguous"
DEFAULT_SYNTH_DIRS = (SYNTH_EVAL_DIR, SYNTH_EVAL_AMBIGUOUS_DIR)
SYNTH_TRAIN_DIR = Path(__file__).resolve().parent / "synth_train"
SYNTH_TRAIN_FILE = SYNTH_TRAIN_DIR / "synth_train.jsonl"
SYNTH_TRAIN_ALL_FILE = SYNTH_TRAIN_DIR / "synth_train_all.jsonl"   # production variant: every ambiguous word

# Locations on the GPU machine (override with --model / --data-dir).
DEFAULT_MODEL = r"D:\HeMed All\HeMed Final Model\model_non_anon"
DEFAULT_DATA_DIR = r"D:\De ID All\NER_anon_dataset\regular"


def default_output_dir(model_name):
    """training/models/<base model name>_<timestamp>: every run gets its own folder."""
    return str(Path(__file__).resolve().parent / "models" / f"{Path(str(model_name)).name}_{time.strftime('%Y%m%d-%H%M')}")


def get_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(ALL_LABELS),
        id2label=ID_2_LABEL,
        label2id=LABEL_2_ID,
        # Checkpoints that are already NER models ship a head of a different size.
        ignore_mismatched_sizes="ner" in str(model_name).lower(),
    )
    return tokenizer, model


def make_args(output_dir, *, validate, lr=5e-5, batch_size=8, epochs=DEFAULT_EPOCHS,
              weight_decay=0.01, save_total_limit=2, seed=SEED):
    """TrainingArguments; with `validate`, evaluate each epoch and keep the best by F1."""
    kwargs = dict(
        output_dir=str(output_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=weight_decay,
        save_total_limit=save_total_limit,
        seed=seed,
        data_seed=seed,
    )
    if validate:
        kwargs.update(eval_strategy="epoch", save_strategy="epoch",
                      load_best_model_at_end=True, metric_for_best_model="f1")
    else:
        kwargs.update(eval_strategy="no", save_strategy="no")
    return TrainingArguments(**kwargs)


def run_training(model_name, train_ds, val_ds, output_dir, *, early_stopping=False, patience=DEFAULT_PATIENCE,
                 kind="train", data_info=None, test_ds=None, extra_eval=None, notes=None, seed=SEED,
                 run_name=None, **hparams):
    """Train, save the model to `output_dir`, write ``training_run.json`` beside it.

    If `test_ds` is given, the final (best) model is also scored on it. `extra_eval` is
    ``{name: dataset}`` of synthetic sets, scored the same way and reported separately; they never
    influence which epoch is kept. `early_stopping` only has an effect when there is a validation set.
    Returns the Trainer.
    """
    set_seed(seed)  # before the model is built, so the new classifier head is initialised from the seed
    tokenizer, model = get_tokenizer_and_model(model_name)
    validate = val_ds is not None
    use_early_stopping = bool(early_stopping and validate and patience and patience > 0)
    args = make_args(output_dir, validate=validate, seed=seed, **hparams)
    record = RunRecord(output_dir, kind=kind, base_model=model_name, data=data_info, notes=notes, seed=seed,
                       run_name=run_name)
    record.doc["early_stopping_patience"] = patience if use_early_stopping else None
    record.start(args)
    try:
        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)] if use_early_stopping else None,
        )
        print(f"------------- Start training: {model_name} -> {output_dir} (seed {seed}) -------------")
        trainer.train()
        trainer.save_model(str(output_dir))
        for ckpt in Path(output_dir).glob("checkpoint-*"):   # the best model is saved above; checkpoints are ~3 GB each
            shutil.rmtree(ckpt, ignore_errors=True)
        test_metrics = None
        if test_ds is not None:
            test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")
            print(f"Test metrics: {test_metrics}")
        synthetic_metrics = {}
        for set_name, set_ds in (extra_eval or {}).items():
            m = trainer.evaluate(set_ds, metric_key_prefix=set_name)
            synthetic_metrics[set_name] = m
            print(f"Synthetic set {set_name}: P {m.get(set_name + '_precision', 0):.3f} "
                  f"R {m.get(set_name + '_recall', 0):.3f} F1 {m.get(set_name + '_f1', 0):.3f}")
        record.finish(trainer, test_metrics, synthetic_metrics)
    except BaseException as exc:
        record.fail(exc)
        raise
    print(f"Run documented in {record.path}")
    return trainer


def load_extra_train(paths, fraction=1.0, seed=0):
    """Synthetic TRAINING records from `paths` (jsonl, same format as the real data); `fraction` keeps a fixed random
    subset (always the same one for the same fraction, so seeds differ only in the training randomness)."""
    records = []
    for path in paths or []:
        records += load_records(path, ignore_ids=set())
    if 0 < fraction < 1:
        records = random.Random(seed).sample(records, int(round(len(records) * fraction)))
    return records


def split_ambiguous_sets(synth, folders):
    """Add ``ambig_seen`` / ``ambig_heldout``: the ``ambiguous`` evaluation documents whose word is / is not one of the
    words the synthetic training set uses (training/synth_train/word_split.json). Same split for every run, with or
    without synthetic training data, so runs stay comparable."""
    split_file = SYNTH_TRAIN_DIR / "word_split.json"
    if "ambiguous" not in synth or not split_file.exists():
        return synth
    meta = None
    for d in folders:
        if (Path(d) / "ambiguous.meta.json").exists():
            meta = json.loads((Path(d) / "ambiguous.meta.json").read_text(encoding="utf-8"))
    if meta is None:
        return synth
    seen = set(json.loads(split_file.read_text(encoding="utf-8"))["seen"])
    out = dict(synth)
    out["ambig_seen"] = [r for r in synth["ambiguous"] if meta.get(str(r["id"]), {}).get("word") in seen]
    out["ambig_heldout"] = [r for r in synth["ambiguous"] if meta.get(str(r["id"]), {}).get("word") not in seen]
    return out


def train_ner(model_name, data_dir, output_dir, *, seeds=DEFAULT_SEEDS, synth_eval_dir=DEFAULT_SYNTH_DIRS,
              early_stopping=True, extra_train=None, extra_fraction=1.0, staged=False, stage1_epochs=1,
              limit=None, run_group=None, **hparams):
    """Train on the train split, select the best epoch on validation, score on test (if present) and on the
    synthetic evaluation sets in `synth_eval_dir`. Repeated for every seed in `seeds`; each seed gets its own
    folder ``<output_dir>/seed_<n>/`` and, at the end, ``summary.json`` / ``summary.md`` hold mean and std.

    `extra_train`: jsonl file(s) of synthetic training documents. Default (`staged=False`) they are MIXED into the real
    training split (`extra_fraction` keeps a fixed random share of them). With `staged=True` the model first trains
    `stage1_epochs` epochs on the synthetic documents alone, then continues on the real split as usual (validation,
    early stopping, test). Validation and test are always real data only.
    `limit`: cap the documents of every split/set (smoke tests). `run_group`: nest the committed run copies under
    ``training/runs/<run_group>/`` (campaigns).

    Like MedNER's original, this does NOT apply `ignorable_ids.txt`: a pre-made split
    folder is used exactly as given (the ignore list only applies to `--data` files).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    splits = load_splits(data_dir, ignore_ids=set())
    data_info = {
        "data_dir": str(Path(data_dir).resolve()),
        "ignore_list_applied": False,
        "splits": {name: describe_split(find_split_file(data_dir, name), recs)
                   for name, recs in splits.items()},
    }
    cap = (lambda recs: recs[:limit]) if limit else (lambda recs: recs)
    extra_records = cap(load_extra_train(extra_train, extra_fraction))
    real_train = cap(splits["train"])
    data_info["extra_train"] = {"files": [str(Path(p).resolve()) for p in (extra_train or [])],
                                "fraction": extra_fraction, "documents": len(extra_records),
                                "mode": "none" if not extra_records else ("staged: synthetic first" if staged else "mixed"),
                                "real_train_documents": len(real_train)}
    train_ds = prepare_dataset(real_train if (staged or not extra_records) else real_train + extra_records, tokenizer)
    stage1_ds = prepare_dataset(extra_records, tokenizer) if (staged and extra_records) else None
    val_ds = prepare_dataset(cap(splits["validation"]), tokenizer)
    test_ds = prepare_dataset(cap(splits["test"]), tokenizer) if "test" in splits else None

    synth, synth_paths = {}, {}
    dirs = [synth_eval_dir] if isinstance(synth_eval_dir, (str, Path)) else list(synth_eval_dir or [])
    for d in dirs:  # one or several folders of *.jsonl sets; a set name must be unique
        for name, recs in load_extra_eval_sets(d).items():
            if name in synth:
                raise ValueError(f"synthetic set {name!r} found in more than one folder")
            synth[name], synth_paths[name] = recs, Path(d) / f"{name}.jsonl"
    if synth:
        data_info["synthetic_eval_sets"] = {name: describe_split(synth_paths[name], recs) for name, recs in synth.items()}
    synth = split_ambiguous_sets(synth, dirs)
    extra_eval = {name: prepare_dataset(cap(recs), tokenizer) for name, recs in synth.items()}

    experiment = Path(output_dir).name
    group = f"{run_group}/" if run_group else ""
    done = []
    try:
        for seed in seeds:
            seed_dir = Path(output_dir) / f"seed_{seed}"
            base = model_name
            if stage1_ds is not None:   # stage 1: synthetic documents only, no validation; its model is stage 2's start
                s1_hparams = dict(hparams, epochs=stage1_epochs)
                run_training(model_name, stage1_ds, None, seed_dir / "stage1_synth", kind="train-stage1",
                             data_info=data_info, seed=seed, run_name=f"{group}{experiment}/seed_{seed}_stage1",
                             notes="Stage 1 of a staged run: synthetic documents only.", **s1_hparams)
                base = str(seed_dir / "stage1_synth")
            run_training(base, train_ds, val_ds, seed_dir, kind="train", data_info=data_info,
                         test_ds=test_ds, extra_eval=extra_eval, seed=seed, early_stopping=early_stopping,
                         run_name=f"{group}{experiment}/seed_{seed}", **hparams)
            done.append(seed)
    finally:  # a crash in seed 3 must not lose the table for seeds 1-2
        if done:
            summarize_experiment(Path(output_dir), mirror_name=f"{group}{experiment}")
    return Path(output_dir)


def train_on_all(model_name, output_dir, data_path=None, data_dir=None, extra_train=None, **hparams):
    """PRODUCTION model: train on every record (train + validation + test), no validation.

    Use only after the recipe is settled with `train` (which holds out validation/test), and set
    ``--epochs`` to the best epoch found there, since nothing here selects an epoch.
    Data: either one annotated file (`data_path`, ignore list applied) or all splits of a
    split folder (`data_dir`, merged exactly as stored, like the development run saw them).
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if data_path:
        records = load_records(data_path)
        data_info = {"ignore_list_applied": True, "all_data": describe_split(data_path, records)}
    else:
        splits = load_splits(data_dir, ignore_ids=set())
        records = [r for name in SPLIT_NAMES for r in splits.get(name, [])]
        data_info = {
            "data_dir": str(Path(data_dir).resolve()),
            "ignore_list_applied": False,
            "merged_splits": {name: describe_split(find_split_file(data_dir, name), recs)
                              for name, recs in splits.items()},
            "all_data": describe_records(records),
        }
    extra_records = load_extra_train(extra_train)
    if extra_records:
        data_info["extra_train"] = {"files": [str(Path(p).resolve()) for p in extra_train], "documents": len(extra_records)}
    train_ds = prepare_dataset(records + extra_records, tokenizer)
    return run_training(model_name, train_ds, None, output_dir, kind="train-all",
                        data_info=data_info,
                        notes="Production model: trained on ALL data; no held-out evaluation exists for it.",
                        **hparams)


def run_kfold_cv(model_name, data_path, output_root, k=5, **hparams):
    """k-fold CV with early stopping, to check stability across data slices."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    records = load_records(data_path)
    seed = hparams.get("seed", SEED)
    for fold, (tr, va) in enumerate(KFold(n_splits=k, shuffle=True, random_state=seed).split(records), 1):
        print(f"\n--- FOLD {fold}/{k} ---")
        train_recs = [records[i] for i in tr]
        val_recs = [records[i] for i in va]
        data_info = {"ignore_list_applied": True, "source_file": str(Path(data_path).resolve()),
                     "fold": fold, "k": k, "kfold_seed": seed,
                     "train": describe_records(train_recs), "validation": describe_records(val_recs)}
        train_ds = prepare_dataset(train_recs, tokenizer)
        val_ds = prepare_dataset(val_recs, tokenizer)
        trainer = run_training(model_name, train_ds, val_ds, Path(output_root) / f"CV_Fold_{fold}",
                               early_stopping=True, kind="cv-fold", data_info=data_info, **hparams)
        print(f"Fold {fold} metrics: {trainer.evaluate()}")


def train_final_production_model(model_name, data_path, output_dir, **hparams):
    """95% train / 5% validation with early stopping, so we don't over-memorize."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    seed = hparams.get("seed", SEED)
    train_recs, val_recs = train_test_split(load_records(data_path), test_size=0.05, random_state=seed)
    data_info = {"ignore_list_applied": True, "source_file": str(Path(data_path).resolve()),
                 "holdout_fraction": 0.05, "holdout_seed": seed,
                 "train": describe_records(train_recs), "validation": describe_records(val_recs)}
    train_ds = prepare_dataset(train_recs, tokenizer)
    val_ds = prepare_dataset(val_recs, tokenizer)
    return run_training(model_name, train_ds, val_ds, output_dir, early_stopping=True, kind="final",
                        data_info=data_info, **hparams)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    hp = argparse.ArgumentParser(add_help=False)
    hp.add_argument("--model", default=DEFAULT_MODEL,
                    help="HF model id or local path of the base model (default: %(default)s)")
    hp.add_argument("--lr", type=float, default=5e-5)
    hp.add_argument("--batch-size", type=int, default=8)
    hp.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="maximum epochs (default: %(default)s)")
    hp.add_argument("--patience", type=int, default=DEFAULT_PATIENCE,
                    help="early stopping: stop after this many epochs without a better validation F1; "
                         "0 disables it (default: %(default)s)")
    hp.add_argument("--weight-decay", type=float, default=0.01)

    out_help = "default: training/models/<model name>_<timestamp>"

    s = sub.add_parser("split", help="stratified train/validation/test split of an annotated JSONL")
    s.add_argument("--data", required=True, help="annotated .jsonl")
    s.add_argument("--out", required=True, help="directory for train/validation/test.jsonl")
    s.add_argument("--seed", type=int, default=SEED)

    t = sub.add_parser("train", parents=[hp],
                       help="train with validation (best epoch by F1, early stopping), score on test and on the "
                            "synthetic sets; one run per seed plus a summary table")
    t.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                   help="directory with train/validation(/test) files (default: %(default)s)")
    t.add_argument("--output-dir", default=None, help=out_help + "; holds one seed_<n> folder per seed")
    t.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
                   help="random seeds, one full run each (default: %(default)s)")
    t.add_argument("--synth-eval-dir", nargs="+", default=[str(d) for d in DEFAULT_SYNTH_DIRS],
                   help="folder(s) with synthetic evaluation sets (*.jsonl), scored after training "
                        "(default: %(default)s)")
    t.add_argument("--no-synth-eval", action="store_true", help="skip the synthetic evaluation sets")
    t.add_argument("--extra-train", nargs="+", default=None,
                   help="synthetic TRAINING jsonl file(s) (e.g. training/synth_train/synth_train.jsonl)")
    t.add_argument("--extra-fraction", type=float, default=1.0, help="share of the extra documents to use (default 1)")
    t.add_argument("--staged", action="store_true",
                   help="train on the extra documents first (--stage1-epochs), then on the real split; default is mixing")
    t.add_argument("--stage1-epochs", type=int, default=1)
    t.add_argument("--limit", type=int, default=None, help="cap every split at N documents (smoke tests only)")

    a = sub.add_parser("train-all", parents=[hp],
                       help="PRODUCTION: train on all data (train+validation+test), no validation")
    a.add_argument("--data", default=None, help="one annotated .jsonl (ignore list applied); "
                   "if omitted, all splits of --data-dir are merged")
    a.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="split folder to merge (default: %(default)s)")
    a.add_argument("--output-dir", default=None, help=out_help)
    a.add_argument("--seed", type=int, default=SEED)
    a.add_argument("--extra-train", nargs="+", default=None,
                   help="synthetic training jsonl added to ALL real data (production: synth_train_all.jsonl)")

    c = sub.add_parser("cv", parents=[hp], help="k-fold cross-validation")
    c.add_argument("--data", required=True)
    c.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "models"), help="folder for CV_Fold_<n> subfolders")
    c.add_argument("-k", type=int, default=5)
    c.add_argument("--seed", type=int, default=SEED)

    f = sub.add_parser("final", parents=[hp], help="95/5 split with early stopping")
    f.add_argument("--data", required=True)
    f.add_argument("--output-dir", default=None, help=out_help)
    f.add_argument("--seed", type=int, default=SEED)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.command == "split":
        write_splits(load_records(args.data), args.out, seed=args.seed)
        return

    if args.output_dir is None:
        args.output_dir = default_output_dir(args.model)
    hparams = dict(lr=args.lr, batch_size=args.batch_size, epochs=args.epochs,
                   weight_decay=args.weight_decay, patience=args.patience)
    if args.command == "train":
        train_ner(args.model, args.data_dir, args.output_dir, seeds=args.seeds,
                  synth_eval_dir=None if args.no_synth_eval else args.synth_eval_dir, extra_train=args.extra_train,
                  extra_fraction=args.extra_fraction, staged=args.staged, stage1_epochs=args.stage1_epochs,
                  limit=args.limit, **hparams)
    elif args.command == "train-all":
        train_on_all(args.model, args.output_dir, data_path=args.data, data_dir=args.data_dir,
                     extra_train=args.extra_train, seed=args.seed, **hparams)
    elif args.command == "cv":
        run_kfold_cv(args.model, args.data, args.output_dir, k=args.k, seed=args.seed, **hparams)
    elif args.command == "final":
        train_final_production_model(args.model, args.data, args.output_dir, seed=args.seed, **hparams)


if __name__ == "__main__":
    main()
