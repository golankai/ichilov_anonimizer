"""End-to-end smoke test of ``train_ner`` on CPU with a tiny random BERT (no downloads, a few seconds).

Checks the plumbing, not the quality: one folder per seed, early stopping, per-label metrics, scoring of the synthetic
sets, nested run records, summary files, and that the committed runs/ folder is not touched.
"""
import json
import re
import sys
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("accelerate")

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "inference"))

from tokenizers import Tokenizer, models, pre_tokenizers
from tokenizers.processors import TemplateProcessing
from transformers import BertConfig, BertModel, PreTrainedTokenizerFast

import aggregate_runs
import dataset as ds
import run_record
import train

TRAINING = Path(__file__).parent.parent


@pytest.fixture(autouse=True)
def no_experiment_trackers(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")   # a local wandb install would create a wandb/ folder
    monkeypatch.setenv("WANDB_DISABLED", "true")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    words = set()
    for p in [*(TRAINING / "dummy_data").glob("*.jsonl"), *(TRAINING / "synth_eval").glob("*.jsonl")]:
        for r in ds.load_jsonl(p):
            words |= set(re.findall(r"\w+|[^\w\s]+", r["text"]))
    vocab = {t: i for i, t in enumerate(["[PAD]", "[UNK]", "[CLS]", "[SEP]"] + sorted(words))}
    t = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    t.pre_tokenizer = pre_tokenizers.Whitespace()
    t.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tok = PreTrainedTokenizerFast(tokenizer_object=t, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
                                  sep_token="[SEP]")
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                     intermediate_size=64, max_position_embeddings=512)
    path = tmp_path_factory.mktemp("tiny") / "tiny_bert"
    BertModel(cfg).save_pretrained(path)  # no head, like the real base model
    tok.save_pretrained(path)
    return str(path)


def test_train_ner_two_seeds(tiny_model, tmp_path, monkeypatch):
    runs = tmp_path / "runs_mirror"
    monkeypatch.setattr(run_record, "RUNS_DIR", runs)
    monkeypatch.setattr(aggregate_runs, "RUNS_DIR", runs)
    out = tmp_path / "exp_tiny"
    train.train_ner(tiny_model, str(TRAINING / "dummy_data"), str(out), seeds=[1, 2], epochs=2, patience=1,
                    batch_size=8, synth_eval_dir=str(TRAINING / "synth_eval"))

    for seed in (1, 2):
        rec = json.loads((out / f"seed_{seed}" / "training_run.json").read_text(encoding="utf-8"))
        assert rec["status"] == "completed" and rec["seed"] == seed and rec["schema_version"] == 2
        assert rec["hyperparameters"]["seed"] == seed
        assert rec["early_stopping_patience"] == 1
        res = rec["results"]
        assert res["epochs_planned"] == 2 and res["best_epoch"] in (1.0, 2.0)
        assert res["per_epoch_validation"] and "per_label" in res["per_epoch_validation"][0]
        assert set(res["test"]) == {"overall", "per_label", "other"}
        assert set(res["synthetic"]) == {"rare_hard", "long"}
        assert res["synthetic"]["rare_hard"]["per_label"]["ID"]["support"] > 0
        assert set(rec["data"]["synthetic_eval_sets"]) == {"rare_hard", "long"}
        assert (runs / "exp_tiny" / f"seed_{seed}" / "training_run.json").exists()

    summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert summ["seeds"] == [1, 2] and summ["overall"]["synthetic: long"]["f1"]["stats"]["n"] == 2
    assert (runs / "exp_tiny" / "summary.md").exists()
    assert not (TRAINING / "runs" / "exp_tiny").exists()                    # the committed folder is untouched


def test_no_synthetic_sets_when_disabled(tiny_model, tmp_path, monkeypatch):
    monkeypatch.setattr(run_record, "RUNS_DIR", tmp_path / "m")
    monkeypatch.setattr(aggregate_runs, "RUNS_DIR", tmp_path / "m")
    out = tmp_path / "exp_nosynth"
    train.train_ner(tiny_model, str(TRAINING / "dummy_data"), str(out), seeds=[7], epochs=1, batch_size=8,
                    synth_eval_dir=None)
    rec = json.loads((out / "seed_7" / "training_run.json").read_text(encoding="utf-8"))
    assert rec["results"]["synthetic"] == {} and "synthetic_eval_sets" not in rec["data"]
