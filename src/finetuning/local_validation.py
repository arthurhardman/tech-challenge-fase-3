"""Treino pequeno e REAL para validar o pipeline em CPU.

O caminho principal da entrega continua sendo LoRA/PEFT em ``train_lora.py``.
Este arquivo existe para que preparação -> treino -> checkpoint -> inferência
possa ser executada de ponta a ponta mesmo em uma máquina sem GPU e sem download
do Hugging Face. O modelo é um Transformer pequeno, então ele serve como teste de
engenharia, não como substituto do LLaMA/Falcon/Mistral da entrega.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHUNKS_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "official" / "protocol_chunks.jsonl"
FAQ_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "official" / "protocol_faq.jsonl"
EVAL_PATH = PROJECT_ROOT / "data" / "knowledge_base" / "official" / "qa_eval.jsonl"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "finetuning" / "local_validation"

SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]
TOKEN_RE = re.compile(r"[\wÀ-ÿ]+|[^\w\s]", re.UNICODE)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


class Tokenizer:
    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inverse = {v: k for k, v in vocab.items()}
        self.pad_id = vocab["<pad>"]
        self.bos_id = vocab["<bos>"]
        self.eos_id = vocab["<eos>"]
        self.unk_id = vocab["<unk>"]

    @staticmethod
    def tokens(text: str) -> list[str]:
        return TOKEN_RE.findall(str(text).lower())

    @classmethod
    def build(cls, texts: Iterable[str], max_vocab: int = 5000):
        counts: dict[str, int] = {}
        for text in texts:
            for token in cls.tokens(text):
                counts[token] = counts.get(token, 0) + 1
        vocab = {tok: i for i, tok in enumerate(SPECIAL)}
        for token, _ in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
            if len(vocab) >= max_vocab:
                break
            if token not in vocab:
                vocab[token] = len(vocab)
        return cls(vocab)

    def encode(self, text: str, max_len: int) -> list[int]:
        ids = [self.bos_id] + [self.vocab.get(t, self.unk_id) for t in self.tokens(text)] + [self.eos_id]
        return ids[:max_len]

    def decode(self, ids: Iterable[int]) -> str:
        out = []
        for idx in ids:
            token = self.inverse.get(int(idx), "<unk>")
            if token == "<eos>":
                break
            if token not in {"<pad>", "<bos>"}:
                out.append(token)
        return re.sub(r"\s+([,.;:!?])", r"\1", " ".join(out)).strip()

    def save(self, path: Path):
        path.write_text(json.dumps(self.vocab, ensure_ascii=False, indent=2), encoding="utf-8")


class PairDataset(Dataset):
    def __init__(self, pairs, tokenizer: Tokenizer, max_src: int, max_tgt: int):
        self.pairs, self.tok, self.max_src, self.max_tgt = pairs, tokenizer, max_src, max_tgt

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src, tgt = self.pairs[idx]
        return self.tok.encode(src, self.max_src), self.tok.encode(tgt, self.max_tgt)


def _collate(batch, pad_id: int):
    srcs, tgts = zip(*batch)
    src = torch.full((len(batch), max(map(len, srcs))), pad_id, dtype=torch.long)
    tgt = torch.full((len(batch), max(map(len, tgts))), pad_id, dtype=torch.long)
    for i, ids in enumerate(srcs):
        src[i, :len(ids)] = torch.tensor(ids)
    for i, ids in enumerate(tgts):
        tgt[i, :len(ids)] = torch.tensor(ids)
    return src, tgt


class TinyTransformer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, d_model: int = 96, layers: int = 2, max_len: int = 128):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.net = nn.Transformer(
            d_model=d_model, nhead=4, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=256,
            dropout=0.1, batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def _embed(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.token(ids) * math.sqrt(self.d_model) + self.pos(pos)

    def forward(self, src, tgt_in):
        causal = torch.triu(torch.ones((tgt_in.size(1), tgt_in.size(1)), dtype=torch.bool, device=tgt_in.device), diagonal=1)
        hidden = self.net(
            self._embed(src), self._embed(tgt_in), tgt_mask=causal,
            src_key_padding_mask=src.eq(self.pad_id),
            tgt_key_padding_mask=tgt_in.eq(self.pad_id),
            memory_key_padding_mask=src.eq(self.pad_id),
        )
        return self.out(hidden)

    @torch.no_grad()
    def generate(self, src, bos_id: int, eos_id: int, max_new_tokens: int = 64):
        self.eval()
        generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        for _ in range(max_new_tokens):
            nxt = self(src, generated)[:, -1].argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            if torch.all(nxt.squeeze(1).eq(eos_id)):
                break
        return generated


@dataclass
class Config:
    seed: int = 42
    max_src: int = 96
    max_tgt: int = 72
    batch_size: int = 8
    pretrain_epochs: int = 2
    finetune_epochs: int = 12
    learning_rate: float = 0.002


def _token_f1(pred: str, ref: str) -> float:
    p, r = Tokenizer.tokens(pred), Tokenizer.tokens(ref)
    if not p or not r:
        return float(p == r)
    pc, rc = {x: p.count(x) for x in set(p)}, {x: r.count(x) for x in set(r)}
    overlap = sum(min(pc.get(x, 0), rc.get(x, 0)) for x in pc)
    if not overlap:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def _train(model, loader, opt, criterion, epochs, device):
    hist = []
    for epoch in range(1, epochs + 1):
        losses = []
        model.train()
        for src, tgt in loader:
            src, tgt = src.to(device), tgt.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(src, tgt[:, :-1])
            loss = criterion(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        hist.append({"epoch": epoch, "loss": float(np.mean(losses))})
    return hist


def _data():
    chunks = load_jsonl(CHUNKS_PATH)
    all_qa = load_jsonl(FAQ_PATH)
    eval_qa = load_jsonl(EVAL_PATH)
    eval_questions = {x.get("instruction") for x in eval_qa}
    train_qa = [x for x in all_qa if x.get("question") not in eval_questions]

    # Um pouco de cada documento entra no pré-treino para o modelo não ficar
    # concentrado nas primeiras páginas de um único PDF.
    by_source: dict[str, list[dict]] = {}
    for row in chunks:
        by_source.setdefault(row.get("source", "sem_fonte"), []).append(row)
    selected = []
    positions = {k: 0 for k in by_source}
    while len(selected) < 100:
        added = False
        for source in sorted(by_source):
            pos = positions[source]
            if pos < len(by_source[source]):
                selected.append(by_source[source][pos])
                positions[source] += 1
                added = True
                if len(selected) >= 100:
                    break
        if not added:
            break

    pre_pairs = []
    for row in selected:
        words = row["text"].split()
        if len(words) < 24:
            continue
        split = min(max(12, len(words) // 2), 55)
        pre_pairs.append((" ".join(words[:split]), " ".join(words[split:split + 45])))
    train_pairs = [(x["question"], x["answer"]) for x in train_qa]
    return pre_pairs, train_pairs, eval_qa


def run(output_dir: Path = DEFAULT_OUTPUT, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pre_pairs, train_pairs, eval_rows = _data()
    texts = [x for pair in pre_pairs + train_pairs for x in pair]
    texts += [x["instruction"] + " " + x["response"] for x in eval_rows]
    tok = Tokenizer.build(texts)

    model = TinyTransformer(len(tok.vocab), tok.pad_id, max_len=max(cfg.max_src, cfg.max_tgt) + 4).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    collate = lambda batch: _collate(batch, tok.pad_id)

    pre_loader = DataLoader(PairDataset(pre_pairs, tok, cfg.max_src, cfg.max_tgt), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate)
    pre_hist = _train(model, pre_loader, opt, criterion, cfg.pretrain_epochs, device)

    train_loader = DataLoader(PairDataset(train_pairs, tok, cfg.max_src, cfg.max_tgt), batch_size=cfg.batch_size, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate / 2)
    ft_hist = _train(model, train_loader, opt, criterion, cfg.finetune_epochs, device)

    details = []
    model.eval()
    for row in eval_rows:
        ids = torch.tensor([tok.encode(row["instruction"], cfg.max_src)], dtype=torch.long, device=device)
        generated = model.generate(ids, tok.bos_id, tok.eos_id, max_new_tokens=cfg.max_tgt - 1)
        pred = tok.decode(generated[0].tolist())
        details.append({
            "question": row["instruction"], "reference": row["response"],
            "prediction": pred, "token_f1": _token_f1(pred, row["response"]),
            "source": row.get("source"), "page": row.get("page"),
        })

    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    tok.save(output_dir / "tokenizer.json")
    torch.save({"model_state": model.state_dict(), "config": asdict(cfg), "vocab_size": len(tok.vocab), "pad_id": tok.pad_id}, output_dir / "tiny_transformer.pt")
    metrics = {
        "mode": "real-local-validation", "device": device,
        "pretraining_pairs": len(pre_pairs), "training_examples": len(train_pairs),
        "evaluation_examples": len(eval_rows),
        "pretraining_final_loss": pre_hist[-1]["loss"],
        "finetuning_final_loss": ft_hist[-1]["loss"],
        "mean_token_f1": float(np.mean([x["token_f1"] for x in details])),
        "pretraining_history": pre_hist, "finetuning_history": ft_hist,
        "examples": details,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    m = run()
    print(json.dumps({k: v for k, v in m.items() if k not in {"examples", "pretraining_history", "finetuning_history"}}, indent=2, ensure_ascii=False))
