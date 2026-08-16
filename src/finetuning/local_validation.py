"""Validação local REAL do pipeline de fine-tuning da Fase 3.

O treinamento oficial da entrega continua sendo o caminho LoRA/PEFT em
``train_lora.py``. Este módulo é a alternativa reproduzível em CPU: ele treina
um Transformer pequeno de verdade, salva checkpoint e mede a qualidade em um
conjunto separado de perguntas de SRAG.

A versão inicial usava só 95 pares de pré-treino e 80 Q&As de domínio. Isso era
suficiente como smoke test, mas pouco para um modelo gerativo treinado do zero.
Agora aproveitamos mais do material que já existe no projeto:

1. 300 pares balanceados extraídos dos protocolos oficiais;
2. 300 Q&As públicos de MedQuAD/PubMedQA para ensinar o formato pergunta->resposta;
3. 80 Q&As curados de SRAG para adaptação final do domínio;
4. 20 perguntas de SRAG continuam totalmente fora do treino para avaliação.

Nos exemplos de SRAG o modelo também recebe um pequeno trecho do protocolo da
mesma fonte/página. Isso aproxima a validação local do fluxo real do assistente,
que usa RAG antes de chamar a LLM.
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

from .dataset_prep import load_medquad, load_pubmedqa

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
    def build(cls, texts: Iterable[str], max_vocab: int = 6500):
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
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        d_model: int = 112,
        layers: int = 2,
        max_len: int = 172,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.token = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.net = nn.Transformer(
            d_model=d_model,
            nhead=4,
            num_encoder_layers=layers,
            num_decoder_layers=layers,
            dim_feedforward=256,
            dropout=0.1,
            batch_first=True,
        )
        self.out = nn.Linear(d_model, vocab_size)

    def _embed(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        return self.token(ids) * math.sqrt(self.d_model) + self.pos(pos)

    def forward(self, src, tgt_in):
        causal = torch.triu(
            torch.ones((tgt_in.size(1), tgt_in.size(1)), dtype=torch.bool, device=tgt_in.device),
            diagonal=1,
        )
        hidden = self.net(
            self._embed(src),
            self._embed(tgt_in),
            tgt_mask=causal,
            src_key_padding_mask=src.eq(self.pad_id),
            tgt_key_padding_mask=tgt_in.eq(self.pad_id),
            memory_key_padding_mask=src.eq(self.pad_id),
        )
        return self.out(hidden)

    @torch.no_grad()
    def generate(self, src, bos_id: int, eos_id: int, max_new_tokens: int = 80):
        """Greedy decoding com controles simples contra repetição.

        O modelo local é pequeno e, sem esse cuidado, pode entrar em loops como
        "o guia orienta... o guia orienta...". Não é beam search sofisticado;
        apenas evitamos repetições óbvias para manter a saída legível.
        """
        self.eval()
        generated = torch.full((src.size(0), 1), bos_id, dtype=torch.long, device=src.device)
        for step in range(max_new_tokens):
            logits = self(src, generated)[:, -1]

            # Evita encerrar antes de formar uma resposta mínima.
            if step < 4:
                logits[:, eos_id] = -1e9

            for batch_idx in range(generated.size(0)):
                history = generated[batch_idx].tolist()
                if len(history) > 1:
                    logits[batch_idx, history[-1]] = -1e9
                # Tokens repetidos muitas vezes recebem uma pequena penalização.
                for token_id in set(history[-20:]):
                    if history.count(token_id) >= 3 and token_id != eos_id:
                        logits[batch_idx, token_id] -= 2.0

            nxt = logits.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, nxt], dim=1)
            if torch.all(nxt.squeeze(1).eq(eos_id)):
                break
        return generated


@dataclass
class Config:
    seed: int = 42
    max_src: int = 128
    max_tgt: int = 80
    max_vocab: int = 6500
    d_model: int = 112
    layers: int = 2
    batch_size: int = 16
    domain_batch_size: int = 8
    max_pretraining_pairs: int = 300
    max_general_medical_examples: int = 300
    pretrain_epochs: int = 2
    general_medical_epochs: int = 1
    finetune_epochs: int = 22
    pretrain_learning_rate: float = 0.0015
    general_learning_rate: float = 0.001
    finetune_learning_rate: float = 0.0008
    context_words: int = 105


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


def _page_contexts(chunks: list[dict]) -> dict[tuple[str, int], list[str]]:
    grouped: dict[tuple[str, int], list[str]] = {}
    for row in chunks:
        key = (row.get("source", ""), row.get("page"))
        grouped.setdefault(key, []).append(row.get("text", ""))
    return grouped


def _rag_input(question: str, source: str, page: int, contexts: dict, max_words: int) -> str:
    """Seleciona o chunk da página mais relacionado à pergunta."""
    candidates = contexts.get((source, page), [])
    if not candidates:
        return f"pergunta: {question}"
    q_tokens = set(Tokenizer.tokens(question)) - {"o", "a", "de", "do", "da", "e", "em", "que", "um", "uma"}
    def score(text: str) -> tuple[int, int]:
        tokens = Tokenizer.tokens(text)
        overlap = sum(1 for token in q_tokens if token in tokens)
        return overlap, -len(tokens)
    best = max(candidates, key=score)
    context = " ".join(best.split()[:max_words])
    return f"pergunta: {question} contexto: {context}"


def _balanced_pretraining_pairs(chunks: list[dict], limit: int) -> list[tuple[str, str]]:
    # Faz round-robin entre os PDFs para que um documento grande não domine o treino.
    by_source: dict[str, list[dict]] = {}
    for row in chunks:
        by_source.setdefault(row.get("source", "sem_fonte"), []).append(row)

    pairs: list[tuple[str, str]] = []
    positions = {source: 0 for source in by_source}
    while len(pairs) < limit:
        added = False
        for source in sorted(by_source):
            pos = positions[source]
            if pos >= len(by_source[source]):
                continue
            row = by_source[source][pos]
            positions[source] += 1
            added = True
            words = row.get("text", "").split()
            if len(words) >= 32:
                split = min(max(18, len(words) // 2), 50)
                pairs.append((" ".join(words[:split]), " ".join(words[split:split + 45])))
            if len(pairs) >= limit:
                break
        if not added:
            break
    return pairs


def _sample_general_medical_pairs(limit: int, seed: int) -> list[tuple[str, str]]:
    """Amostra Q&As públicos de MedQuAD e PubMedQA.

    A amostra é determinística. O ajuste SRAG vem logo depois, então estes
    exemplos ensinam formato e vocabulário médico sem substituir o domínio do projeto.
    """
    if limit <= 0:
        return []

    examples = load_medquad() + load_pubmedqa()
    rng = random.Random(seed)
    rng.shuffle(examples)
    selected = examples[:limit]

    pairs = []
    for ex in selected:
        question = ex.instruction
        if ex.input:
            question += f" contexto: {ex.input}"
        pairs.append((question, ex.output))
    return pairs


def _data(cfg: Config):
    chunks = load_jsonl(CHUNKS_PATH)
    all_qa = load_jsonl(FAQ_PATH)
    eval_qa = load_jsonl(EVAL_PATH)
    eval_questions = {x.get("instruction") for x in eval_qa}
    train_qa = [x for x in all_qa if x.get("question") not in eval_questions]
    contexts = _page_contexts(chunks)

    pre_pairs = _balanced_pretraining_pairs(chunks, cfg.max_pretraining_pairs)
    medical_pairs = _sample_general_medical_pairs(cfg.max_general_medical_examples, cfg.seed)
    train_pairs = [
        (
            _rag_input(x["question"], x.get("source", ""), x.get("page"), contexts, cfg.context_words),
            x["answer"],
        )
        for x in train_qa
    ]
    # No teste não usamos a fonte/página gold para montar o contexto. O contexto
    # vem do mesmo retriever usado pelo assistente, evitando vazar a resposta esperada.
    from src.assistant.knowledge.retriever import ProtocolRetriever
    retriever = ProtocolRetriever(
        incluir_medquad=False, incluir_pubmedqa=False, incluir_oficiais=True
    )
    eval_pairs = []
    for x in eval_qa:
        hits = retriever.buscar(x["instruction"], top_k=3)
        # Mantém pedaços dos três melhores resultados, como no RAG real, sem usar a fonte gold.
        per_hit = max(25, cfg.context_words // max(1, len(hits)))
        context = " ".join(" ".join(hit.texto.split()[:per_hit]) for hit in hits)
        src = f"pergunta: {x['instruction']}"
        if context:
            src += f" contexto: {context}"
        eval_pairs.append((src, x["response"]))
    return pre_pairs, medical_pairs, train_pairs, eval_pairs, eval_qa


def run(output_dir: Path = DEFAULT_OUTPUT, cfg: Config | None = None) -> dict:
    cfg = cfg or Config()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    pre_pairs, medical_pairs, train_pairs, eval_pairs, eval_rows = _data(cfg)
    texts = [x for pair in pre_pairs + medical_pairs + train_pairs + eval_pairs for x in pair]
    tok = Tokenizer.build(texts, max_vocab=cfg.max_vocab)

    model = TinyTransformer(
        len(tok.vocab),
        tok.pad_id,
        d_model=cfg.d_model,
        layers=cfg.layers,
        max_len=max(cfg.max_src, cfg.max_tgt) + 12,
    ).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=tok.pad_id)
    collate = lambda batch: _collate(batch, tok.pad_id)

    pre_loader = DataLoader(
        PairDataset(pre_pairs, tok, cfg.max_src, cfg.max_tgt),
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pretrain_learning_rate, weight_decay=0.01)
    pre_hist = _train(model, pre_loader, opt, criterion, cfg.pretrain_epochs, device)

    medical_hist = []
    if medical_pairs and cfg.general_medical_epochs > 0:
        medical_loader = DataLoader(
            PairDataset(medical_pairs, tok, cfg.max_src, cfg.max_tgt),
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate,
        )
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.general_learning_rate, weight_decay=0.01)
        medical_hist = _train(
            model, medical_loader, opt, criterion, cfg.general_medical_epochs, device
        )

    train_loader = DataLoader(
        PairDataset(train_pairs, tok, cfg.max_src, cfg.max_tgt),
        batch_size=cfg.domain_batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.finetune_learning_rate, weight_decay=0.01)
    ft_hist = _train(model, train_loader, opt, criterion, cfg.finetune_epochs, device)

    details = []
    model.eval()
    for row, (source_text, reference) in zip(eval_rows, eval_pairs):
        ids = torch.tensor([tok.encode(source_text, cfg.max_src)], dtype=torch.long, device=device)
        generated = model.generate(ids, tok.bos_id, tok.eos_id, max_new_tokens=cfg.max_tgt - 1)
        pred = tok.decode(generated[0].tolist())
        details.append({
            "question": row["instruction"],
            "reference": reference,
            "prediction": pred,
            "token_f1": _token_f1(pred, reference),
            "source": row.get("source"),
            "page": row.get("page"),
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok.save(output_dir / "tokenizer.json")
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": asdict(cfg),
            "vocab_size": len(tok.vocab),
            "pad_id": tok.pad_id,
        },
        output_dir / "tiny_transformer.pt",
    )

    metrics = {
        "mode": "real-local-validation",
        "validation_version": 2,
        "device": device,
        "pretraining_pairs": len(pre_pairs),
        "general_medical_examples": len(medical_pairs),
        "domain_finetuning_examples": len(train_pairs),
        "training_examples": len(medical_pairs) + len(train_pairs),
        "evaluation_examples": len(eval_rows),
        "vocab_size": len(tok.vocab),
        "pretraining_final_loss": pre_hist[-1]["loss"],
        "general_medical_final_loss": medical_hist[-1]["loss"] if medical_hist else None,
        "finetuning_final_loss": ft_hist[-1]["loss"],
        "mean_token_f1": float(np.mean([x["token_f1"] for x in details])),
        "median_token_f1": float(np.median([x["token_f1"] for x in details])),
        "pretraining_history": pre_hist,
        "general_medical_history": medical_hist,
        "finetuning_history": ft_hist,
        "examples": details,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


if __name__ == "__main__":
    metrics = run()
    hidden = {"examples", "pretraining_history", "general_medical_history", "finetuning_history"}
    print(json.dumps({k: v for k, v in metrics.items() if k not in hidden}, indent=2, ensure_ascii=False))
