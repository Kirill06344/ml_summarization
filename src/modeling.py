"""Загрузка модели, сглаженный loss и генерация."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import model_input


def load_tokenizer(name: str, revision: str | None = None):
    from transformers import AutoTokenizer

    # legacy=False — исправленная токенизация токенов после спецсимволов; fast-вариант mT5 собирается из spiece.model.
    return AutoTokenizer.from_pretrained(name, revision=revision, use_fast=True, legacy=False)


def load_model(name: str, revision: str | None = None):
    from transformers import AutoModelForSeq2SeqLM

    return AutoModelForSeq2SeqLM.from_pretrained(name, revision=revision)


def tiny_model():
    """Случайная крошечная mT5 для тестов и CPU-дымов: настоящий тракт без скачивания гигабайт весов."""
    from transformers import MT5Config, MT5ForConditionalGeneration

    config = MT5Config(vocab_size=250_112, d_model=16, d_kv=8, d_ff=32, num_layers=1, num_decoder_layers=1,
                       num_heads=2, decoder_start_token_id=0, pad_token_id=0, eos_token_id=1)
    return MT5ForConditionalGeneration(config)


def smoothed_loss(logits: torch.Tensor, labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100,
                           label_smoothing=smoothing)


class SummarizerLoss(torch.nn.Module):
    """Модель вместе с loss: DistributedDataParallel оборачивает один forward, поэтому loss считается внутри него."""

    def __init__(self, model, smoothing: float, amp_dtype: torch.dtype | None):
        super().__init__()
        self.model, self.smoothing, self.amp_dtype = model, smoothing, amp_dtype

    def forward(self, input_ids, attention_mask, labels):
        model = self.model
        with torch.autocast(input_ids.device.type, dtype=self.amp_dtype, enabled=self.amp_dtype is not None):
            encoded = model.encoder(input_ids=input_ids, attention_mask=attention_mask)[0]
            decoded = model.decoder(input_ids=model.prepare_decoder_input_ids_from_labels(labels),
                                    encoder_hidden_states=encoded, encoder_attention_mask=attention_mask,
                                    use_cache=False)[0]
            # Голова на 250 тысяч слов применяется только к настоящим токенам: на паддинге логиты батча занимали бы
            # лишний гигабайт памяти T4. Loss при этом тот же, что у полного forward (проверено тестом).
            present = labels != -100
            return smoothed_loss(model.lm_head(decoded[present]), labels[present], self.smoothing)


class _Fp32Linear(torch.nn.Linear):
    def forward(self, hidden_states):
        with torch.autocast(hidden_states.device.type, enabled=False):
            return super().forward(hidden_states.float())


def keep_feed_forward_output_in_fp32(model) -> int:
    """Считать выходной слой `wo` в fp32 внутри autocast, иначе fp16 даёт inf и затем NaN с первого батча.

    Веса `wo` у mT5 настолько велики, что его выход не помещается в fp16: воспроизведено на google/mt5-small, первый
    нечисловой выход — decoder.block.2.layer.2.DenseReluDense.wo. Сама библиотека держит этот слой в fp32
    (`_keep_in_fp32_modules`), но только для `model.half()`, а не для autocast. Вне autocast результат не меняется.

    Подменяется класс, а не метод экземпляра: при копировании модуля привязанный метод считал бы на весах оригинала.
    """
    patched = 0
    for name, module in model.named_modules():
        if name.endswith('.wo'):
            module.__class__ = _Fp32Linear
            patched += 1
    if not patched:
        raise RuntimeError('слой wo не найден: реализация transformers изменилась, fp16 снова даст NaN')
    return patched


@torch.inference_mode()
def generate(model, tokenizer, texts: list[str], *, batch_size: int = 32, max_source: int = 512, max_target: int = 84,
             beams: int = 4, length_penalty: float = 0.6, no_repeat_ngram: int = 2,
             amp_dtype: torch.dtype | None = None) -> list[str]:
    """Резюме в исходном порядке текстов. Длинные идут первыми: нехватка памяти вскроется в начале, а не через час."""
    model.eval()
    device = next(model.parameters()).device
    order = sorted(range(len(texts)), key=lambda i: -len(texts[i]))
    output = [''] * len(texts)
    for start in range(0, len(order), batch_size):
        picked = order[start:start + batch_size]
        batch = tokenizer([model_input(texts[i])[:max_source * 12] for i in picked], max_length=max_source,
                          truncation=True, padding=True, return_tensors='pt').to(device)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            ids = model.generate(**batch, max_length=max_target, num_beams=beams, length_penalty=length_penalty,
                                 no_repeat_ngram_size=no_repeat_ngram, early_stopping=True)
        decoded = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for i, summary in zip(picked, decoded):
            output[i] = summary.strip()
    return output
