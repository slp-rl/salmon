from abc import ABC, abstractmethod
from fairseq.data.codedataset import Paddings, Shifts
import torch
from typing import Dict, List, Mapping

from .slm_tokenizers import PGSLMSpeechTokenizer, UNITS, DURATIONS, F0, SpeechTokenizer
from .utils import build_pgslm_speech_lm, build_speech_lm, nll, probability, remove_spaces_and_punctuation
from torch.nn.utils.rnn import pad_sequence

from transformers import AutoModelForCausalLM, AutoTokenizer


import whisper

SRC_TOKENS = "src_tokens"
DUR_SRC = "dur_src"
F0_SRC = "f0_src"
MASK = "mask"
F0_MASK = "f0_mask"
DUR_MASK = "dur_mask"
TARGET = "target"
DUR_TARGET = "dur_target"
F0_TARGET = "f0_target"
SRC_LENGTHS = "src_lengths"

PGSLM_INPUT_KEYS = (SRC_TOKENS, DUR_SRC, F0_SRC, MASK, F0_MASK, DUR_MASK, TARGET, DUR_TARGET, F0_TARGET)


class InferenceModel(ABC):
    @abstractmethod
    def log_likelihood(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        ...

    @abstractmethod
    def to(self, device):
        ...


class InferenceModelFactory:
    @staticmethod
    def get_model(config: Mapping, base_path="./") -> InferenceModel:
        if config["model_type"] == "pgslm":
            return PGSLMInferenceModel(config)
        if config["model_type"] == "slm":
            return SLMInferenceModel(config, base_path=base_path)
        if config["model_type"] == "naive":
            return NaiveInferenceModel()

        raise ValueError(f"Model type {config['model_type']} not supported")


class SLMInferenceModel(InferenceModel):
    def __init__(self, config, base_path="./"):
        tokenizer_config = config['tokenizer']
        self.tokenizer = SpeechTokenizer(tokenizer_config)
        self.speech_lm = build_speech_lm(config["model_name"], base_path)
        self.mean_nll = config.get("mean_nll", False)
        self.offset = self.speech_lm.config.offset
        self.padding_value = self.speech_lm.config.pad_token_id
        self.model_name = config["model_name"]

    def log_likelihood(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        self.tokenizer.eval()
        sentece_tokens = self.tokenizer(wavs, self.offset)
        x = pad_sequence(sentece_tokens, batch_first=True, padding_value=self.padding_value)
        self.speech_lm.eval()
        logits = self.speech_lm(input_ids=x).logits
        shifted_x = x[..., 1:]
        shifted_logits = logits[..., :-1, :]
        # Create a mask that is True where the tokens are not padding tokens
        mask = (shifted_x != self.padding_value)
        # Convert the losses to likelihoods
        return -nll(shifted_logits, shifted_x, mask, self.mean_nll)

    def probability(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        self.tokenizer.eval()
        sentece_tokens = self.tokenizer(wavs, self.offset)
        x = pad_sequence(sentece_tokens, batch_first=True, padding_value=self.padding_value)
        self.speech_lm.eval()
        logits = self.speech_lm(input_ids=x).logits
        shifted_x = x[..., 1:]
        shifted_logits = logits[..., :-1, :]
        # Create a mask that is True where the tokens are not padding tokens
        mask = (shifted_x != self.padding_value)
        # Convert the losses to likelihoods
        return probability(shifted_logits, shifted_x, mask)

    def to(self, device):
        self.tokenizer.to(device)
        self.speech_lm.to(device)
        return self

    def __str__(self):
        return f"{self.model_name}"


class PGSLMInferenceModel(InferenceModel):
    def __init__(self, config, base_path="/cs/labs/oabend/avishai.elma/models/pgslm_models"):
        tokenizer_config = config["tokenizer"]
        self.tokenizer = PGSLMSpeechTokenizer(tokenizer_config)
        self.mean_nll = config.get("mean_nll", False)
        self.speech_lm, self.task = build_pgslm_speech_lm(model_type=config["model_name"], data_config=config["data_config"],
                                                          base_path=base_path)
        self.paddings = Paddings(
            self.task.source_dictionary.pad(),
            0,
            self.task.source_f0_dictionary.pad() if self.task.cfg.discrete_f0 else -5.0,
        )
        self.shifts = Shifts(
            self.task.cfg.stream_shifts, self.paddings
        )
        self.device = "cpu"
        self.use_f0 = config.get("use_f0", True)
        self.use_duration = config.get("use_duration", True)
        self.use_units = config.get("use_units", True)
        self.max_token_duration = self.task.cfg.max_token_duration
        self.model_name = config["model_name"]

    def _shift_data(self, tokens: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
        """
        shift the tokens
        """
        feats = {key: [] for key in PGSLM_INPUT_KEYS}
        for token in tokens:
            code, dur, f0 = token[UNITS].long().to(self.device), token[DURATIONS].to(self.device), token[F0].to(
                self.device)
            if self.task.cfg.discrete_f0:
                f0 = f0.long()
            if self.task.cfg.discrete_duration:
                dur = dur.long()
            dur = dur.clamp(0, self.max_token_duration)
            code, code_mask, dur, dur_mask, f0, f0_mask = self.shifts(code, dur, f0)

            # src
            feats[SRC_TOKENS].append(code[:-1])
            feats[DUR_SRC].append(dur[:-1])
            feats[F0_SRC].append(f0[:-1])

            # target
            feats[TARGET].append(code[1:])
            feats[DUR_TARGET].append(dur[1:])
            feats[F0_TARGET].append(f0[1:])

            # masks
            feats[MASK].append(code_mask[1:].logical_or(code_mask[:-1]))
            feats[DUR_MASK].append(dur_mask[1:].logical_or(dur_mask[:-1]))
            feats[F0_MASK].append(f0_mask[1:].logical_or(f0_mask[:-1]))

        return feats

    def log_likelihood(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        wavs = [xi.squeeze(0) for xi in wavs]
        tokens = self.tokenizer(wavs, -1)
        feats = self._shift_data(tokens)

        model_inputs = {}
        model_inputs[SRC_LENGTHS] = torch.LongTensor([u.numel() for u in feats[SRC_TOKENS]])
        model_inputs[SRC_TOKENS] = pad_sequence(feats[SRC_TOKENS], batch_first=True, padding_value=self.paddings.code)
        model_inputs[DUR_SRC] = pad_sequence(feats[DUR_SRC], batch_first=True, padding_value=self.paddings.dur)
        model_inputs[F0_SRC] = pad_sequence(feats[F0_SRC], batch_first=True, padding_value=self.paddings.f0)

        net_output = self.speech_lm(**model_inputs)

        pgslm_nll = torch.zeros(len(wavs), device=self.device)
        if self.use_units:
            code_nll_input = {
                "logits": net_output["token"],
                "target": pad_sequence(feats[TARGET], batch_first=True, padding_value=self.paddings.code),
                "mask": ~pad_sequence(feats[MASK], batch_first=True, padding_value=True)
            }

            pgslm_nll += nll(**code_nll_input, mean_nll=self.mean_nll)

        if self.use_duration:
            dur_nll_input = {
                "logits": net_output["duration"],
                "target": pad_sequence(feats[DUR_TARGET], batch_first=True, padding_value=self.paddings.dur),
                "mask": ~pad_sequence(feats[DUR_MASK], batch_first=True, padding_value=True)
            }

            pgslm_nll += nll(**dur_nll_input, mean_nll=self.mean_nll)

        if self.use_f0:
            f0_nll_input = {
                "logits": net_output["f0"],
                "target": pad_sequence(feats[F0_TARGET], batch_first=True, padding_value=self.paddings.f0),
                "mask": ~pad_sequence(feats[F0_MASK], batch_first=True, padding_value=True)
            }

            pgslm_nll += nll(**f0_nll_input, mean_nll=self.mean_nll)

        return -pgslm_nll

    def to(self, device):
        self.speech_lm.to(device)
        self.tokenizer.to(device)
        self.device = device
        return self

    def __str__(self):
        return f"{self.model_name}"


class NaiveInferenceModel(InferenceModel):
    def __init__(self):
        self.model_name = "ASR(large)+LM"
        self.whisper = whisper.load_model("large", download_root="/cs/labs/adiyoss/amitroth/ckpts/whisper")
        self.llama = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", device_map="auto")
        self.llama.eval()
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.llama.to(device=self.device)

    def log_likelihood(self, wavs: List[torch.Tensor]) -> torch.Tensor:
        # to support batches
        losses = []
        with torch.no_grad():
            for wav_path in wavs:
                text = self.whisper.transcribe(wav_path)
                text = remove_spaces_and_punctuation(text['text'])
                text_tokens = self.tokenizer(text, return_tensors="pt").to(device=self.device)

                logits = self.llama(**text_tokens).logits
                shifted_logits = logits[..., :-1, :]
                x = pad_sequence(text_tokens['input_ids'], batch_first=True, padding_value=-1)
                shifted_x = x[..., 1:]
                mask = (shifted_x != -1)

                l = -nll(shifted_logits, shifted_x, mask, True)
                losses.append(l)

            return torch.tensor(losses, device=self.device)

    def to(self, device):
        return self

    def __str__(self):
        return f"{self.model_name}"
