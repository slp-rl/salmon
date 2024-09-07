from typing import Dict, List, Union, Mapping
from torch import FloatTensor, LongTensor
import torch
from baselines.utils import get_gslm_speech_encoder, get_pgslm_speech_encoder
import json

UNITS = "units"
DURATIONS = "durations"
F0 = "f0"

PGSLM_TOKEN_KEYS = (UNITS, DURATIONS, F0)


class SpeechTokenizer(torch.nn.Module):

    def __init__(self, config: Mapping) -> None:
        """
        init method for SpeechTokenizer
        config needs to contain a dense_model_name, a quantizer_model_name, an encoder_vocab_size, and a deduplicate flag
        """
        super().__init__()
        self.encoder = get_gslm_speech_encoder(config['dense_model_name'], config['quantizer_model_name'],
                                               config['encoder_vocab_size'], config['deduplicate'],
                                               need_f0=config['need_f0'])

    @classmethod
    def from_pretrained(cls, path_to_config: str) -> 'SpeechTokenizer':
        """
        a class method to create a SpeechTokenizer from a pretrained config path
        """
        with open(path_to_config, 'r') as f:
            config = json.load(f)
        return cls(config)

    def forward(self, x: Union[List[FloatTensor], FloatTensor], offset: int) -> List[LongTensor]:
        """
        tokenizes a list of audio tensors
        x: a list of audio tensors (or a single audio tensor)
        offset: the offset to add to the tokens
        """
        if isinstance(x, FloatTensor):
            x = [x]
        offset = torch.tensor(offset, device=x[0].device)
        return [self.encoder(x_i)[UNITS].long() + offset for x_i in x]

    def to(self, device):
        """
        moves tokenizer to device
        """
        self.encoder.to(device)
        return self


class PGSLMSpeechTokenizer(torch.nn.Module):
    def __init__(self, config: Mapping) -> None:
        """
        init method for PGSLMSpeechTokenizer
        config needs to contain a dense_model_name, a quantizer_model_name, an encoder_vocab_size, and a deduplicate flag
        """
        super().__init__()
        self.encoder = get_pgslm_speech_encoder(config)

    @classmethod
    def from_pretrained(cls, path_to_config: str) -> 'PGSLMSpeechTokenizer':
        """
        a class method to create a PGSLMSpeechTokenizer from a pretrained config path
        """
        with open(path_to_config, 'r') as f:
            config = json.load(f)
        return cls(config)

    def forward(self, x: Union[List[FloatTensor], FloatTensor], _: int = -1) -> List[Dict[str, torch.Tensor]]:
        """
        tokenizes a list of audio tensors
        x: a list of audio tensors (or a single audio tensor)
        """
        if isinstance(x, FloatTensor):
            x = [x]
        tokens = [
            {
                key: item[key].long()
                for key in PGSLM_TOKEN_KEYS
            }
            for item in self.encoder.batch_forward(x)
        ]

        # clipping duration
        tokens[0]['durations'] = tokens[0]['durations'].clamp(max=31)

        return tokens

    def to(self, device):
        """
        moves tokenizer to device
        """
        self.encoder.to(device)
        return self


