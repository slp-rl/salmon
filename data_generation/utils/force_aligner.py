# !/usr/bin/env python3

# This script is based on the forced aligner in torchaudio: https://github.com/facebookresearch/fairseq/blob/main/examples/mms
# Author: Eyal Cohen
# Date: 2023-03-10
# Description: This script is used to align a given audio file with a given transcript.

import argparse
import os
import re
import tempfile
from dataclasses import dataclass
from itertools import chain, islice
from typing import Any, Dict, List, Optional, Union

import torch
import torchaudio
from torch import Tensor
from torchaudio.functional import TokenSpan
from torchaudio.pipelines._wav2vec2.aligner import IAligner, ITokenizer, _align_emission_and_tokens
from unikud.framework import Unikud

EN_SPEECH_FILE = torchaudio.utils.download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SANITY_DIR = "sanity_data/forced_aligner"
HEB_SPEECH_FILE = os.path.join(SANITY_DIR, "common_voice_he_38371341.mp3")
HEB_TRANS = "תחושת שוויון וחלוקה צודקת בקבלת החלטות"

EN_TRANS = "i had that curiosity beside me at this moment"
EN_TRANS_EOS_STAR = "i had that curiosity beside *"  # with star at the end
EN_TRANS_MIDDLE_STAR = f"i had that * beside me at this moment"  # with star in the middle
EN_TRANS_BOS_STAR = f"* curiosity beside me at this moment"  # with star at the beginning
EN_TRANS_LST = [EN_TRANS, EN_TRANS_EOS_STAR, EN_TRANS_MIDDLE_STAR, EN_TRANS_BOS_STAR]
EN_TRANS_OUT_DIRS = ["en", "en_eos_star", "en_middle_star", "en_bos_star"]

UROMAN_PATH = "/cs/labs/adiyoss/amitroth/uroman/bin/uroman.pl"


@dataclass
class TextSpan:
    """TextSpan()
    Text with time stamps and score.
    """

    text: str
    """The text"""
    start: Union[int, float]
    """The start time (inclusive)."""
    end: Union[int, float]
    """The end time (exclusive)."""
    score: float
    """The score of the this word."""

    def __len__(self) -> int:
        """Returns the time span"""
        return self.end - self.start

    def __repr__(self):
        if isinstance(self.start, float) or isinstance(self.end, float):
            return f"{self.text} [{self.start:4.3f}, {self.end:4.3f}){self.score:>6.2f}"
        else:
            return f"{self.text} [{self.start:3d}, {self.end:3d}){self.score:>6.2f}"


@dataclass
class WordSpan:
    """WordSpan()
    Word with time stamps and score.
    """

    word: str
    """The word"""
    start: int
    """The start time (inclusive) in emission time axis."""
    end: int
    """The end time (exclusive) in emission time axis."""
    score: float
    """The score of the this word."""

    def __len__(self) -> int:
        """Returns the time span"""
        return self.end - self.start

    def __repr__(self):
        return f"{self.word:<15}[{self.start:3d}, {self.end:3d}){self.score:>6.2f}"


class Tokenizer(ITokenizer):
    def __init__(
            self,
            dictionary: Dict[str, int],
            iso_code: str = "heb",
            diacritizer: bool = True,
            uroman_path: str = UROMAN_PATH,
    ):
        self.c2i = dictionary
        self.i2c = {v: k for k, v in dictionary.items()}
        self.iso_code = iso_code
        self.diacritizer = Unikud(device="cuda") if diacritizer and iso_code == "heb" else None
        self.uroman_path = uroman_path

    def __call__(self, transcript: str) -> List[List[int]]:
        if self.diacritizer is not None:
            transcript = self.diacritizer(transcript)
            transcript = uromanize(transcript, self.uroman_path, self.iso_code)

        tokens = [[self.c2i[c] for c in word if c in self.c2i] for word in transcript.split()]

        return [t for t in tokens if len(t) > 0]  # return only non empty tokens


class ForceAligner(IAligner):
    def __init__(self, tokenizer: Tokenizer, blank: int = 0):
        self.blank = blank
        self.tokenizer = tokenizer

    def __call__(
            self, emission: Tensor, transcript: Union[str, List[str]], return_as_words: bool = False
    ) -> Union[List[List[TokenSpan]], List[List[WordSpan]]]:
        """Aligns the given emission and tokens.

        Args:
            emission (Tensor): Emission tensor. 2D tensor of shape: `(time, n_classes)`.
            transcript (str | List[str]): Transcript, either a string or a list of words.
            return_as_words (bool, optional): If True, returns the aligned tokens as words.

        Raises:
            ValueError: If the input emission is not 2D.

        Returns:
            List[List[TokenSpan]] | [List[WordSpan]]: List of token spans or word spans.

        Example:
            # >>> text = "i had that curiosity beside me at this moment"
            # >>> transcript = text.split()
            # >>> words_span = aligner(emission, tokenizer(transcript), return_as_words=True)
        """
        tokens = self.tokenizer(transcript)
        trans_words = transcript.split() if isinstance(transcript, str) else transcript
        if emission.ndim != 2:
            raise ValueError(f"The input emission must be 2D. Found: {emission.shape}")

        aligned_tokens, scores = _align_emission_and_tokens(emission, _flatten(tokens), self.blank)
        spans = merge_tokens(aligned_tokens, scores)
        spans = _unflatten(spans, [len(ts) for ts in tokens])
        if return_as_words:
            spans = merge_word_tokens(spans, trans_words)
        return spans

    def get_transcript_span(
            self,
            words_span: List[WordSpan],
            time_as: str = "frame",
            frame_size: Optional[int] = None,
            sr: Optional[int] = None,
    ) -> TextSpan:
        """Returns the transcript span in the given time units.

        Args:
            words_span (List[WordSpan]): List of word spans.
            time_as (str, optional): Time units. Can be "frame" or "sec". Defaults to "frame".
            frame_size (Optional[int], optional): Frame size. Defaults to None. Required when time_as is "sec".
            sr (Optional[int], optional): Sample rate. Defaults to None. Required when time_as is "sec".

        Returns:
            TextSpan: The transcript span.

        Example:
            # >>> words_span = aligner(emission[0], transcript, return_as_words=True)
            # >>> transc_spans = aligner.get_transcript_span(words_span)
            # >>> transc_spans_sec = aligner.get_transcript_span(words_span, time_as="sec", frame_size=waveform.shape[1] / emission.shape[1], sr=16000)
        """
        assert time_as in ["frame", "sec"], f"Invalid time units: {time_as}"
        assert time_as == "sec" and frame_size is not None, "Frame size must be given when time_as is 'sec'"

        start = words_span[0].start
        end = words_span[-1].end
        if time_as == "sec":
            start = start * frame_size / sr
            end = end * frame_size / sr
        return TextSpan(
            text=" ".join([w.word for w in words_span]),
            start=start,
            end=end,
            score=sum(w.score * len(w) for w in words_span) / sum(len(w) for w in words_span),
        )


def normalize_uroman(text):
    text = text.lower()
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(" +", " ", text)
    return text.strip()


def uromanize(transcript: Union[str, List[str]], uroman_pl: str, iso: str = "he") -> str:
    """Universal romanizer that uses the uroman.pl script to convert text in different languages to romanized text.

    Args:
        transcript (str | List[str]): The transcript as string or as list of strings.
        uroman_pl (str): Path to the uroman.pl script.
        iso (str, optional): ISO code of the language. Defaults to "he".

    Returns:
        str:
    """
    if isinstance(transcript, str):
        transcript = [transcript]

    tf = tempfile.NamedTemporaryFile()
    tf2 = tempfile.NamedTemporaryFile()
    with open(tf.name, "w") as f:
        for t in transcript:
            f.write(t + "\n")

    assert os.path.exists(uroman_pl), "uroman not found"
    cmd = f"perl {uroman_pl} -l {iso} "
    cmd += f" < {tf.name} > {tf2.name}"
    os.system(cmd)
    outtexts = []
    with open(tf2.name) as f:
        for line in f:
            line = re.sub(r"\s+", " ", line).strip()
            outtexts.append(line)
    return " ".join(outtexts)


def _flatten(list_: List[List[Any]]) -> List[Any]:
    return list(chain.from_iterable(list_))


def _unflatten(list_: List[Any], lengths: List[int]) -> List[List[Any]]:
    assert len(list_) == sum(lengths), "Lengths must sum to list length"
    it = iter(list_)
    return [list(islice(it, l)) for l in lengths]


def merge_tokens(tokens: Tensor, scores: Tensor, blank: int = 0) -> List[TokenSpan]:
    """Removes repeated tokens and blank tokens from the given CTC token sequence.

    Args:
        tokens (Tensor): Alignment tokens (unbatched) returned from :py:func:`forced_align`.
            Shape: `(time, )`.
        scores (Tensor): Alignment scores (unbatched) returned from :py:func:`forced_align`.
            Shape: `(time, )`. When computing the token-size score, the given score is averaged
            across the corresponding time span.

    Returns:
        list of TokenSpan

    Example:
        # >>> aligned_tokens, scores = forced_align(emission, targets, input_lengths, target_lengths)
        # >>> token_spans = merge_tokens(aligned_tokens[0], scores[0])
    """
    if tokens.ndim != 1 or scores.ndim != 1:
        raise ValueError("`tokens` and `scores` must be 1D Tensor.")
    if len(tokens) != len(scores):
        raise ValueError("`tokens` and `scores` must be the same length.")

    # Compute the difference between consecutive tokens.
    diff = torch.diff(
        tokens, prepend=torch.tensor([-1], device=tokens.device), append=torch.tensor([-1], device=tokens.device)
    )
    # Compute the change points and mask out the points where the new value is blank
    changes_wo_blank = torch.nonzero((diff != 0)).squeeze().tolist()

    tokens = tokens.tolist()
    spans = [
        TokenSpan(token=token, start=start, end=end, score=scores[start:end].mean().item())
        for start, end in zip(changes_wo_blank[:-1], changes_wo_blank[1:])
        if (token := tokens[start]) != blank
    ]
    return spans


def merge_word_tokens(tokens_spans: List[List[TokenSpan]], transcript: Union[str, List[str]]) -> List[WordSpan]:
    """Merges the tokens spans into a list of WordSpan.

    Args:
        tokens_spans (List[TokenSpan]): Tokens spans returned by :py:func:`merge_tokens`.
        transcript (str | List[str]): Transcript, either a string or a list of words.

    Returns:
        List[WordSpan]: list of TextSpan
    """
    trans_words = transcript.split() if isinstance(transcript, str) else transcript
    words_span = []
    for word_spans, word in zip(tokens_spans, trans_words):
        start = word_spans[0].start
        end = word_spans[-1].end
        score = sum(s.score * len(s) for s in word_spans) / sum(len(s) for s in word_spans)
        words_span.append(WordSpan(word=word, start=start, end=end, score=score))
    return words_span


def get_device(device: Optional[str] = None):
    if device is not None:
        if device == "cpu":
            return torch.device("cpu")
        elif device.startswith("cuda"):
            return torch.device(device)
        # Now check if device is only a device number:
        elif device.isdigit():
            return torch.device(f"cuda:{device}")
        else:
            raise ValueError(f"Invalid device: {device}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_waveform(file_path: str, sr: int) -> Tensor:
    waveform, file_sr = torchaudio.load(file_path)
    if file_sr != sr:
        waveform = torchaudio.transforms.Resample(file_sr, sr)(waveform)
    return waveform


def prepare_data(with_star: bool, device: torch.device) -> tuple[torch.nn.Module, Dict[str, int], int]:
    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star).to(device)

    c2i = bundle.get_dict()
    return model, c2i, bundle.sample_rate


def get_emission(waveform: Tensor, model: torch.nn.Module, device: torch.device) -> Tensor:
    with torch.inference_mode():
        emission, _ = model(waveform.to(device))
    return emission


def debug_words_as_audio(waveform: Tensor, words_span: List[WordSpan], sr: int, num_frames: int, out_dir: str):
    """Creates audio files for each word and one for the full transcript by slicing the waveform.

    Args:
        waveform (Tensor): Waveform of the audio file.
        words_span (List[TextSpan]): List of word spans.
        sr (int): Sample rate.
        num_frames (int): Number of frames in the emission.
        out_dir (str): Output directory.
    """
    print(f"Saves words as audio files to {out_dir}")
    os.makedirs(out_dir, exist_ok=True)

    ratio = waveform.shape[1] / num_frames
    for i, word_span in enumerate(words_span):
        file_path = os.path.join(out_dir, f"word_{i}.wav")
        text = word_span.word
        start = int(word_span.start * ratio)
        end = int(word_span.end * ratio)
        torchaudio.save(file_path, waveform[:, start:end], sample_rate=sr)
        print(f"{text:<12} {word_span.score:>.2f}, {start / sr:.3f})- {end / sr:.3f} sec, {file_path}")
    file_path = os.path.join(out_dir, "full.wav")
    transcript = " ".join([w.word for w in words_span])
    start = int(words_span[0].start * ratio)
    end = int(words_span[-1].end * ratio)
    torchaudio.save(file_path, waveform[:, start:end], sample_rate=sr)
    print(f"{transcript} {start / sr:.3f} - {end / sr:.3f} sec, {file_path}")


def test(
        model: torch.nn.Module,
        waveform: Tensor,
        aligner: ForceAligner,
        transcript: str,
        sr: int,
        device: torch.device,
        out_dir: str,
):
    """Tests the aligner on the given waveform and transcripts

    Args:
        model (torch.nn.Module): The model.
        waveform (Tensor): The waveform.
        aligner (ForceAligner): The aligner.
        transcript (str): Transcript to test.
        sr (int): Sample rate.
        device (torch.device): Torch device.
        out_dir (str): output directory.
    """
    emission = get_emission(waveform, model, device)
    words_span = aligner(emission[0], transcript, return_as_words=True)
    trans_spans_sec = aligner.get_transcript_span(
        words_span, time_as="sec", frame_size=waveform.shape[1] / emission.shape[1], sr=sr
    )

    print("=" * 60)
    print(f"Transcript: {transcript}")
    debug_words_as_audio(waveform, words_span, sr, emission.shape[1], out_dir)
    print(f"Words spans:")
    for word_span in words_span:
        print(word_span)

    print(f"Transcript span:\n{trans_spans_sec}")
    print("=" * 60)


def arg_parser():
    examples = """Examples:
    python forced_aligner.py --audio sanity_data/forced_aligner/common_voice_he_38371341.mp3 --textfile sanity_data/forced_aligner/common_voice_he_38371341.txt --outdir sanity_data/forced_aligner/output
    python forced_aligner.py --audio sanity_data/forced_aligner/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav --textfile sanity_data/forced_aligner/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.txt --outdir sanity_data/forced_aligner/output

    Run the default test:
    python forced_aligner.py --test

    Run with user input:
    python forced_aligner.py --audio sanity_data/forced_aligner/common_voice_he_38371341.mp3 --outdir sanity_data/forced_aligner/output
    """

    parser = argparse.ArgumentParser(
        description="Forced aligner", epilog=examples, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--audio", type=str, default=HEB_SPEECH_FILE, help="Path to audio file")
    parser.add_argument("--textfile", type=str, help="Path to text file")
    parser.add_argument("--outdir", type=str, default="sanity_data/forced_aligner/output", help="Output directory")
    parser.add_argument("--uroman", type=str, default=UROMAN_PATH, help="Path to uroman.pl script in the uroman repo")
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device. Can be 'cpu', 'cuda' or 'cuda:<num>', or a device number"
    )
    parser.add_argument("--test", action="store_true", help="Run the default test")
    return parser.parse_args()


def main():
    args = arg_parser()
    device = get_device(args.device)

    with_star = True
    model, c2i, sr = prepare_data(with_star, device)
    tokenizer = Tokenizer(c2i, uroman_path=args.uroman)
    aligner = ForceAligner(tokenizer=tokenizer)
    if args.test:
        print("Testing Hebrew file:")
        waveform = load_waveform(HEB_SPEECH_FILE, sr)
        out_dir = os.path.join(SANITY_DIR, "heb")
        test(model, waveform, aligner, HEB_TRANS, sr, device, out_dir)

        print("Testing English file with star at different locations:")
        waveform = load_waveform(EN_SPEECH_FILE, sr)
        for trans, dirname in zip(EN_TRANS_LST, EN_TRANS_OUT_DIRS):
            out_dir = os.path.join(SANITY_DIR, dirname)
            test(model, waveform, aligner, trans, sr, device, out_dir)
    else:
        waveform = load_waveform(args.audio, sr)
        if args.textfile is not None:
            with open(args.textfile) as f:
                transcript = f.read()
        else:
            transcript = input("Enter transcript: ")
        test(model, waveform, aligner, transcript, sr, device, args.outdir)


if __name__ == "__main__":
    main()