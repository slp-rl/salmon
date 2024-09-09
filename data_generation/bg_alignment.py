import argparse
import json
import os
import random

from pathlib import Path

from utils.dataset import BackgroundDataset, TextDataset
from tqdm import tqdm
from utils.microsoft_tts import generate_speech

from utils.audio import *


def bg_alignment(args):

    speakers = [
        # {'speaker': "en-GB-SoniaNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-AriaNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-DavisNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-GuyNeural", 'styles': ["happy", "sad"], 'style_degree': 1.6},
        # {'speaker': "en-US-JaneNeural", 'styles': ["happy", "sad"], 'style_degree': 2.0},
        # {'speaker': "en-US-JasonNeural", 'styles': ["happy", "sad"], 'style_degree': 1.8},
        # {'speaker': "en-US-JennyNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-NancyNeural", 'styles': ["cheerful", "sad"]},
        {'speaker': "en-US-SaraNeural", 'styles': ["happy", "sad"], 'style_degree': 2.0}
        # {'speaker': "en-US-TonyNeural", 'styles': ["cheerful", "sad"]},

    ]
    background_dataset = BackgroundDataset()
    bg_classes = random.choices(background_dataset.get_classes(), k=args.n_samples)

    metadata = list()

    for i, bg_class in tqdm(enumerate(bg_classes)):
        snr_range = random.choice(args.snrs)
        snr = random.uniform(snr_range[0], snr_range[1])
        speaker = random.choice(speakers)
        text, bg_classes, bg_paths = generate_sample(background_dataset=background_dataset, bg_class=bg_class, speaker=speaker,
                                           index=i,
                                           snr=snr, args=args)

        metadata.append({
            "index": i,
            "text_bg_class": bg_class,
            "text": text,
            "snr": snr,
            "audio": [
                {"index": j, "audio_bg_class": bg, "bg_path": str(bg_paths[j])} for j, bg in enumerate(bg_classes)
            ]
        })

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def generate_sample(background_dataset, bg_class, speaker, index, snr, args):
    # generate speech
    text_dataset = TextDataset(f"/cs/labs/adiyoss/amitroth/slm-benchmark/txt/background/{bg_class}.txt")
    text = text_dataset.get_random_text()
    speech_audio = generate_speech(out_path="wav",
                                   text=text,
                                   speaker=speaker['speaker'],
                                   style="neutral")

    # sample background noises
    negative_classes = list(background_dataset.get_classes())
    negative_classes.remove(bg_class)
    bg_classes = [bg_class] + random.sample(negative_classes, args.n_options - 1)

    bg_paths = [background_dataset.get_audio_path(bg) for bg in bg_classes]
    bg_audios = [Audio(bg_path) for bg_path in bg_paths]
    padded_bg_audios = list()
    for bg_audio in bg_audios:
        if len(speech_audio) > len(bg_audio):
            # pad
            print("Warning: bg sound is shorter then speech. padding")
            padded_bg_audios.append(bg_audio.pad(len(speech_audio)))
        else:
            padded_bg_audios.append(bg_audio[:len(speech_audio)])

    out_audios = speech_audio.mix_audios(padded_bg_audios, snr)
    for i, together_out_audio in enumerate(out_audios):
        together_out_audio.write_audio(args.output_dir / f"together/sample_{index}_{i}.wav",
                                       normalize=args.normalize)

    for i, bg_audio in enumerate(bg_audios):
        before_out_audio = bg_audio[:4 * Audio.GENERAL_SAMPLING_RATE] / speech_audio
        after_out_audio = speech_audio / bg_audio[:4 * Audio.GENERAL_SAMPLING_RATE]

        before_out_audio.write_audio(args.output_dir / f"before/sample_{index}_{i}.wav", normalize=args.normalize)
        after_out_audio.write_audio(args.output_dir / f"after/sample_{index}_{i}.wav", normalize=args.normalize)

    return text, bg_classes, bg_paths


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cs/labs/adiyoss/amitroth/salmon_v2/bg_alignment"
    )

    parser.add_argument('--n_samples', default=220, type=int, help='Number of samples for benchmark')
    parser.add_argument('--n_options', default=2, type=int, help='Number of options in each sample including the true')
    parser.add_argument('--snrs', default=[(0.01, 0.02), (0.1, 0.2), (1, 2), (10, 20)], type=list,
                        help='a constant to multiply the background noise with')
    parser.add_argument('--normalize', default=False, type=bool, help='normalize final audio')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.order == "all":
        os.makedirs(args.output_dir / "together", exist_ok=True)
        os.makedirs(args.output_dir / "before", exist_ok=True)
        os.makedirs(args.output_dir / "after", exist_ok=True)

    print(f"creating benchmark in {args.output_dir}")
    bg_alignment(args)
