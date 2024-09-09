import argparse
import json
import os
import random
from pathlib import Path
from tqdm import tqdm

from utils.general import seed_everything
from utils.audio import Audio
from utils.dataset import BackgroundDataset, LJSpeechDataset


def bg_consistency(args):
    background_dataset = BackgroundDataset()
    speech_dataset = LJSpeechDataset()
    metadata = list()
    for i in tqdm(range(args.n_samples)):
        snr_range = random.choice(args.snrs)
        snr = random.uniform(snr_range[0], snr_range[1])
        speech_path = speech_dataset.get_audio_path()
        bg_wavs_paths = generate_sample(speech_path=speech_path, dataset=background_dataset, index=i, snr=snr, args=args)

        metadata.append({
            "index": i + args.offset,
            "speech_path": str(speech_path),
            "snr": snr,
            "audio": [
                {"index": j, "bg_wav_path": str(bg)} for j, bg in enumerate(bg_wavs_paths)
            ]
        })

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def generate_sample(speech_path, dataset, index, snr, args):
    speech_audio = Audio(speech_path)
    bg_paths = dataset.sample(args.n_options, args.distract_method, speech_length=len(speech_audio))
    split_frame = int(len(speech_audio) * args.split_ratio)

    # positive sample
    positive_bg_audio = Audio(bg_paths[0])
    if len(speech_audio) > len(positive_bg_audio):
        # pad
        print(f"Warning: bg sound is shorter then speech. padding speech :{len(speech_audio) / 16000} bg: {len(positive_bg_audio) / 16000}")
        positive_bg_audio = positive_bg_audio.pad(len(speech_audio))
    else:
        positive_bg_audio = positive_bg_audio[:len(speech_audio)]

    bg_audios = [Audio(bg_path) for bg_path in bg_paths[1:]]

    negative_mixes = [(speech_audio[split_frame:], bg_audio) for bg_audio in bg_audios]
    mixed_audios = Audio.mix_audio_pairs([
        (speech_audio, positive_bg_audio)] + negative_mixes, snr=snr)

    mixed_audios[0].write_audio(args.output_dir / f"sample_{index+ args.offset}_0.wav", normalize=args.normalize)
    audio_start = mixed_audios[0][:split_frame]
    for i, audio_end in enumerate(mixed_audios[1:]):
        negative_audio = audio_start / audio_end
        negative_audio.write_audio(args.output_dir / f"sample_{index + args.offset}_{i + 1}.wav", normalize=args.normalize)
    return bg_paths


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/cs/labs/adiyoss/amitroth/salmon_v2/bg_consistency/in_domain_additional_samples/',
                        help='Path to save outputs')
    parser.add_argument('--n_samples', default=100, type=int, help='Number of samples for benchmark')
    parser.add_argument('--n_options', default=2, type=int, help='Number of options in each sample including the true')
    parser.add_argument('--distract-method', default='in_domain', choices=['in_domain', 'random'],
                        help='Which method to use for generating distractors from [in_domain, random]')
    parser.add_argument('--split-ratio', default=0.5, type=float,
                        help='Number of options in each sample including the true')
    parser.add_argument('--snrs', default=[(0.01, 0.02), (0.1, 0.2), (1, 2), (5, 10)], type=list, help='a constant to multiply the background noise with')
    parser.add_argument('--seed', default=47, type=int, help='random seed, use -1 for non-determinism')
    parser.add_argument('--normalize', default=False, type=bool, help='normalize final audio')

    parser.add_argument(
        "--offset",
        type=int,
        help="style degree",
        default=200,
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    os.makedirs(Path(args.output_dir), exist_ok=True)
    seed_everything(args.seed)

    bg_consistency(args)
