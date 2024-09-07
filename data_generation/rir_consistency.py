
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve
from pydub import AudioSegment
import torch
import random
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import json
import torch.nn.functional as F

from utils.audio import Audio
from utils.dataset import RirDataset, LJSpeechDataset
from utils.general import seed_everything


def simulate_rir(signal: Audio, rir: Audio, alpha=0.2):
    conv = torch.tensor(np.convolve(signal.data, rir.data, mode='full'), dtype=torch.float64)
    blended = alpha * conv[:len(signal)] + (1 - alpha) * signal.data
    return Audio.audio_from_wav(torch.tensor(blended, dtype=torch.float64), sr=16000)[:len(signal)]



def rir_consistency(args):
    rir_dataset = RirDataset(path="/cs/labs/adiyoss/amitroth/slm-benchmark/audio/rir_filtered")
    speech_dataset = LJSpeechDataset()
    metadata = list()

    for i in tqdm(range(args.n_samples)):
        speech_path = speech_dataset.get_audio_path()
        rirs = random.sample(rir_dataset.get_audio_paths(), args.n_options)

        generate_sample(speech_path=speech_path, rirs=rirs, index=i, args=args)

        metadata.append({
            "index": i,
            "speech_path": str(speech_path),
            "rirs": [str(r) for r in rirs]
        })

    print(metadata)
    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def generate_sample(speech_path, rirs, index, args):
    speech = Audio(speech_path)

    # # TODO need to clip
    # output_start = simulate_rir(speech[:len(speech)//2], Audio(rirs[0]))
    #
    # for i, rir in enumerate(rirs):
    #     output_end = simulate_rir(speech[len(speech)//2:], Audio(rir))
    #     output = output_start / output_end
    #     output.write_audio(args.output_dir / f"sample_{index}_{i}.wav", normalize=args.normalize)


    positive_audio = simulate_rir(speech, Audio(rirs[0]))
    positive_audio.write_audio(args.output_dir / f"sample_{index}_0.wav", normalize=args.normalize)

    output_start = simulate_rir(speech[:len(speech)//2], Audio(rirs[0]))
    output_end = simulate_rir(speech[len(speech) // 2:], Audio(rirs[1]))
    output = output_start / output_end
    output.write_audio(args.output_dir / f"sample_{index}_1.wav", normalize=args.normalize)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/cs/labs/adiyoss/amitroth/salmon_v2/rir_consistency/',
                        help='Path to save outputs')
    parser.add_argument('--n_samples', default=200, type=int, help='Number of samples for benchmark')
    parser.add_argument('--n_options', default=2, type=int, help='Number of options in each sample including the true')
    parser.add_argument('--seed', default=43, type=int, help='random seed, use -1 for non-determinism')
    parser.add_argument('--normalize', default=False, type=bool, help='normalize final audio')

    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)

    os.makedirs(Path(args.output_dir), exist_ok=True)
    seed_everything(args.seed)

    rir_consistency(args)

    # rir_dataset = RirDataset()
    # speech_dataset = LJSpeechDataset()
    #
    # speech_path = speech_dataset.get_audio_path()
    # rirs = random.sample(rir_dataset.get_audio_paths(), 2)
    #
    # speech = Audio(speech_path)
    #
    # # TODO need to clip
    # output_start = simulate_rir(speech[:len(speech) // 2], Audio(rirs[0]))
    #
    # for i, rir in enumerate(rirs):
    #     print(f"{len(Audio(rir))} - {len(speech)}")
    #     output_end = simulate_rir(speech[len(speech) // 2:], Audio(rir))
    #     print(f"{len(output_start)} - {len(output_end)}")
    #     speech[:len(speech) // 2].write_audio(args.output_dir / f"sample_{i}_start_speech.wav", normalize=args.normalize)
    #     output_start.write_audio(args.output_dir / f"sample_{i}_start_conv.wav", normalize=args.normalize)
    #     output = output_start / output_end
    #     output.write_audio(args.output_dir / f"sample_{i}.wav", normalize=args.normalize)
