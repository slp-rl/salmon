import argparse
import json
import os
import random
import math
from pathlib import Path

from tqdm import tqdm

from utils.general import seed_everything
from utils.audio import Audio
from utils.dataset import VCTKDataset


def speaker_consistency(args):
    vctk = VCTKDataset(metadata_path="/cs/labs/adiyoss/amitroth/slm-benchmark/audio/vctk_metadata.json")  # we use vctk recordings and concatenate them to create the samples
    metadata = list()

    # we only have these texts available in VCTK across all speakers
    texts = vctk.get_texts(min_words=4)

    samples_per_text = math.ceil(args.n_samples / len(texts))

    speakers_per_text = {t: random.sample(vctk.get_speakers(text_id=t), 3 * args.n_options * samples_per_text) for t in texts}

    for i in tqdm(range(args.n_samples)):
        text = texts[int(i / samples_per_text)]
        if args.gender:
            first_speaker = speakers_per_text[text].pop()
            gender = vctk.get_gender(first_speaker)
            negative_speakers = list()
            index = 0
            while len(negative_speakers) < args.n_options - 1:
                if vctk.get_gender(speakers_per_text[text][index]) != gender:
                    negative_speakers.append(speakers_per_text[text].pop(index))
                    index = 0
                else:
                    index += 1
            speakers = [first_speaker] + negative_speakers
            assert len(speakers) == args.n_options
        else:
            speakers = [speakers_per_text[text].pop() for _ in range(args.n_options)]

        generate_sample(dataset=vctk, text=text, speakers=speakers, index=i, args=args)

        metadata.append({
            "index": i,
            "text": text,
            "speakers": speakers
        })

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)


def generate_sample(dataset, text, speakers, index, args):
    """
    Generate a sample for the speaker consistency benchmark
    """
    # text, path, alignment
    start_audio_path = dataset.get_audio(speaker=speakers[0], text_id=text)
    start_audio = Audio(start_audio_path[1])
    end_audio_path_list = [dataset.get_audio(speaker=speakers[i], text_id=text) for i in range(args.n_options)]

    cut_index = int((len(start_audio_path[2]) - 1) * args.split_ratio)
    start_cut = (start_audio_path[2][cut_index][1] + start_audio_path[2][cut_index+1][0]) / 2

    for i in range(args.n_options):
        end_cut = (end_audio_path_list[i][2][cut_index][1] + end_audio_path_list[i][2][cut_index+1][0]) / 2
        audio = start_audio[:int(start_cut * Audio.GENERAL_SAMPLING_RATE)] / Audio(end_audio_path_list[i][1])[int(end_cut * Audio.GENERAL_SAMPLING_RATE):]
        audio.write_audio(args.output_dir / f"sample_{index}_{i}.wav", normalize=args.normalize)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/cs/labs/adiyoss/amitroth/salmon_v2/speaker_consistency/random',
                        help='Path to save outputs')
    parser.add_argument('--n_samples', default=200, type=int, help='Number of samples for benchmark')
    parser.add_argument('--n_options', default=2, type=int, help='Number of options in each sample including the true')
    parser.add_argument('--seed', default=43, type=int, help='random seed, use -1 for non-determinism')
    parser.add_argument('--split-ratio', default=0.5, type=float, help='where to split the recording')
    parser.add_argument('--gender', default=False, type=bool, help='set distractors to be from the opposite gender')
    parser.add_argument('--normalize', default=False, type=bool, help='normalize final audio')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    args.output_dir = Path(args.output_dir)

    os.makedirs(Path(args.output_dir), exist_ok=True)
    seed_everything(args.seed)

    speaker_consistency(args)
