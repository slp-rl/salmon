import argparse
import json
import os
import random
from pathlib import Path

from tqdm import tqdm

from utils.general import seed_everything
from utils.audio import Audio
from utils.dataset import ExpressoDataset


def emotion_consistency(args):
    expresso = ExpressoDataset(metadata_path="/cs/labs/adiyoss/amitroth/slm-benchmark/audio/metadata.json")  # TODO replace to a different dataset
    metadata = list()

    text_indexes = {speaker: expresso.get_indexes(speaker=speaker, style="default", minimum_word_count=12) for speaker in expresso.get_speakers()}

    for i in tqdm(range(args.n_samples)):
        speaker = random.choice(expresso.get_speakers())
        emotions = random.sample(expresso.get_styles(), args.n_options)
        text_index = text_indexes[speaker].pop()
        text = generate_sample(dataset=expresso, speaker=speaker, emotions=emotions, text_id=text_index, index=i, args=args)

        metadata.append({
            "index": i,
            "speech_path": speaker,
            "text": text,
            "audio": [
                {"index": j, "emotion": emotion} for j, emotion in enumerate(emotions)
            ]
        })

    with open(args.output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f)



def generate_sample(dataset, speaker, emotions, text_id, index, args):
    """
    Generate a sample for the emotion consistency benchmark
    emotions = ["pos", "neg1", "neg2", ... ]
    """
    positive_audio_data = dataset.get_audio(speaker=speaker, style=emotions[0], id=text_id)
    negative_audio_data_list = [dataset.get_audio(speaker=speaker, style=emotion, id=text_id) for emotion in emotions[1:]]

    text = positive_audio_data[0]
    positive_audio = Audio(positive_audio_data[1])
    positive_audio.write_audio(str(args.output_dir / f"sample_{index}_0.wav"))

    cut_index = int((len(positive_audio_data[2])-1) * args.split_ratio)
    positive_cut = (positive_audio_data[2][cut_index][1] + positive_audio_data[2][cut_index+1][0]) / 2
    for i, negative_audio_data in enumerate(negative_audio_data_list):
        negative_cut = (negative_audio_data[2][cut_index][1] + negative_audio_data[2][cut_index+1][0]) / 2  # start of second part
        negative_audio = positive_audio[:int(positive_cut * Audio.GENERAL_SAMPLING_RATE)] / Audio(negative_audio_data[1])[int(negative_cut * Audio.GENERAL_SAMPLING_RATE):]
        negative_audio.write_audio(args.output_dir / f"sample_{index}_{i + 1}.wav", normalize=args.normalize)

    return text

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', default='/cs/labs/adiyoss/amitroth/salmon_v2/emotion_consistency',
                        help='Path to save outputs')
    parser.add_argument('--n_samples', default=200, type=int, help='Number of samples for benchmark')
    parser.add_argument('--n_options', default=2, type=int, help='Number of options in each sample including the true')
    parser.add_argument('--split-ratio', default=0.5, type=float,
                        help='where to split the recording')
    parser.add_argument('--seed', default=43, type=int, help='random seed, use -1 for non-determinism')
    parser.add_argument('--normalize', default=False, type=bool, help='normalize final audio')


    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    args.output_dir = Path(args.output_dir)

    os.makedirs(Path(args.output_dir), exist_ok=True)
    seed_everything(args.seed)

    emotion_consistency(args)
