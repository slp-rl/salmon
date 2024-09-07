import argparse
import json
import os
import random
from pathlib import Path
from utils.microsoft_tts import generate_speech
from tqdm import tqdm
from utils.dataset import EmotionDataset, TextDataset
from itertools import zip_longest


def emotion_alignment(output_path, args):
    output_path = Path(output_path)

    speakers = [
        # {'speaker': "en-GB-SoniaNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-AriaNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-DavisNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-GuyNeural", 'styles': ["happy", "sad"], 'style_degree': 1.6},
        # {'speaker': "en-US-JaneNeural", 'styles': ["happy", "sad"], 'style_degree': 2.0},
        # {'speaker': "en-US-JasonNeural", 'styles': ["happy", "sad"], 'style_degree': 1.8},
        # {'speaker': "en-US-JennyNeural", 'styles': ["cheerful", "sad"]},
        # {'speaker': "en-US-NancyNeural", 'styles': ["cheerful", "sad"]},
        {'speaker': "en-US-SaraNeural", 'styles': [("happy", 2.0), ("sad", 1.6)]}
        # {'speaker': "en-US-TonyNeural", 'styles': ["cheerful", "sad"]},

    ]

    # samples texts
    sad_dataset = TextDataset("/cs/labs/adiyoss/amitroth/slm-benchmark/txt/emotion/sad.txt")
    happy_dataset = TextDataset("/cs/labs/adiyoss/amitroth/slm-benchmark/txt/emotion/happy.txt")
    sad_texts = sad_dataset.sample_list_of_texts(int(args.offset + args.samples_num / 2))
    happy_texts = happy_dataset.sample_list_of_texts(args.offset + args.samples_num - int(args.offset + args.samples_num / 2))
    alternating_texts = [
        item for pair in zip_longest(
            [(sad_text, "sad") for sad_text in sad_texts],
            [(happy_text, "happy") for happy_text in happy_texts]
        )
        for item in pair if item is not None]

    metadata = list()
    print(len(alternating_texts))
    for i in tqdm(range(args.offset, args.offset + args.samples_num), desc="Generating samples"):
        text, emotion = alternating_texts[i]
        speaker = random.choice(speakers)
        sample_output = generate_sample(output_path=output_path, text=text, text_emotion=emotion, speaker=speaker,
                                          index=i,
                                          args=args)

        sample_output = sorted(sample_output, key=lambda x: x[0])
        sample_metadata = {
            "index": i,
            "text_emotion": emotion,
            "text": text,
            "speaker": speaker["speaker"],
            "audio": [
                {"index": j, "speech_style": style, "audio_path": path} for j, style, path in sample_output
            ]
        }

        metadata.append(sample_metadata)

    with open(output_path / "metadata.json", "w") as f:
        json.dump(metadata, f)


def generate_sample(output_path, text, text_emotion, speaker, index, args):
    # the true sample must be with index 0
    # netural is 1
    # false is 2
    sample_metadata = list()
    for style in speaker['styles']:
        # path = str(output_path / f"sample_{index}_{speaker['speaker']}_{style}.wav")
        if text_emotion == style[0]:
            j = 0
        else:
            j = 1

        path = str(output_path / f"sample_{index}_{j}.wav")

        generate_speech(
            out_path=path,
            text=text,
            speaker=speaker['speaker'],
            style=style[0],
            style_degree=style[1]
        )

        sample_metadata.append(
            (j, style, path)
        )
    return sample_metadata


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=str,
        default="/cs/labs/adiyoss/amitroth/salmon_v2/emotion_alignment_v2/"
    )

    parser.add_argument(
        "--samples-num",
        type=int,
        help="amount of samples to generate",
        default=364,
    )

    parser.add_argument(
        "--offset",
        type=int,
        help="style degree",
        default=0,
    )

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    emotion_alignment(
        output_path=args.output_dir,
        args=args
    )
