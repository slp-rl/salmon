from __future__ import annotations

import json
import sys
import csv
from pathlib import Path
from glob import iglob, glob
import shutil
import os
import random
import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils.general import regex_rglob
from utils.audio import safe_load_audio, get_length, safe_write_audio, concat_audio, load_audio


def parse_RAVDESS(original_dataset_path, parsed_dataset_path):
    """
        dont need to run this function, wavs appended to git.
    """
    original_dataset_path = Path(original_dataset_path)
    parsed_dataset_path = Path(parsed_dataset_path)
    neutral_dataset_path = parsed_dataset_path / "neutral"
    happy_dataset_path = parsed_dataset_path / "happy"
    sad_dataset_path = parsed_dataset_path / "sad"

    for p in [neutral_dataset_path, happy_dataset_path, sad_dataset_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    neutral_wavs = regex_rglob(original_dataset_path, "03-01-01-01-0.-01-..\.wav")
    happy_wavs = regex_rglob(original_dataset_path, "03-01-03-02-0.-01-..\.wav")
    sad_wavs = regex_rglob(original_dataset_path, "03-01-04-02-0.-01-..\.wav")

    for wav_path in neutral_wavs:
        wav = safe_load_audio(wav_path)
        safe_write_audio(neutral_dataset_path / wav_path.name, wav)

    for speaker in range(1, 25):
        wav_1 = safe_load_audio(
            parsed_dataset_path / "neutral" / "03-01-01-01-01-01-{:02d}.wav".format(speaker)
        )

        wav_2 = safe_load_audio(
            parsed_dataset_path / "neutral" / "03-01-01-01-02-01-{:02d}.wav".format(speaker)
        )

        wav_3 = concat_audio(wav_1, wav_2)

        safe_write_audio(parsed_dataset_path / "neutral" / "03-01-01-01-03-01-{:02d}.wav".format(speaker), wav_3)

    for wav_path in happy_wavs:
        wav = safe_load_audio(wav_path)
        safe_write_audio(happy_dataset_path / wav_path.name, wav)

    for speaker in range(1, 25):
        wav_1 = safe_load_audio(
            parsed_dataset_path / "happy" / "03-01-03-02-01-01-{:02d}.wav".format(speaker)
        )

        wav_2 = safe_load_audio(
            parsed_dataset_path / "happy" / "03-01-03-02-02-01-{:02d}.wav".format(speaker)
        )

        wav_3 = concat_audio(wav_1, wav_2)

        safe_write_audio(parsed_dataset_path / "happy" / "03-01-03-02-03-01-{:02d}.wav".format(speaker), wav_3)

    for wav_path in sad_wavs:
        wav = safe_load_audio(wav_path)
        safe_write_audio(sad_dataset_path / wav_path.name, wav)

    for speaker in range(1, 25):
        wav_1 = safe_load_audio(
            parsed_dataset_path / "sad" / "03-01-04-02-01-01-{:02d}.wav".format(speaker)
        )

        wav_2 = safe_load_audio(
            parsed_dataset_path / "sad" / "03-01-04-02-02-01-{:02d}.wav".format(speaker)
        )

        wav_3 = concat_audio(wav_1, wav_2)

        safe_write_audio(parsed_dataset_path / "sad" / "03-01-04-02-03-01-{:02d}.wav".format(speaker), wav_3)


def get_leaves(ontology_path, output_path):
    """
        get a lost of leaves from audioset's ontology
    """
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)

    leaves = [o['name'] for o in ontology if len(o['child_ids']) == 0]

    with open(output_path, 'w') as f:
        json.dump(leaves, f)


def parse_FSD(original_dataset_path, parsed_dataset_path, leaves_path):
    with open(leaves_path, 'r') as f:
        leaves = json.load(f)

    data = dict()
    for leaf_id, leaf_name in tqdm.tqdm(leaves.items()):
        diff_leaves = set(leaves.keys())
        diff_leaves.remove(leaf_id)
        count = process_FSD_class(original_dataset_path, parsed_dataset_path, leaf_name, leaf_id, diff_leaves,
                                  min_length=5, max_length=15)
        data[leaf_name] = count

    print(data)
    with open("/cs/labs/adiyoss/amitroth/slm-benchmark/audio/background_noises/metadata.json", 'w') as f:
        json.dump(data, f)


def process_FSD_class(original_dataset_path, parsed_dataset_path, class_name, class_id, leaves, min_length=5,
                      max_length=30):
    original_dataset_path = Path(original_dataset_path)
    parsed_dataset_path = Path(parsed_dataset_path)

    p = parsed_dataset_path / class_name
    if not os.path.exists(p):
        os.makedirs(p)

    def single_leaf_id(classes, class_id, leaves):
        if class_id not in classes:
            return False

        for leaf in leaves:
            if leaf in classes:
                return False

        return True

    count = 0
    with open(original_dataset_path, 'r') as f:
        gt_csv = csv.reader(f, delimiter=',')

        lines = list(gt_csv)
        for l in lines[1:]:
            if single_leaf_id(l[2], class_id, leaves):
                wav, sr = load_audio(f"/cs/labs/adiyoss/amitroth/datasets/FSD50K.dev_audio/{l[0]}.wav")
                length = get_length(wav, sr)

                if min_length <= length <= max_length:
                    count += 1
                    shutil.copy(f"/cs/labs/adiyoss/amitroth/datasets/FSD50K.dev_audio/{l[0]}.wav", p / f"{l[0]}.wav")

    return count


def get_FSD50K_leaves(audioset_ontology_path, vocabulary_path):
    with open(vocabulary_path, 'r') as f:
        vocab = list(csv.reader(f, delimiter=','))
        vocab_ids = [l[2] for l in vocab]

    with open(audioset_ontology_path, 'r') as f:
        ontology = json.load(f)

    def nodes_in_vocab(child_ids):
        count = 0
        for id in child_ids:
            if id in vocab_ids:
                count += 1
        return count

    leaves = {o['id']: o['name'] for o in ontology if nodes_in_vocab(o['child_ids']) == 0 and o["id"] in vocab_ids}
    print(leaves)
    return leaves


def sample_dir(input_path: pathlib.Path, output_path: pathlib.Path, samples_num=3):
    wavs = list(input_path.glob("*"))

    if len(wavs) == 0:
        return

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    sampled_wavs = random.sample(wavs, min(samples_num, len(wavs)))

    for sample in sampled_wavs:
        shutil.copy(sample, output_path / sample.name)


def glob_dirs(input_path: pathlib.Path, output_path: pathlib.Path):
    dirs = [dir for dir in input_path.glob("*") if os.path.isdir(dir)]

    for dir in dirs:
        sample_dir(dir, output_path / dir.name)
