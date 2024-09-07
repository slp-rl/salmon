import json
import sys
from pathlib import Path
import os
import random
from collections import defaultdict
from tqdm import tqdm
import torch

from utils.audio import Audio
from utils.force_aligner import get_device, prepare_data, Tokenizer, ForceAligner, UROMAN_PATH, load_waveform, \
    get_emission

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

class BackgroundDataset:

    ALLOWED_CLASSES = [
        "Drill",
        "Gunshot, gunfire",
        "Drum kit",
        "Computer keyboard",
        "Bark",
        "Siren",
        "Applause",
        "Meow",
        "Chicken, rooster",
        "Cricket",
        "Frog",
        "Electric guitar",
        "Acoustic guitar",
        "Piano",
        "Marimba, xylophone",
        "Church bell",
        "Engine starting",
        "Doorbell",
        "Toilet flush",
        "Fireworks",
        "Waves"
    ]

    def __init__(self, path):
        self.path = Path(path)
        self.classes = list([p.name for p in self.path.glob("*/") if p.is_dir()])

        for allowed_class in BackgroundDataset.ALLOWED_CLASSES:
            if allowed_class not in self.classes:
                raise Exception(f"Invalid allowed class {allowed_class}")

    def get_classes(self):
        return BackgroundDataset.ALLOWED_CLASSES

    def get_audio_path(self, noise_class):
        if noise_class not in self.get_classes():
            raise Exception(
                f"{noise_class} is not a valid background noise class. choose a class from {self.get_classes()}")

        all_class = list((self.path / noise_class).glob("*.wav"))
        wav_path = random.choice(all_class)

        return wav_path

    def get_audio_list(self, noise_class, min_len=0):
        all_audios = list((self.path / noise_class).glob("*.wav"))
        filtered = [audio for audio in all_audios if len(Audio(audio)) > min_len]
        if len(filtered) == 0:
            raise Exception(f"No audio found for class {noise_class} with length > {min_len}")
        return filtered

    def sample(self, n_options, distract_method, speech_length):
        if distract_method == "in_domain":
            # sample all recordings from same class
            sampled_class = random.choice(self.get_classes())
            samples_in_class = random.sample(self.get_audio_list(sampled_class, speech_length), n_options)
            return samples_in_class

        elif distract_method == "random":
            # sample each recording from different class
            sampled_classes = random.sample(self.get_classes(), n_options)
            sample_for_class = [random.choice(self.get_audio_list(sampled_class, speech_length)) for
                                sampled_class in sampled_classes]
            return sample_for_class

        else:
            raise Exception(f"Invalid distract method {distract_method}")


class LJSpeechDataset:

    """
        this implementation works with a subset of LJSpeech dataset, all recordings in the specified folder
    """

    def __init__(self, path):
        self.path =Path(path)
        self.wavs = list(self.path.glob("*.wav"))

    def get_audio_path(self, max_length = 128000):
        # maximum length of 8 seconds
        while True:
            c = random.choice(self.wavs)
            if len(Audio(c)) < max_length:
                return c


class RirDataset:

    def __init__(self, path):
        self.path = Path(path)

    def get_audio_path(self, name):
        return self.path / f"{name}.wav"

    def get_audio_paths(self):
        return list(self.path.glob("*.wav"))


class VCTKDataset:
    # SPEAKERS_BLACKLIST = [270,280, 312, 230, 5, 292, 343, 335, 233, 274, 265, 227, 253, 307]
    SPEAKERS_BLACKLIST = [270]

    COMMON_TEXTS = range(1,24)

    def __init__(self, path, metadata_path=None, device="cuda:0"):
        self.path = Path(path)
        self.speakers = [int(p.stem[1:]) for p in Path(self.path / "txt").glob("*")]
        self.texts = range(0, 366)

        self.data = defaultdict(lambda: defaultdict(dict))
        self.metadata_path = metadata_path

        self.speaker_info = {}
        with open(self.path / "speaker-info.txt", 'r') as file:
            for line in file:
                columns = line.strip().split()
                speaker_id = columns[0][1:]
                gender = columns[2]
                self.speaker_info[speaker_id] = gender

        if os.path.isfile(self.metadata_path):
            with open(self.metadata_path) as f:
                self.data = json.load(f)
        else:
            print("Metadata not found, aligning dataset")
            model, c2i, sr = prepare_data(True, get_device(device))
            tokenizer = Tokenizer(c2i, uroman_path=UROMAN_PATH)
            aligner = ForceAligner(tokenizer=tokenizer)

            for text_id in range(1, 24):
                for speaker in self.get_speakers(text_id):
                    audio_path = self.get_audio_path(speaker_id=speaker, text_id=text_id)
                    text = self.get_text(speaker_id=speaker, text_id=text_id)

                    waveform = load_waveform(str(audio_path), sr)
                    emission = get_emission(waveform, model, device)
                    words_span = aligner(emission[0], text.lower().replace('-', ' '), return_as_words=True)
                    ratio = waveform.shape[1] / emission.shape[1]
                    alignment = list()
                    for w in words_span:
                        alignment.append(
                            (w.start * ratio / sr, w.end * ratio / sr)
                        )

                    self.data[speaker][text_id] = {"text": text, "audio_path": str(audio_path), "alignment": alignment}

            with open(self.metadata_path, "w") as f:
                json.dump(self.data, f)

    def get_texts(self, min_words=4):
        # get the text indexes with minimum number of words
        return [text_index for text_index in VCTKDataset.COMMON_TEXTS if len(self.get_text(speaker_id=226, text_id=text_index).split()) >= min_words]

    def get_gender(self, speaker_id):
        return self.speaker_info[str(speaker_id)]

    def get_speakers(self, text_id):
        # get all speakers for a given text
        return [s for s in self.speakers if
                self.get_audio_path(speaker_id=s, text_id=text_id).exists()
                and s not in VCTKDataset.SPEAKERS_BLACKLIST]

    def get_text(self, speaker_id, text_id):
        assert speaker_id in self.speakers
        assert text_id in self.texts

        with open(self.path / f"txt/p{speaker_id}/p{speaker_id}_{text_id:03}.txt") as f:
            text = f.readlines()[0].replace("\n", "")
        return text

    def get_audio_path(self, speaker_id, text_id):
        assert speaker_id in self.speakers
        assert text_id in self.texts
        return self.path / "wav16_trimmed_padded_wav" / f"p{speaker_id}_{text_id:03}_mic2.wav"

    def get_audio(self, speaker: int, text_id: int):
        assert text_id in self.texts
        assert speaker in self.get_speakers(text_id), f"{speaker} not in {self.get_speakers(text_id)}"

        return list(self.data[str(speaker)][str(text_id)].values())


class ExpressoDataset:
    # TODO change Expresso's init like VCTK and remove align_dataset()

    def __init__(self, path=Path("/cs/dataset/Download/adiyoss/expresso"), metadata_path=None):
        self.path = path
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.metadata_path = metadata_path

        if os.path.isfile(self.metadata_path):
            with open(self.metadata_path) as f:
                self.data = json.load(f)
        else:
            print("Warning: dataset is not aligned.")

            with open(self.path / "read_transcriptions.txt", 'r') as f:
                for l in f.readlines():
                    file_name, text = l.split("\t", 1)
                    speaker_style, id = file_name.rsplit('_', 1)
                    speaker, style = speaker_style.split("_", 1)
                    text = text.replace("\n", "")
                    style_dir = style.split("_")[0]
                    audio_path = str(
                        self.path / f"audio_48khz/read/{speaker}/{style_dir}/base/{speaker}_{style}_{id}.wav")
                    self.data[speaker][style][str(id)] = {"text": text, "audio_path": audio_path}

        self.speakers = list(self.data.keys())
        self.styles = ["happy", "sad", "whisper"]

    def get_speakers(self):
        return self.speakers

    def get_styles(self):
        return self.styles

    def get_ids(self, speaker, style):
        return [int(k) for k in self.data[speaker][style].keys()]

    def get_audio(self, speaker, style, id):
        assert speaker in self.get_speakers()
        assert style in self.get_styles()

        return list(self.data[speaker][style][str(id)].values())

    def get_indexes(self, speaker, style, minimum_word_count):
        return [int(d[0]) for d in self.data[speaker][style].items() if len(d[1]['text'].split()) >= minimum_word_count]

    def align_dataset(self, device="cuda:0"):
        model, c2i, sr = prepare_data(True, get_device(device))
        tokenizer = Tokenizer(c2i, uroman_path=UROMAN_PATH)
        print(c2i)
        aligner = ForceAligner(tokenizer=tokenizer)

        for speaker in self.get_speakers():
            for style in ["sad", "happy", "whisper"]:
                for id in tqdm(self.get_ids(speaker, style), desc=f"Aligning {speaker} {style}"):
                    audio = self.data[speaker][style][str(id)]
                    waveform = load_waveform(audio["audio_path"], sr)
                    emission = get_emission(waveform, model, device)
                    words_span = aligner(emission[0], audio["text"].lower().replace('-', ' '), return_as_words=True)
                    ratio = waveform.shape[1] / emission.shape[1]
                    audio["aligner"] = list()
                    for w in words_span:
                        audio["aligner"].append(
                            (w.start * ratio / sr, w.end * ratio / sr)
                        )

        with open(self.metadata_path, "w") as f:
            json.dump(self.data, f)


class TextDataset:

    def __init__(self, txt_path):
        self.path = txt_path

        with open(txt_path, 'r') as f:
            self.lines = f.readlines()

        self.len = len(self.lines)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        if item >= self.len:
            raise Exception(f"Dataset contains {self.len} lines. [0-{self.len - 1}]")

        return self.lines[item]

    def get_random_text(self):
        return self[random.randint(0, self.len - 1)]

    def sample_list_of_texts(self, n):
        assert n <= self.len
        return random.sample(self.lines, n)


