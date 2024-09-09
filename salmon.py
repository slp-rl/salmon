import argparse
from torch.utils.data import DataLoader, Dataset
import torchaudio
import json
import torch
# from tqdm import tqdm
from pathlib import Path

from baselines.inference import InferenceModelFactory


class SalmonDataset(Dataset):
    def __init__(self, salmon_path, part, load_audio=True):
        self.data = []
        self.salmon_path = Path(salmon_path)
        self.load_audio = load_audio
        dir_path = self.salmon_path / part
        paths = list(dir_path.glob("*.wav"))

        max_sample_index = -1
        for path in paths:
            stem = str(path.stem)
            parts = stem.split("_")
            sample_index = int(parts[1])
            if sample_index > max_sample_index:
                max_sample_index = sample_index

        self.data = [[] for _ in range(max_sample_index + 1)]

        for path in paths:
            stem = str(path.stem)
            parts = stem.split("_")
            sample_index = int(parts[1])
            self.data[sample_index].append(str(path))

        for sample_list in self.data:
            sample_list.sort()

        self.data = [lst for lst in self.data if lst]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_files = self.data[idx]
        if self.load_audio:
            sample_audios = [torchaudio.load(sample_file) for sample_file in sample_files]
            return [s[0] for s in sample_audios]
        else:
            return sample_files


def collate_fn(batch):
    pos, neg = zip(*batch)
    return list(pos), list(neg)


def main():
    parser = argparse.ArgumentParser(description='Run SALMon')
    parser.add_argument("-s", "--salmon_path", type=str, help="Path to the downloaded SALMon dataset")
    parser.add_argument("-c", "--inference_model_config", type=str, required=True, help="inference model config json")
    parser.add_argument("-p", "--parts", type=str, nargs="+", default=["all"], help="parts")
    parser.add_argument("-b", "--batch_size", type=int, default=1, help="batch size")

    args = parser.parse_args()
    salmon_path = args.salmon_path
    config_path = args.inference_model_config

    with open(config_path) as f:
        inference_model_config = json.load(f)

    inference_model = InferenceModelFactory.get_model(inference_model_config)

    if torch.cuda.is_available():
        inference_model = inference_model.to("cuda")

    if args.parts[0] == "all":
        args.parts = [
            'bg_alignment/before',
            'bg_all_consistency/',
            'bg_domain_consistency/',
            'gender_consistency/',
            'rir_consistency/',
            'sentiment_alignment/',
            'sentiment_consistency/',
            'speaker_consistency/',
        ]

    print(f"Calculating {len(args.parts)} parts of SALMon for {inference_model} model")

    for part in args.parts:
        dataset = SalmonDataset(salmon_path, part, load_audio=True)
        assert len(dataset) > 0, f"no samples found for {part}"
        dataloader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        res_list = []

        with torch.no_grad():
            for sample_files in dataloader:
                pos_sample, neg_sample = sample_files
                pos_likelihood = inference_model.log_likelihood(pos_sample)
                neg_likelihood = inference_model.log_likelihood(neg_sample)
                res = torch.zeros_like(pos_likelihood)

                res[pos_likelihood > neg_likelihood] = 1
                res[pos_likelihood == neg_likelihood] = 0.5
                res[pos_likelihood < neg_likelihood] = 0

                res_list.append(res)

        res_list = torch.cat(res_list)
        print(f"SALMon [{part}]: {res_list.float().mean().cpu():.3f}")


if __name__ == "__main__":
    main()
