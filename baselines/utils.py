import os
from typing import List, Tuple
import zipfile
import wget
import torch
from transformers import AutoModelForCausalLM, AutoConfig
from textless.data.speech_encoder import SpeechEncoder
from textless.data.f0_preprocess import PromptNormalize, F0BinQuantizer
from accelerate import init_empty_weights, load_checkpoint_and_dispatch, infer_auto_device_map
from fairseq import checkpoint_utils
from torch import FloatTensor, LongTensor
from torch.nn.functional import cross_entropy
import string


FAIRSEQ = "fairseq"
HF = "hf"

"""
This file contains utils for speech_lm
"""

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ROOT_URL = 'https://dl.fbaipublicfiles.com/textless_nlp/twist/lms/'


def get_gslm_speech_encoder(dense_model_name, quantizer_model_name, vocab_size,
                            deduplicate, need_f0, f0_func="yaapt", f0_normalizer=None, f0_quantizer=None):
    """
    get speech encoder using textless library
    :param dense_model_name: dense model name
    :param quantizer_model_name: quantizer model name
    :param vocab_size: vocab size
    :param deduplicate: deduplicate
    :param need_f0: need f0
    """
    return SpeechEncoder.by_name(
        dense_model_name=dense_model_name,
        quantizer_model_name=quantizer_model_name,
        vocab_size=vocab_size,
        deduplicate=deduplicate,
        need_f0=need_f0,
        f0_normalizer=f0_normalizer,
        f0_quantizer=f0_quantizer
    ) #TODO add         f0_func=f0_func



def get_audio_files_from_manifest(manifest_path: str) -> Tuple[str, List[str], List[int]]:
    fnames, sizes = [], []
    with open(manifest_path, "r") as f:
        root_dir = f.readline().strip()
        for line in f:
            items = line.strip().split("\t")
            assert (
                    len(items) == 2
            ), f"File must have two columns separated by tab. Got {line}"
            fnames.append(items[0])
            sizes.append(int(items[1]))
    return root_dir, fnames, sizes


def unzip_file(zip_path, extract_path):
    """
    unzip file
    :param zip_path: path to zip file
    :param extract_path: path to extract to
    """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"File extracted to {extract_path}")


def maybe_download_speech_lm(name, base_path):
    """
    downloads speech lm
    :param name: name of model
    :param base_path: base path to download to
    """
    print(base_path)
    if not os.path.exists(base_path):
        os.mkdir(base_path)

    ckpt_dir = os.path.join(base_path, name)
    if not os.path.exists(ckpt_dir):
        url = ROOT_URL + name + '.zip'
        zip_path = ckpt_dir + '.zip'
        print(f"Downloading from {url}")
        filename = wget.download(url, zip_path)
        unzip_file(filename, ckpt_dir)

    return os.path.abspath(ckpt_dir)


def build_speech_lm(model_type, multi_gpu_load=False, base_path='/cs/labs/adiyoss/amitroth/ckpts/'):
    """
    builds speech lm
    retruns model
    """
    ckpt_dir = maybe_download_speech_lm(model_type, base_path)

    if multi_gpu_load:
        config = AutoConfig.from_pretrained(ckpt_dir)
        with init_empty_weights():
            lm_model = AutoModelForCausalLM.from_config(config)
        device_map = infer_auto_device_map(lm_model, max_memory={0: "30GiB"})
        lm_model = load_checkpoint_and_dispatch(lm_model, ckpt_dir, device_map=device_map,
                                                no_split_module_classes=["model.layers"])
    else:
        lm_model = AutoModelForCausalLM.from_pretrained(ckpt_dir)
    lm_model.eval()

    return lm_model


def build_pgslm_speech_lm(model_type, data_config, base_path='/cs/labs/adiyoss/amitroth/ckpts/'):
    print(f"loading model from {os.path.join(base_path, model_type)}")
    models, _, task = checkpoint_utils.load_model_ensemble_and_task([os.path.join(base_path, model_type)],
                                                                    arg_overrides={"data": data_config})
    model = models[0]
    model.eval()
    return model, task


def get_pgslm_speech_encoder(config):
    mean, scale, log = config['mean_f0'], config['scale_f0'], config['log_f0']
    f0_normalizer = None
    if mean or scale or log:
        f0_normalizer = PromptNormalize(mean, scale, log)
    f0_bins_path = config.get('f0_bins_path', None)
    f0_quantizer = None
    if f0_bins_path is not None:
        f0_quantizer = F0BinQuantizer(f0_bins_path)

    tokenizer = get_gslm_speech_encoder(
        dense_model_name=config['dense_model_name'],
        quantizer_model_name=config['quantizer_model_name'],
        vocab_size=config['encoder_vocab_size'],
        deduplicate=config['deduplicate'],
        need_f0=config['need_f0'],
        f0_normalizer=f0_normalizer,
        f0_quantizer=f0_quantizer,
        f0_func=config.get('f0_func', 'yaapt'),

    )

    return tokenizer


def nll(logits: FloatTensor, target: LongTensor, mask: LongTensor, mean_nll:bool = False) -> FloatTensor:
    """
    calculate the negative log likelihood of the logits given the target
    :param logits: logits
    :param target: target
    :param mask: mask
    :return: nll
    """
    # Calculate the cross-entropy loss for each sequence
    losses = cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)),
        target.long().contiguous().view(-1), reduction='none')

    # Reshape the losses to match the original sequences
    losses = losses.view(*target.size())

    # Use the mask to ignore the losses of the padding tokens
    masked_losses = losses * mask

    # Sum the losses to get the total loss for each sequence
    ll = masked_losses.sum(dim=-1)
    if mean_nll:
        return ll / mask.sum(dim=-1)
    return ll


def probability(logits: FloatTensor, target: LongTensor, mask: LongTensor) -> FloatTensor:
    """
    calculate the probability over time of the logits given the target
    :param logits: logits
    :param target: target
    :param mask: mask
    :return: nll
    """
    # Calculate the cross-entropy loss for each sequence
    losses = cross_entropy(
        logits.contiguous().view(-1, logits.size(-1)),
        target.long().contiguous().view(-1), reduction='none')

    # Reshape the losses to match the original sequences
    losses = losses.view(*target.size())

    # Use the mask to ignore the losses of the padding tokens
    masked_losses = losses * mask

    # Sum the losses to get the total loss for each sequence
    return masked_losses


def build_lm(model_type, library, data_config=None, base_path='./'):
    """
    returns speech lm model that can be used for inference
    :param model_type: model type
    :param library: library to use to load model
    :param data_config: data config for fairseq, might be None
    :param base_path: base path to download model to (if not already downloaded)
    return: model
    """
    if library == FAIRSEQ:
        return build_pgslm_speech_lm(model_type, data_config, base_path)
    elif library == HF:
        return build_speech_lm(model_type, base_path=base_path)
    else:
        raise ValueError(f"Library {library} not supported")



def remove_spaces_and_punctuation(text):
    # Create a translation table that maps punctuation to None
    translator = str.maketrans('', '', string.punctuation)
    # Remove punctuation using the translation table
    no_punctuation = text.translate(translator)
    # Remove spaces
    no_spaces = no_punctuation.replace(" ", "")
    return no_spaces