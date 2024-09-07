import julius
import soundfile as sf
import torchaudio

import torch
import sys
import typing as tp



class Audio:

    GENERAL_SAMPLING_RATE = 16000

    def __init__(self, path=None):
        if path is not None:
            self.load_audio(path)
        else:
            self.data = None

    def load_audio(self, path):
        data, sr = sf.read(path)
        data = torch.from_numpy(data)

        if data.ndim == 2:
            data = torch.mean(data, axis=1)

        if sr != Audio.GENERAL_SAMPLING_RATE:
            data = Audio.resample(data, sr, Audio.GENERAL_SAMPLING_RATE)

        self.data = Audio.normalize_audio(data)

    def write_audio(self, out_path, normalize=True):
        assert self.data.dtype.is_floating_point, "wav is not floating point"
        # assert self.data.isfinite().all()
        if not self.data.isfinite().all():
            print("Warning: audio is not finite")
        if normalize:
            self.data = Audio.normalize_audio(self.data)
        sf.write(file=out_path, data=self.data, samplerate=Audio.GENERAL_SAMPLING_RATE)

    def pad(self, n):
        a = Audio()
        a.data = torch.nn.functional.pad(self.data, (0, n - len(self)), mode='constant', value=0)
        return a


    def __add__(self, other):
        """
            sum 2 recordings together
        """
        min_length = min(len(self), len(other))
        merged_data = torch.stack((self.data[:min_length], other.data[:min_length]), dim=1)
        sum_data = merged_data.sum(dim=1) / 2

        sum = Audio()
        sum.data = sum_data
        return sum

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        a = Audio()
        a.data = self.data[index]
        return a

    def __mul__(self, snr):
        """
            mul by float
        """
        result = Audio()
        result.data = self.data * float(snr)
        return result

    def __rmul__(self, snr):
        """
            mul by float
        """
        return self.__mul__(snr)

    def __truediv__(self, other):
        """
            concat audio one after the other
            a = b / c
        """
        result = Audio()
        result.data = torch.cat((self.data, other.data), dim=0)
        return result

    def __str__(self):
        return f"Audio[{len(self)}]"

    @staticmethod
    def audio_from_wav(wav, sr, normalize=True):
        a = Audio()
        if sr != Audio.GENERAL_SAMPLING_RATE:
            wav = Audio.resample(wav, sr, Audio.GENERAL_SAMPLING_RATE)

        if normalize:
            a.data = Audio.normalize_audio(wav)
        else:
            a.data = wav

        return a

    @staticmethod
    def normalize_audio(wav: torch.Tensor, normalize: bool = True,
                        strategy: str = 'peak', peak_clip_headroom_db: float = 1,
                        rms_headroom_db: float = 18, loudness_headroom_db: float = 14,
                        loudness_compressor: bool = False, log_clipping: bool = False,
                        sample_rate: tp.Optional[int] = None,
                        stem_name: tp.Optional[str] = None) -> torch.Tensor:
        """Normalize the audio according to the prescribed strategy (see after).

        Args:
            wav (torch.Tensor): Audio data.
            normalize (bool): if `True` (default), normalizes according to the prescribed
                strategy (see after). If `False`, the strategy is only used in case clipping
                would happen.
            strategy (str): Can be either 'clip', 'peak', or 'rms'. Default is 'peak',
                i.e. audio is normalized by its largest value. RMS normalizes by root-mean-square
                with extra headroom to avoid clipping. 'clip' just clips.
            peak_clip_headroom_db (float): Headroom in dB when doing 'peak' or 'clip' strategy.
            rms_headroom_db (float): Headroom in dB when doing 'rms' strategy. This must be much larger
                than the `peak_clip` one to avoid further clipping.
            loudness_headroom_db (float): Target loudness for loudness normalization.
            loudness_compressor (bool): If True, uses tanh based soft clipping.
            log_clipping (bool): If True, basic logging on stderr when clipping still
                occurs despite strategy (only for 'rms').
            sample_rate (int): Sample rate for the audio data (required for loudness).
            stem_name (str, optional): Stem name for clipping logging.
        Returns:
            torch.Tensor: Normalized audio.
        """
        scale_peak = 10 ** (-peak_clip_headroom_db / 20)
        scale_rms = 10 ** (-rms_headroom_db / 20)
        if strategy == 'peak':
            rescaling = (scale_peak / wav.abs().max())
            if normalize or rescaling < 1:
                wav = wav * rescaling
        elif strategy == 'clip':
            wav = wav.clamp(-scale_peak, scale_peak)
        elif strategy == 'rms':
            mono = wav.mean(dim=0)
            rescaling = scale_rms / mono.pow(2).mean().sqrt()
            if normalize or rescaling < 1:
                wav = wav * rescaling
            Audio._clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
        elif strategy == 'loudness':
            assert sample_rate is not None, "Loudness normalization requires sample rate."
            wav = Audio.normalize_loudness(wav, sample_rate, loudness_headroom_db, loudness_compressor)
            Audio._clip_wav(wav, log_clipping=log_clipping, stem_name=stem_name)
        else:
            assert wav.abs().max() < 1
            assert strategy == '' or strategy == 'none', f"Unexpected strategy: '{strategy}'"
        return wav

    @staticmethod
    def _clip_wav(wav: torch.Tensor, log_clipping: bool = False, stem_name: tp.Optional[str] = None) -> None:
        """Utility function to clip the audio with logging if specified."""
        max_scale = wav.abs().max()
        if log_clipping and max_scale > 1:
            clamp_prob = (wav.abs() > 1).float().mean().item()
            print(f"CLIPPING {stem_name or ''} happening with proba (a bit of clipping is okay):",
                  clamp_prob, "maximum scale: ", max_scale.item(), file=sys.stderr)
        wav.clamp_(-1, 1)

    def mix_audios(self, noises_list: tp.List, snr):
        # combines Audio with different background noises, normalizes all by same value
        signal = self.data
        signal_energy = torch.mean(signal ** 2)

        max_a = 0
        noise_params = []

        for noise in noises_list:
            noise = noise.data
            noise = noise[torch.arange(len(signal)) % len(noise)]
            noise_energy = torch.mean(noise ** 2)
            g = torch.sqrt(10.0 ** (-snr / 10) * signal_energy / noise_energy)
            a = torch.sqrt(1 / (1 + g ** 2))
            b = torch.sqrt(g ** 2 / (1 + g ** 2))
            noise_params.append((a, b, noise))
            max_a = max(max_a, a)

        mixed_signals = []
        for a, b, noise in noise_params:
            mixed_signal = max_a * signal + b * noise
            mixed_signals.append(Audio.audio_from_wav(mixed_signal, 16000, normalize=False))

        return mixed_signals

    @staticmethod
    def mix_audio_pairs(audio_pairs: tp.List[tp.Tuple], snr):
        # combines Audio with different background noises, normalizes all by same value
        max_a = 0
        mix_params = []

        for speech, noise in audio_pairs:
            # noise = Audio.normalize_audio(noise[:len(speech)].data)
            signal = speech.data
            signal_energy = torch.mean(signal ** 2)

            noise = noise.data
            noise = noise[torch.arange(len(signal)) % len(noise)]
            noise_energy = torch.mean(noise ** 2)
            g = torch.sqrt(10.0 ** (-snr / 10) * signal_energy / noise_energy)
            a = torch.sqrt(1 / (1 + g ** 2))
            b = torch.sqrt(g ** 2 / (1 + g ** 2))
            mix_params.append((a, b, signal, noise))
            max_a = max(max_a, a)

        mixed_signals = []
        for a, b, signal, noise in mix_params:
            mixed_signal = max_a * signal + b * noise
            mixed_signals.append(Audio.audio_from_wav(mixed_signal, 16000, normalize=False))

        return mixed_signals



    @staticmethod
    def normalize_loudness(wav: torch.Tensor, sample_rate: int, loudness_headroom_db: float = 14,
                           loudness_compressor: bool = False, energy_floor: float = 2e-3):
        """Normalize an input signal to a user loudness in dB LKFS.
        Audio loudness is defined according to the ITU-R BS.1770-4 recommendation.

        Args:
            wav (torch.Tensor): Input multichannel audio data.
            sample_rate (int): Sample rate.
            loudness_headroom_db (float): Target loudness of the output in dB LUFS.
            loudness_compressor (bool): Uses tanh for soft clipping.
            energy_floor (float): anything below that RMS level will not be rescaled.
        Returns:
            torch.Tensor: Loudness normalized output data.
        """
        energy = wav.pow(2).mean().sqrt().item()
        if energy < energy_floor:
            return wav
        transform = torchaudio.transforms.Loudness(sample_rate)
        input_loudness_db = transform(wav).item()
        # calculate the gain needed to scale to the desired loudness level
        delta_loudness = -loudness_headroom_db - input_loudness_db
        gain = 10.0 ** (delta_loudness / 20.0)
        output = gain * wav
        if loudness_compressor:
            output = torch.tanh(output)
        assert output.isfinite().all(), (input_loudness_db, wav.pow(2).mean().sqrt())
        return output

    @staticmethod
    def resample(wav, orig_sr, target_sr):
        return julius.resample_frac(wav, int(orig_sr), int(target_sr))


if __name__ == '__main__':
    bg = Audio("/Users/amitroth/PycharmProjects/slm-benchnark/audio/background_noises_samples/Applause/198087.wav")
    speech = Audio("/Users/amitroth/PycharmProjects/slm-benchnark/audio/lj_speech/LJ001-0001.wav")

    out1 = bg + 3 * speech
    out1.write_audio("output1.wav")

    out2 = bg / (5 * bg)
    out2.write_audio("output2.wav")





# def concat_audio(data_1, data_2):
#     """
#         audio should be in same sampling rates
#     """
#     combined_audio = np.concatenate((data_1, data_2))
#     return combined_audio
#
#
# def resample(y, orig_sr, target_sr):
#     return librosa.resample(y=y, orig_sr=orig_sr, target_sr=target_sr)
#
#
# def load_audio(path):
#     data, sr = sf.read(path)
#     return data, sr
#
#
# def write_audio(path, data, sr):
#     sf.write(file=path, data=data, samplerate=sr)
#
#
# def normalize_audio(data):
#     peak_normalized_audio = pyln.normalize.peak(data, -1.0)
#     return peak_normalized_audio


