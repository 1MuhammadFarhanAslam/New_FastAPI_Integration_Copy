from huggingface_hub import hf_hub_download
import numpy as np
import librosa
import torch
import torchaudio
from audiocraft.metrics import CLAPTextConsistencyMetric
import bittensor as bt

class MetricEvaluator:
    @staticmethod
    def calculate_snr(file_path, silence_threshold=1e-4, constant_signal_threshold=1e-2):
        snr_score = None
        try:
            audio_signal, _ = librosa.load(file_path, sr=None)
            if np.max(np.abs(audio_signal)) < silence_threshold:
                return -np.inf
            elif np.var(audio_signal) < constant_signal_threshold:
                return -np.inf
            signal_power = np.mean(audio_signal**2)
            noise_signal = librosa.effects.preemphasis(audio_signal)
            noise_power = np.mean(noise_signal**2)
            if noise_power < 1e-10:
                return np.inf
            snr_score = 10 * np.log10(signal_power / noise_power)
        except Exception as e:
            bt.logging.error(f"Error calculating SNR: {e}")
        return snr_score

    @staticmethod
    def calculate_hnr(file_path):
        hnr_score = None
        try:
            y, _ = librosa.load(file_path, sr=None)
            if np.max(np.abs(y)) < 1e-4 or np.var(y) < 1e-2:
                return 0
            harmonic, percussive = librosa.effects.hpss(y)
            harmonic_power = np.mean(harmonic**2)
            noise_power = np.mean(percussive**2)
            hnr_score = 10 * np.log10(harmonic_power / max(noise_power, 1e-10))
        except Exception as e:
            bt.logging.error(f"Error calculating HNR: {e}")
        return hnr_score

    @staticmethod
    def calculate_consistency(file_path, text):
        consistency_score = None
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pt_file = hf_hub_download(repo_id="lukewys/laion_clap", filename="music_audioset_epoch_15_esc_90.14.pt")
            clap_metric = CLAPTextConsistencyMetric(pt_file, model_arch='HTSAT-base').to(device)
            
            def convert_audio(audio, from_rate, to_rate, to_channels):
                resampler = torchaudio.transforms.Resample(orig_freq=from_rate, new_freq=to_rate)
                audio = resampler(audio)
                if to_channels == 1:
                    audio = audio.mean(dim=0, keepdim=True)
                return audio

            audio, sr = torchaudio.load(file_path)
            audio = convert_audio(audio, from_rate=sr, to_rate=sr, to_channels=1)

            clap_metric.update(audio.unsqueeze(0), [text], torch.tensor([audio.shape[1]]), torch.tensor([sr]))
            consistency_score = clap_metric.compute()
        except Exception as e:
            bt.logging.error(f"Error calculating music consistency score: {e}")
        return consistency_score

class Normalizer:
    @staticmethod
    def normalize_quality(quality_metric):
        return 1 / (1 + np.exp(-((quality_metric - 20) / 10)))

    @staticmethod
    def normalize_consistency(score):
        if score is not None:
            if score > 0:
                return (score + 1) / 2  # Normalizes from [0, 1] to [0.5, 1]
            else:
                return 0
        return 0

class Aggregator:
    @staticmethod
    def geometric_mean(scores):
        """Calculate the geometric mean of the scores, avoiding any non-positive values."""
        scores = [max(score, 0.0001) for score in scores.values()]  # Avoid math errors
        product = np.prod(scores)
        return product ** (1.0 / len(scores))

class MusicQualityEvaluator:
    def __init__(self):
        self.metric_evaluator = MetricEvaluator()
        self.normalizer = Normalizer()
        self.aggregator = Aggregator()

    def evaluate_music_quality(self, file_path, text=None):
        snr_score, hnr_score, consistency_score = None, None, None

        try:
            snr_score = self.metric_evaluator.calculate_snr(file_path)
            if snr_score in [None, -np.inf, np.inf]:
                snr_score = 0
            bt.logging.info(f'.......SNR......: {snr_score} dB')
        except Exception as e:
            bt.logging.error(f"Failed to calculate SNR: {e}")

        try:
            hnr_score = self.metric_evaluator.calculate_hnr(file_path)
            if hnr_score is None:
                hnr_score = 0
            bt.logging.info(f'.......HNR......: {hnr_score} dB')
        except Exception as e:
            bt.logging.error(f"Failed to calculate HNR: {e}")

        try:
            consistency_score = self.metric_evaluator.calculate_consistency(file_path, text)
            if consistency_score is None:
                consistency_score = 0
            bt.logging.info(f'.......Consistency Score......: {consistency_score}')
        except Exception as e:
            bt.logging.error(f"Failed to calculate Consistency score: {e}")

        # Normalize scores and calculate aggregate score
        normalized_snr = self.normalizer.normalize_quality(snr_score)
        normalized_hnr = self.normalizer.normalize_quality(hnr_score)
        normalized_consistency = self.normalizer.normalize_consistency(consistency_score)

        aggregate_quality = self.aggregator.geometric_mean({'snr': normalized_snr, 'hnr': normalized_hnr})
        aggregate_score = self.aggregator.geometric_mean({'quality': aggregate_quality, 'normalized_consistency': normalized_consistency}) if consistency_score > 0.2 else 0

        bt.logging.info(f'Normalized Metrics: SNR = {normalized_snr}dB, HNR = {normalized_hnr}, Consistency = {normalized_consistency}')
        bt.logging.info(f'.......Aggregate Score......: {aggregate_score}')
        return aggregate_score
