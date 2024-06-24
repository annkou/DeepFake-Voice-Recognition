import os
import librosa
import numpy as np
import torch
import torchaudio


class AudioPreprocessor:

    def __init__(
        self,
        audio_dir,
        target_sample_rate,
        annotations_file,
        noise_threshold=0.005,
        device="cpu",
    ):
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.device = device
        self.annotations_file = annotations_file
        self.noise_threshold = noise_threshold

    def __getitem__(self, file_path):
        try:
            signal, sr = torchaudio.load(file_path)
            signal = signal.to(self.device)
            label = self.annotations_file[
                self.annotations_file["file_path"] == file_path
            ]["label"].values[0]
            signal = self._downsample(signal, sr)
            signal = self._turn_to_mono(signal)
            ############################# remove silence step is not needed #############################
            # signal = self._remove_silence(signal)
            return signal, self.target_sample_rate
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def _downsample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate).to(
                self.device
            )
            signal = resampler(signal)
        return signal

    def _turn_to_mono(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _remove_silence(self, signal):
        signal_np = signal.cpu().numpy()
        rms = librosa.feature.rms(y=signal_np[0])
        mask = rms > self.noise_threshold
        mask = np.repeat(mask, librosa.frames_to_samples(1, hop_length=512))
        mask = mask[: len(signal_np[0])]
        signal_np = signal_np[:, mask[0]]
        return torch.from_numpy(signal_np).to(self.device)

    def save_cleaned_audio(self, signal, sr, output_path):
        torchaudio.save(output_path, signal.cpu(), sr)

    def clean_audio_files(self, output_dir):
        # Create output directories for real and fake if they don't exist
        os.makedirs(os.path.join(output_dir, "REAL"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "FAKE"), exist_ok=True)

        # Process files in REAL and FAKE directories
        for label in ["REAL", "FAKE"]:
            input_dir = os.path.join(self.audio_dir, label)
            for file_name in os.listdir(input_dir):
                file_path = os.path.join(input_dir, file_name)
                signal, sr = self[file_path]
                if signal is not None:
                    output_path = os.path.join(output_dir, label, file_name)
                    self.save_cleaned_audio(signal, sr, output_path)
                    print(f"Saved cleaned audio to {output_path}")
