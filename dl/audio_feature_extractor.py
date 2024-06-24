import os
import pickle

import tqdm
import librosa
import torchaudio
from torch._prims_common import DeviceLikeType

import numpy as np
import pandas as pd


class AudioFeatureExtractor:

    def __init__(
        self,
        input_dir: str,
        annotations_df: pd.DataFrame,
        segment_length: float,
        output_dir: str = "mel_spectograms",
        audio_features_filename: str = "audio_features.csv",
        image_annotation_filename: str = "image_annotations.csv",
        oversample: bool = False,
        oversample_ignore: list[str] = [],
        device: DeviceLikeType = "cpu",
        verbose: bool = False,
    ):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.device = device
        self.annotations_df = annotations_df
        self.segment_length = segment_length
        self.image_annotations = []
        self.audio_features_filename = audio_features_filename
        self.image_annotation_filename = image_annotation_filename
        self.verbose = verbose
        self.oversample = oversample
        self.oversample_ignore = oversample_ignore

    def _load_and_preprocess(self, file_path):
        """Load and preprocess audio file."""
        try:
            signal, sr = torchaudio.load(file_path)
            return signal.to(self.device), sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None

    def _split_and_convert_to_melgram(
        self,
        signal,
        sr,
        file_name,
        label,
        oversample=False,
    ):
        """AUDIO FEATURES: \n
        Extracts features from an audio signal. \n
        Args:
            signal (torch.Tensor): Audio signal.
            sr (int): Sample rate of the audio signal.
            file_name (str): Name of the audio file.
            label (str): Label of the audio file.
            oversample (bool): Whether to oversample the segments or not. Oversampling is done by shifting the segment by half of its length.
        Returns:
            Directory: Directory containing mel spectrograms in pickle format.
            List: List of annotations for images.
        """
        total_samples = signal.shape[1]

        # Calculate the number of segments based on the segment length and audio length
        segment_samples = int(self.segment_length * sr)
        num_segments = int(np.ceil(total_samples / float(segment_samples)))

        if oversample:
            num_segments *= 2

        if self.verbose:
            print(
                f"{file_name}(spectogram):\n  Total samples: {total_samples}, Segment samples: {segment_samples}, Num segments: {num_segments}"
            )

        os.makedirs(self.output_dir, exist_ok=True)
        for i in range(num_segments):

            if oversample:
                start_frame = i * np.floor(segment_samples / 2.0)
                end_frame = min(total_samples, start_frame + segment_samples)
            else:
                start_frame = i * segment_samples
                end_frame = min(total_samples, (i + 1) * segment_samples)

            start_frame = int(start_frame)
            end_frame = int(end_frame)

            segment = signal[:, start_frame:end_frame].to("cpu")

            # Can pad or trim the feature array to a fixed length or simply ignore it
            if segment.shape[1] < segment_samples:
                continue  # Ignore segments that are shorter than the desired length

            # Convert segment to numpy array if needed
            y_segment = segment.numpy().squeeze()

            # Compute the mel spectrogram with specific window size (25ms)
            n_fft = int(sr * 0.05)  # Number of samples in a 50ms window
            hop_length = int(
                sr * 0.0235
            )  # Hop length of 25ms (fix it show final shape is as squared as possible) --> 128x128

            mel_spectrogram = librosa.feature.melspectrogram(
                y=y_segment,
                sr=sr,
                n_mels=128,
                win_length=500,
                n_fft=n_fft,
                hop_length=hop_length,
                fmax=sr / 2,
            )
            # hop_length=160, n_fft=1024, n_mels=128, fmax=8000
            mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # mel_spectrograms.append(mel_spectrogram_db)
            # print(mel_spectrogram_db.shape)

            # Append annotation
            self.image_annotations.append(
                {
                    "image_name": f"{file_name}_{i}",
                    "original_sample": file_name,
                    "LABEL": label,
                }
            )

            # Save the results in a pickle file
            pickle_file = os.path.join(self.output_dir, f"{file_name}_{i}.pkl")
            with open(pickle_file, "wb") as f:
                pickle.dump(mel_spectrogram_db, f)

    def _extract_features(self, signal, sr, file_name, oversample=False):
        """AUDIO FEATURES: \n
        Extracts features from an audio signal. \n
        Args:
            signal (torch.Tensor): Audio signal.
            sr (int): Sample rate of the audio signal.
            file_name (str): Name of the audio file.
        Returns:
            List: List of audio features.
        """
        try:
            # Calculate the number of segments based on the segment length and audio length
            # segment size in samples
            # segment_length_samples = int(self.segment_length * sr)
            # num_segments = int(np.ceil(signal.shape[1] / float(segment_length_samples)))

            features = []
            total_samples = signal.shape[1]

            # Calculate the number of segments based on the segment length and audio length
            segment_samples = int(self.segment_length * sr)
            num_segments = int(np.ceil(total_samples / float(segment_samples)))

            if oversample:
                num_segments *= 2

            if self.verbose:
                print(
                    f"{file_name}(features):\n  Total samples: {total_samples}, Segment samples: {segment_samples}, Num segments: {num_segments}"
                )

            os.makedirs(self.output_dir, exist_ok=True)

            for i in range(num_segments):
                # start_frame = i * segment_samples
                # end_frame = min(signal.shape[1], (i + 1) * segment_samples)

                if oversample:
                    start_frame = i * np.floor(segment_samples / 2.0)
                    end_frame = min(total_samples, start_frame + segment_samples)
                else:
                    start_frame = i * segment_samples
                    end_frame = min(total_samples, (i + 1) * segment_samples)

                start_frame = int(start_frame)
                end_frame = int(end_frame)

                segment = signal[:, start_frame:end_frame].to("cpu")

                # Can pad or trim the feature array to a fixed length or simply ignore it
                if segment.shape[1] < segment_samples:
                    continue  # Ignore segments that are shorter than the desired length

                # Extract audio for this segment
                y_segment = segment.numpy().squeeze()

                chroma_stft = np.mean(librosa.feature.chroma_stft(y=y_segment, sr=sr))
                rms = np.mean(librosa.feature.rms(y=y_segment))
                spec_cent = np.mean(
                    librosa.feature.spectral_centroid(y=y_segment, sr=sr)
                )
                spec_bw = np.mean(
                    librosa.feature.spectral_bandwidth(y=y_segment, sr=sr)
                )
                rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_segment, sr=sr))
                zcr = np.mean(librosa.feature.zero_crossing_rate(y_segment))
                mfccs = librosa.feature.mfcc(y=y_segment, sr=sr)
                mfccs_mean = np.mean(mfccs, axis=1)

                features.append(
                    [
                        chroma_stft,
                        rms,
                        spec_cent,
                        spec_bw,
                        rolloff,
                        zcr,
                        *mfccs_mean,
                        file_name,
                    ]
                )

            return features
        except Exception as e:
            print(f"Error processing segment of {file_name}: {e}")
            return None

    def create_dataset_and_images(self):
        labels = ["FAKE", "REAL"]
        feature_list = []

        for label in labels:
            print(f"Processing {label} files...")
            files = os.listdir(os.path.join(self.input_dir, label))
            for file in tqdm.tqdm(files, desc=f"{label} Files Progress"):
                file_path = os.path.join(self.input_dir, label, file)
                file_name = os.path.splitext(file)[0]
                signal, sr = self._load_and_preprocess(file_path)
                if signal is not None:
                    signal = signal / (2**15)

                    single_oversample = (
                        self.oversample
                        and (label == "REAL")
                        and (file_name not in self.oversample_ignore)
                    )

                    self._split_and_convert_to_melgram(
                        signal, sr, file_name, label, oversample=single_oversample
                    )

                    file_features = self._extract_features(
                        signal, sr, file_name, oversample=single_oversample
                    )
                    if file_features:
                        for segment_features in file_features:
                            feature_list.append(segment_features + [label])

        columns = (
            [
                "chroma_stft",
                "rms",
                "spectral_centroid",
                "spectral_bandwidth",
                "rolloff",
                "zero_crossing_rate",
            ]
            + [f"mfcc{i}" for i in range(1, 21)]
            + ["original_sample", "LABEL"]
        )
        # Save results to a CSV file
        features_df = pd.DataFrame(feature_list, columns=columns)
        features_df.to_csv(self.audio_features_filename, index=False)

        images_df = pd.DataFrame(self.image_annotations)
        images_df.to_csv(self.image_annotation_filename, index=False)

        return features_df, images_df
