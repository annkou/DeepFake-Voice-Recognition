import os
import librosa
import pandas as pd


def extract_audio_metadata(directory):
    """
    Extracts metadata from audio files in a directory.
    Args:
        directory (str): Path to directory containing audio files.
    Returns:
        pd.DataFrame: DataFrame containing metadata of audio files.
    """
    metadata = []
    for dirpath, _, filenames in os.walk(directory):
        for filename in filenames:
            label = dirpath.split("/")[-1]
            if filename.endswith(".wav"):
                file_path = os.path.join(dirpath, filename)
                signal, sr = librosa.load(file_path, sr=None, mono=False)
                duration = librosa.get_duration(y=signal, sr=sr)
                num_channels = signal.shape[0] if len(signal.shape) > 1 else 1
                metadata.append(
                    {
                        "file_path": file_path,
                        "filename": filename,
                        "label": label,
                        "sample_rate": sr,
                        "num_channels": num_channels,
                        "duration": duration,
                    }
                )
    return pd.DataFrame(metadata)
