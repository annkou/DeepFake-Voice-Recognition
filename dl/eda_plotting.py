import IPython
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import matplotlib.pyplot as plt
import librosa
import numpy as np
import torchaudio


def plot_metadata_distribution(metadata_df):
    """
    Plots distribution of metadata from audio files.
    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata of audio files.
    """
    class_distr = metadata_df.groupby("label").sum()

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Sample Rate Distribution (Bar Chart)",
            "Number of Channels Distribution (Bar Chart)",
            "Duration Distribution (Bar Chart)",
            "",
            "Class Distribution (Bar Chart)",
            "Class Distribution (Pie Chart)",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "bar"}, {}],
            [{"type": "bar"}, {"type": "pie"}],
        ],
        column_widths=[0.5, 0.5],
    )

    # Plot sample rate distribution
    fig.add_trace(
        go.Bar(
            x=metadata_df["sample_rate"].value_counts().index,
            y=metadata_df["sample_rate"].value_counts(),
            name="Sample Rate",
        ),
        row=1,
        col=1,
    )

    # Plot number of channels distribution
    channels_counts = (
        metadata_df["num_channels"].value_counts().reindex([1, 2], fill_value=0)
    )
    fig.add_trace(
        go.Bar(x=channels_counts.index, y=channels_counts, name="Number of Channels"),
        row=1,
        col=2,
    )

    # Plot duration distribution
    fig.add_trace(
        go.Bar(
            x=metadata_df["duration"].round().value_counts().sort_index().index,
            y=metadata_df["duration"].round().value_counts().sort_index(),
            name="Duration",
        ),
        row=2,
        col=1,
    )

    # Plot class distribution - bar chart
    bar_fig = px.bar(
        class_distr,
        x=class_distr.index,
        y="duration",
        labels={"duration": "Length (seconds)"},
    )
    for trace in bar_fig["data"]:
        fig.add_trace(trace, row=3, col=1)

    # Plot class distribution - pie chart
    pie_fig = px.pie(class_distr, values="duration", names=class_distr.index)
    for trace in pie_fig["data"]:
        fig.add_trace(trace, row=3, col=2)

    # Update layout for the entire figure
    fig.update_layout(
        height=1200,
        width=1200,
        title_text="Audio Metadata Distribution",
        showlegend=False,
    )

    fig.update_xaxes(title_text="Sample Rate", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)

    fig.update_xaxes(title_text="Number of Channels", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)

    fig.update_xaxes(title_text="Duration (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    fig.update_xaxes(title_text="Class", row=3, col=1)
    fig.update_yaxes(title_text="Length (seconds)", row=3, col=1)

    fig.show()


def plot_file_durations(metadata_df):
    """
    Plots the duration of audio files.
    Args:
        metadata_df (pd.DataFrame): DataFrame containing metadata of audio files.
    """
    class_distr = metadata_df.groupby(["filename"])["duration"].mean()
    # do it using plotly
    fig = px.bar(
        class_distr,
        x=class_distr.index,
        y="duration",
        title="Files duration",
        labels={"length (in s)": "Length (seconds)"},
    )
    # rotate x-axis labels
    fig.update_xaxes(tickangle=-45)
    # sort values
    fig.update_xaxes(categoryorder="total descending")
    # fix size of the plot
    fig.update_layout(
        autosize=True,
        width=1200,
        height=700,
        margin=dict(l=50, r=50, b=100, t=100, pad=4),
    )
    fig.show()


def plot_waveform(audio_file_path):
    """
    Plots the waveform of an audio file.
    """
    # Load the audio file
    signal, sr = librosa.load(audio_file_path, sr=None, mono=False)

    # Check if the audio is stereo
    is_stereo = len(signal.shape) > 1

    if is_stereo:
        # Separate the channels
        left_channel = signal[0, :]
        right_channel = signal[1, :]

        # Plot the waveforms
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        librosa.display.waveshow(left_channel, sr=sr)
        plt.title("Left Channel Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.subplot(1, 2, 2)
        librosa.display.waveshow(right_channel, sr=sr)
        plt.title("Right Channel Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()
    else:
        # Plot the waveform for the mono channel
        plt.figure(figsize=(15, 6))
        librosa.display.waveshow(signal, sr=sr)
        plt.title("Mono Channel Waveform")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

        plt.tight_layout()
        plt.show()


def plot_fft(audio_file_path):
    """
    Plots the FFT of an audio file.
    """
    # Load the audio file
    signal, sr = librosa.load(audio_file_path, sr=None, mono=False)

    # Check if the audio is stereo
    is_stereo = len(signal.shape) > 1

    if is_stereo:
        # Separate the channels
        left_channel = signal[0, :]
        right_channel = signal[1, :]

        # Compute the FFT for both channels
        fft_left = np.fft.fft(left_channel)
        fft_right = np.fft.fft(right_channel)

        # Compute the frequency axis
        freqs = np.fft.fftfreq(len(fft_left), 1 / sr)

        # Plot the FFT results
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        plt.plot(freqs[: len(freqs) // 2], np.abs(fft_left)[: len(freqs) // 2])
        plt.title("Left Channel FFT")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        plt.subplot(1, 2, 2)
        plt.plot(freqs[: len(freqs) // 2], np.abs(fft_right)[: len(freqs) // 2])
        plt.title("Right Channel FFT")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        plt.show()
    else:
        # Compute the FFT for the mono channel
        fft_mono = np.fft.fft(signal)

        # Compute the frequency axis
        freqs = np.fft.fftfreq(len(fft_mono), 1 / sr)

        # Plot the FFT results
        plt.figure(figsize=(15, 6))
        plt.plot(freqs[: len(freqs) // 2], np.abs(fft_mono)[: len(freqs) // 2])
        plt.title("Mono Channel FFT")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")

        plt.tight_layout()
        plt.show()


def plot_spectrogram(audio_file_path):
    """
    Plots the spectrogram of an audio file.
    """
    # Load the audio file
    signal, sr = librosa.load(audio_file_path, sr=None, mono=False)

    # Check if the audio is stereo
    is_stereo = len(signal.shape) > 1

    if is_stereo:
        # Separate the channels
        left_channel = signal[0, :]
        right_channel = signal[1, :]

        # Compute the spectrogram for both channels
        S_left = librosa.feature.melspectrogram(y=left_channel, sr=sr)
        S_dB_left = librosa.power_to_db(S_left, ref=np.max)

        S_right = librosa.feature.melspectrogram(y=right_channel, sr=sr)
        S_dB_right = librosa.power_to_db(S_right, ref=np.max)

        # Plot the spectrograms
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        librosa.display.specshow(S_dB_left, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Left Channel Spectrogram")

        plt.subplot(1, 2, 2)
        librosa.display.specshow(S_dB_right, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Right Channel Spectrogram")

        plt.tight_layout()
        plt.show()
    else:
        # Compute the spectrogram for the mono channel
        S_mono = librosa.feature.melspectrogram(y=signal, sr=sr)
        S_dB_mono = librosa.power_to_db(S_mono, ref=np.max)

        # Plot the spectrogram
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(S_dB_mono, sr=sr, x_axis="time", y_axis="mel")
        plt.colorbar(format="%+2.0f dB")
        plt.title("Mono Channel Spectrogram")

        plt.tight_layout()
        plt.show()


def plot_mfcc(audio_file_path):
    """
    Plots the MFCC of an audio file.
    """
    # Load the audio file
    signal, sr = librosa.load(audio_file_path, sr=None, mono=False)

    # Check if the audio is stereo
    is_stereo = len(signal.shape) > 1

    if is_stereo:
        # Separate the channels
        left_channel = signal[0, :]
        right_channel = signal[1, :]

        # Compute the MFCC for both channels
        mfcc_left = librosa.feature.mfcc(y=left_channel, sr=sr, n_mfcc=13)
        mfcc_right = librosa.feature.mfcc(y=right_channel, sr=sr, n_mfcc=13)

        # Plot the MFCCs
        plt.figure(figsize=(20, 5))

        plt.subplot(1, 2, 1)
        librosa.display.specshow(mfcc_left, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title("Left Channel MFCC")

        plt.subplot(1, 2, 2)
        librosa.display.specshow(mfcc_right, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title("Right Channel MFCC")

        plt.tight_layout()
        plt.show()
    else:
        # Compute the MFCC for the mono channel
        mfcc_mono = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=13)

        # Plot the MFCC
        plt.figure(figsize=(15, 6))
        librosa.display.specshow(mfcc_mono, sr=sr, x_axis="time")
        plt.colorbar()
        plt.title("Mono Channel MFCC")

        plt.tight_layout()
        plt.show()


def extract_audio_info(file_path, filename):
    """AUDIO INFO: \n
    Extracts information from an audio file. \n
    Args:
        file_path (str): Path to the audio file.
        filename (str): Name of the audio file.
    """

    signal, sr = librosa.load(file_path, sr=None, mono=False)
    info = torchaudio.info(file_path)
    print(f"Info about audio file \033[95m {filename} \033[0m are: \n{info}")
    print(
        f"The samping rate of the audio file is {sr} Hz\nThe audio contains a total of {info.num_channels} channels\nThe length of the audio file is {librosa.get_duration(y=signal, sr=sr)} seconds\nThe audio contains a total of {info.num_frames} frames\nThe signal contains a total of {info.num_frames * info.num_channels} samples"
    )
    print(
        f"If this value is greater than {info.num_frames} it is due to there being multiple channels in the audio file"
    )
    IPython.display.display(IPython.display.Audio(file_path))
    plot_waveform(file_path)
