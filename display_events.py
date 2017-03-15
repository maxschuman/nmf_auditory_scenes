import matplotlib.pyplot as plt
from nmf_separate import nmf_separate
import numpy as np
from globals import *
import librosa.display


def display_events(audio_path, trained_path, figure_dimensions, out_path, list_of_effects=LIST_OF_EFFECTS, moving_interval=10, threshold=None):
    W, H = nmf_separate(audio_path, trained_path)
    num_effects = len(list_of_effects)
    consolidated_H = np.empty(shape=(num_effects, H.shape[1] - moving_interval + 1))

    for i in range(num_effects):
        row = np.sum(H[i * COMPONENTS_PER_EFFECT:(i + 1) * COMPONENTS_PER_EFFECT, :], axis=0)
        # try the moving average of moving_interval values instead of just the sum
        consolidated_H[i, :] = np.convolve(row, np.ones((moving_interval,)) / moving_interval, mode='valid')
        consolidated_H[i, :] = consolidated_H[i, :] / max(consolidated_H[i, :])

        # binarize using threshold, if it is provided
        if threshold is not None:
            consolidated_H[i, :] = np.where(consolidated_H[i, :] > threshold, 1, 0)

    # find stft and display spectrogram
    audio, sr = librosa.load(audio_path)
    s = librosa.stft(audio, WINDOW_LENGTH, HOP_LENGTH)
    s = np.abs(s)
    librosa.display.specshow(s, x_axis="time", y_axis="linear", sr=sr, hop_length=HOP_LENGTH, cmap='hot')

    # plot stuff
    fig = plt.figure(figsize=(10, 10))
    subplot_rows = figure_dimensions[0]
    subplot_columns = figure_dimensions[1]

    x = HOP_LENGTH * np.arange(0, consolidated_H.shape[1]) / sr
    for j in range(num_effects):
        plt.subplot(subplot_rows, subplot_columns, j + 1)
        plt.plot(x, consolidated_H[j, :])
        plt.ylabel(list_of_effects[j] + " level")
    plt.xlabel("Time (s)")
    fig.savefig(out_path, dpi=100)
    plt.show()


# test
# display_events('test_sounds/test_trained.wav')

