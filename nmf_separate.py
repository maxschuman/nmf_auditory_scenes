import numpy as np
import librosa
import pickle
from globals import *
import sklearn
import matplotlib.pyplot as plt


def nmf_separate(audio_path, trained_path):
    """
    Function to use nmf to do component separation on file at path
    """
    # try:
    #     with open('test_matrix.pkl', 'rb') as infile:
    #         W, H = pickle.load(infile)
    #     return W, H
    # except:
    with open(trained_path, 'rb') as infile:
        trained_components = pickle.load(infile)
    # colormap(trained_components)

    audio, sr = librosa.load(audio_path)
    s = librosa.stft(audio, WINDOW_LENGTH, HOP_LENGTH)
    s = np.abs(s)
    W, H, n_iter = sklearn.decomposition.non_negative_factorization(s.T, W=np.zeros((s.shape[1], trained_components.shape[0])), H=trained_components.T, n_components=trained_components.shape[1], init='custom', update_H=False)
    # colormap(H.T)
    # colormap(W.T)

    with open('test_matrix.pkl', 'wb') as outfile:
        pickle.dump((H.T, W.T), outfile)

    return H.T, W.T
