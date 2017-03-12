import numpy as np
import librosa
import pickle
from globals import *
import sklearn
import matplotlib.pyplot as plt
# import sed_vis

def nmf_separate(path):
	"""
	Function to use nmf to do component separation on file at path
	"""
	try:
		with open('test_matrix.pkl', 'rb') as infile:
			W, H = pickle.load(infile)
		return W, H
	except:
		with open('trained_matrix_concatenated_sounds.pkl', 'rb') as infile:
			trained_components = pickle.load(infile)
		colormap(trained_components)

		audio, sr = librosa.load(path)
		s = librosa.stft(audio, WINDOW_LENGTH, HOP_LENGTH)
		s = np.abs(s)
		W, H, n_iter = sklearn.decomposition.non_negative_factorization(s.T, W=np.zeros((s.shape[1], COMPONENTS_PER_EFFECT * 12)), H=trained_components.T, n_components=COMPONENTS_PER_EFFECT * 12, init='custom', update_H=False)
		print("{0} iterations".format(n_iter))
		colormap(H.T)
		colormap(W.T)
		
		with open('test_matrix.pkl', 'wb') as outfile:
			pickle.dump((H.T, W.T), outfile)

		return H.T, W.T


def test_sed_vis():
	# Load audio signal first
	audio, fs = sed_vis.io.load_audio(test_path)

	# Load event lists
	reference_event_list = sed_vis.io.load_event_list('tests/data/a001.ann')
	estimated_event_list = sed_vis.io.load_event_list(
		'tests/data/a001_system_output.ann')
	event_lists = {'reference': reference_event_list,
				   'estimated': estimated_event_list}

	# Visualize the data
	vis = sed_vis.visualization.EventListVisualizer(
		event_lists=event_lists, audio_signal=audio, sampling_rate=fs)
	vis.show()
