import numpy as np
import librosa
import pickle
from globals import *
import sklearn
import matplotlib.pyplot as plt
# import sed_vis

test_path = 'dev_dataset/dcase2016_task2_dev/sound/dev_1_ebr_-6_nec_1_poly_0.wav'


def colormap(X):
	"""
	Displays np array X on a colormap
	"""
	fig = plt.figure(figsize=(8, 20))
	plt.pcolor(X)
	plt.colorbar()
	# plt.axes().set_aspect('equal', 'datalim')
	plt.show()


def nmf_separate(path):
	"""
	Function to use nmf to do component separation on file at path
	"""
	try:
		dd
		with open('test_matrix.pkl', 'rb') as infile:
			W, H = pickle.load(infile)
	except:
		with open('trained_matrix_single_training.pkl', 'rb') as infile:
			trained_components = pickle.load(infile)
		colormap(trained_components)

		audio, sr = librosa.load(path)
		s = librosa.stft(audio, WINDOW_LENGTH, HOP_LENGTH)
		s = np.abs(s)
		W, H, n_iter = sklearn.decomposition.non_negative_factorization(s.T, W=np.zeros((s.shape[1], COMPONENTS_PER_EFFECT * 12)), H=trained_components.T, n_components=COMPONENTS_PER_EFFECT * 12, init='custom', update_H=False)
		print("{0} iterations".format(n_iter))
		threshold = 0.02
		W = np.where(W>threshold, 1, 0)
		colormap(H.T)
		colormap(W.T)
		# transformer = sklearn.decomposition.NMF(n_components=96, init='custom', max_iter=1)
		# W = transformer.fit_transform(
		# 	s, W=trained_components, H=np.random.rand(96, s.shape[1]))
		# H = transformer.components_
		with open('test_matrix.pkl', 'wb') as outfile:
			pickle.dump((H.T, W.T), outfile)

	# for j in range(100):
	# 	print(H[:8, j])
	# 	# for i in range(H.shape[0] - 8):
	# 	# 	if H[i, j] > 1:
	# 	# 		print(j)
	# 	# 		break
	# colormap(W)
	# colormap(H)

nmf_separate(test_path)


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
