import numpy as np, librosa
from globals import *

# default values for our project
# Also it looks like the sample rate for the sound effects is 22050

def nmf_train(path, num_components):
	"""
	Function to generate nmf components on one sound file.
	"""
	audio, sr = librosa.load(path)
	s = librosa.stft(audio, WINDOW_LENGTH, HOP_LENGTH)
	s = np.abs(s)
	components, activations = librosa.decompose.decompose(s, num_components)
	return components, activations

# nmf_train("dev_dataset/dcase2016_task2_dev/sound/dev_1_ebr_-6_nec_1_poly_0.wav", 8)

def nmf_recompose(W, H, sr, path=False):
	"""
	Function to recombine and potentially save audio results of nmf process
	"""
	s = np.matmul(W, H)

	if path:
		audio = reconstruct_from_mag(s, WINDOW_LENGTH, HOP_LENGTH)
		audio = audio / max(audio)
		librosa.output.write_wav(path, audio, sr)

	return s

def reconstruct_from_mag(S, window_length, hop_length, num_its=8):
	"""
	Function to reconstruct audio signal from magnitude spectrogram, with no information about phase
	"""
	length = (S.shape[1] - 1) * hop_length
	sig = np.random.rand(length)
	for i in range(num_its):
		# compute angle of sig
		sig = librosa.istft(S * np.exp(1j * np.angle(librosa.stft(sig, window_length, hop_length))), hop_length, window_length)

	return sig
