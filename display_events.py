import matplotlib.pyplot as plt
from nmf_separate import nmf_separate
import numpy as np
from globals import *


def display_events(path, list_of_effects = LIST_OF_EFFECTS, moving_interval=10, threshold=None):
	W, H = nmf_separate(path)
	num_effects = len(list_of_effects)
	consolidated_H = np.empty(shape=(num_effects, H.shape[1] - moving_interval + 1))
	
	for i in range(num_effects):
		row = np.sum(H[i * COMPONENTS_PER_EFFECT:(i+1) * COMPONENTS_PER_EFFECT, :], axis=0)
		# try the moving average of moving_interval values instead of just the sum
		consolidated_H[i,:] = np.convolve(row, np.ones((moving_interval,))/moving_interval, mode='valid')
		consolidated_H[i,:] = consolidated_H[i,:] / max(consolidated_H[i,:])

		# binarize using threshold, if it is provided
		if threshold is not None:
			consolidated_H[i,:] = np.where(consolidated_H[i,:] > threshold, 1, 0)

	# plot stuff
	plt.figure(figsize=(18,20))
	for j in range(num_effects):
		plt.subplot(4,3, j + 1)
		plt.plot(consolidated_H[j, :])
		plt.title(list_of_effects[j])

	plt.show()


# test
display_events('test_sounds/test_not_trained.wav', threshold=0.5)
# display_events('test_sounds/test_trained_small.wav', list_of_effects=['Clear Throat', 'Cough', 'Door Slam'])
