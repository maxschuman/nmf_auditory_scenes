import matplotlib.pyplot as plt
from nmf_separate import nmf_separate
import numpy as np
from globals import *


def display_events(path):
	W, H = nmf_separate(path)
	num_effects = len(LIST_OF_EFFECTS)
	consolidated_H = np.empty(shape=(num_effects, H.shape[1]))
	
	for i in range(num_effects):
		consolidated_H[i,:] = np.sum(H[i * COMPONENTS_PER_EFFECT:(i+1) * COMPONENTS_PER_EFFECT, :], axis=0)

	# plot stuff
	plt.figure(figsize=(18,20))
	for j in range(num_effects):
		plt.subplot(4, 3, j + 1)
		plt.plot(consolidated_H[j, :])
		plt.title(LIST_OF_EFFECTS[j])

	plt.show()


# test
display_events('test_sounds/test_trained.wav')
