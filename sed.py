import numpy as np, pickle, matplotlib.pyplot as plt
from nmf_train import *
from globals import *

def scene_activations(path, components, consolidate=np.median, threshold=None):
	"""
	Function that takes in an auditory scene and returns a matrix representing the activations for each sound type based on components.
	consolidate is a function that combines the rows corresponding to components of the same sound class into one value.
	"""
	c, activations = nmf_train(path, components.shape[1], component_start=components)

	num_effects = int(components.shape[1] / COMPONENTS_PER_EFFECT)
	consolidated_activations = np.empty(shape=(num_effects, activations.shape[1]))
	for i in range(num_effects):
		slice = activations[i*COMPONENTS_PER_EFFECT:(i+1)*COMPONENTS_PER_EFFECT, :]
		consolidated_activations[i, :] = consolidate(slice, axis=0)

	# normalize
	consolidated_activations = consolidated_activations / np.amax(consolidated_activations)

	# binarize matrix with threshold, if set
	if threshold is not None:
		consolidated_activations = np.where(consolidated_activations>threshold, 1, 0)
	return consolidated_activations

f = open('trained_matrix_iterative_training.pkl', 'rb')
c = pickle.load(f)
path = "dev_dataset/dcase2016_task2_dev/sound/dev_1_ebr_-6_nec_1_poly_0.wav"
activations = scene_activations(path, c)

plt.figure(figsize=(10,4))
plt.pcolor(activations)
plt.show()
