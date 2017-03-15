import pickle
import os
import numpy as np
from nmf_train import nmf_train
from globals import *
from combine_sounds import combine_sounds

import librosa


root = "dev_dataset/dcase2016_task2_train"
effect_list = [name for name in os.listdir(root)]

trained_matrix = np.random.rand(
    int(WINDOW_LENGTH / 2 + 1), COMPONENTS_PER_EFFECT * (len(effect_list) + 1))

# Training on sounds in training_sounds
root = "training_sounds"
list = [os.listdir(root)[1]] + [os.listdir(root)[5]] + [os.listdir(root)[9]]
print(list)
trained_matrix = np.random.rand(
    int(WINDOW_LENGTH / 2 + 1), COMPONENTS_PER_EFFECT * (len(list) + 1))
for i in range(len(list)):
	x = list[i]
	path = os.path.join(root, x)
	print(path)
	components, activations = nmf_train(path, COMPONENTS_PER_EFFECT)

	trained_matrix[:, i * COMPONENTS_PER_EFFECT:(i + 1) * COMPONENTS_PER_EFFECT] = components

colormap(trained_matrix)
f = open("trained_matrix_concatenated_sounds_small3.pkl", "wb")
pickle.dump(trained_matrix, f)
f.close()
