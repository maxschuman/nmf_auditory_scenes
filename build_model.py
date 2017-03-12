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

# First option: choose one file of each sound type to use to train
# for i in range(len(effect_list)):
#     e = effect_list[i]
#     base = os.path.join(root, e)
#     path = os.path.join(base, [name for name in os.listdir(base)][0])

#     components, activations = nmf_train(path, COMPONENTS_PER_EFFECT)

#     trained_matrix[:, i * COMPONENTS_PER_EFFECT:(i + 1) * COMPONENTS_PER_EFFECT] = components

# Second option: train on all files for each sound and then take
# element-wise median for representation of each event
# for i in range(len(effect_list)):
# 	e = effect_list[i]
# 	base = os.path.join(root, e)
# 	file_list = os.listdir(base)
# 	num_files = len(file_list)
#
# 	training_results = np.empty(
# 		shape=(int(WINDOW_LENGTH / 2 + 1), COMPONENTS_PER_EFFECT, num_files))
#
# 	for j in range(num_files):
# 		# try:
# 			file_name = file_list[j]
# 			path = os.path.join(base, file_name)
# 			print(path)
# 			components, activations = nmf_train(path, COMPONENTS_PER_EFFECT)
# 			training_results[:, :, j] = components
# 		# except:
# 		# 	print(file_list[j])
# 		# 	continue
#
# 	trained_matrix[:, i * COMPONENTS_PER_EFFECT:(
# 		i + 1) * COMPONENTS_PER_EFFECT] = np.median(training_results, axis=2)

# Training on sounds in training_sounds
root = "training_sounds"
librosa.load('dev_dataset/dcase2016_task2_train/clearthroat/clearthroat002.wav')
list = os.listdir(root)
for i in range(len(list)):
	x = list[i]
	path = os.path.join(root, x)
	print(path)
	components, activations = nmf_train(path, COMPONENTS_PER_EFFECT)

	trained_matrix[:, i * COMPONENTS_PER_EFFECT:(i + 1) * COMPONENTS_PER_EFFECT] = components

colormap(trained_matrix)
f = open("trained_matrix_concatenated_sounds.pkl", "wb")
pickle.dump(trained_matrix, f)
f.close()
