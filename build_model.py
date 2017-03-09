import pickle, os, numpy as np
from nmf_train import nmf_train
from globals import *



root = "dev_dataset/dcase2016_task2_train"
effect_list = [name for name in os.listdir(root)]

trained_matrix = np.empty(shape=(int(WINDOW_LENGTH / 2 + 1), COMPONENTS_PER_EFFECT * (len(effect_list) + 1)))

# First option: choose one file of each sound type to use to train
# for i in range(len(effect_list)):
# 	e = effect_list[i]
# 	base = os.path.join(root, e)
# 	path = os.path.join(base, [name for name in os.listdir(base)][0])
	
# 	components, activations = nmf_train(path, COMPONENTS_PER_EFFECT)

# 	trained_matrix[:,i*COMPONENTS_PER_EFFECT:(i+1)*COMPONENTS_PER_EFFECT] = components

# Second option: train on all files for each sound and then take element-wise median for representation of each event
for i in range(len(effect_list)):
	e = effect_list[i]
	base = os.path.join(root, e)
	file_list = os.listdir(base)
	num_files = len(file_list)
	
	training_results = np.empty(shape=(int(WINDOW_LENGTH / 2 + 1), COMPONENTS_PER_EFFECT, num_files))
	
	for j in range(num_files):
		file_name = file_list[j]
		path = os.path.join(base, file_name)
		print(path)
		components, activations = nmf_train(path, COMPONENTS_PER_EFFECT)
		training_results[:,:,j] = components

	trained_matrix[:,i*COMPONENTS_PER_EFFECT:(i+1)*COMPONENTS_PER_EFFECT] = np.median(training_results, axis=2)

print(trained_matrix)
f = open("trained_matrix_single_training.pkl", "wb")
pickle.dump(trained_matrix, f)
f.close()
