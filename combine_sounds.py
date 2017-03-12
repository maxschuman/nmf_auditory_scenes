import os
import wave

def combine_sounds(list_of_paths, out_path):
	"""
	Takes a list of file paths to audio files, concatenates them, and then writes the result to out_path
	"""
	data= []
	for l in list_of_paths:
	    w = wave.open(l, 'rb')
	    data.append( [w.getparams(), w.readframes(w.getnframes())] )
	    w.close()

	output = wave.open(out_path, 'wb')
	output.setparams(data[0][0])
	for i in range(len(list_of_paths)):
		output.writeframes(data[i][1])
	output.close()


# Code for making a set of training sounds using the first NUM_EXAMPLES examples of each sound type
# Saved in training_sounds under name of effect
# NUM_EXAMPLES = 3
# root = "dev_dataset/dcase2016_task2_train"
# effect_list = [name for name in os.listdir(root)]

# out_root = "training_sounds"

# for i in range(len(effect_list)):
#     e = effect_list[i]
#     base = os.path.join(root, e)
#     list_of_paths = []
#     for x in os.listdir(base)[0:NUM_EXAMPLES]:
#     	list_of_paths.append(os.path.join(base, x))

#     out_path = os.path.join(out_root, e) + '.wav'

#     combine_sounds(list_of_paths, out_path)

# create test file that has sounds we've trained on before, one of each
# root = "dev_dataset/dcase2016_task2_train"
# effect_list = [name for name in os.listdir(root)]

# out_root = "test_sounds"
# list_of_paths = []
# for i in range(len(effect_list)):
#     e = effect_list[i]
#     base = os.path.join(root, e)
    
#     list_of_paths.append(os.path.join(base, os.listdir(base)[0]))

# out_path = os.path.join(out_root, 'test_trained.wav')

# combine_sounds(list_of_paths, out_path)

# create a smaller test file
root = "dev_dataset/dcase2016_task2_train"
effect_list = [name for name in os.listdir(root)]

out_root = "test_sounds"
list_of_paths = []
for i in range(len(effect_list)):
    e = effect_list[i]
    base = os.path.join(root, e)
    
    list_of_paths.append(os.path.join(base, os.listdir(base)[5]))

out_path = os.path.join(out_root, 'test_not_trained.wav')

combine_sounds(list_of_paths, out_path)