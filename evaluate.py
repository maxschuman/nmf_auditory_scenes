import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

import librosa.display
from display_events import display_events
from globals import *
from nmf_separate import nmf_separate

SR = 22050.0


def parse_annotations(path):
    result = []
    with open(path) as infile:
        for line in infile.readlines():
            start, end, event = line.strip().split()
            start, end = float(start), float(end)
            result.append([start, end, event])
    return result


def get_filepaths():
    annotations = []
    sounds = []
    for dirpath, dirnames, filenames in os.walk('./dev_dataset/dcase2016_task2_test_public/annotation'):
        for filename in filenames:
            annotations.append(os.path.join(dirpath, filename))

    for dirpath, dirnames, filenames in os.walk('./dev_dataset/dcase2016_task2_test_public/sound'):
        for filename in filenames:
            sounds.append(os.path.join(dirpath, filename))
    return annotations, sounds

annotations, sounds = get_filepaths()
errors = []
print('{} cases'.format(len(annotations)))
with open('new_result.txt', 'w') as outfile:
    total_correct = 0
    total_events = 0
    for anno in range(len(annotations)):
        test_sound = sounds[anno]
        test_annotation = parse_annotations(annotations[anno])

        consolidated_H = display_events(
            test_sound, 'trained_matrix_concatenated_sounds.pkl', None, None, threshold=0.7, display=False)

        onsets = []
        for j in range(len(LIST_OF_EFFECTS)):
            i = 0
            while i < len(consolidated_H[j, :]):
                if consolidated_H[j, i]:
                    counter = 17
                    k = i + 1
                    while k < len(consolidated_H[j, :]):
                        if consolidated_H[j, k]:
                            counter = 17
                        elif counter == 0:
                            break
                        k += 1
                        counter -= 1
                    onsets.append([i, j, None])
                    i = k + 1
                else:
                    i += 1
        onsets.sort(key=lambda x: x[0])

        # print('clustered')

        correct, D = 0, 0
        for a in test_annotation:
            onset, offset, ev = a
            start_f, end_f = map(lambda time: int(
                SR * time / HOP_LENGTH), (onset - 0.200, onset + 0.200))
            ev_index = LIST_OF_EFFECTS.index(ev)

            D_once = False
            correct_once = False
            for event in onsets:
                onset, ev_id, flag = event
                if onset >= start_f and onset <= end_f:
                    D_once = True
                    if ev_id != ev_index:
                        event[2] = 'S'
                    else:
                        correct_once = True
                elif not flag:
                    event[2] = 'I'
            if not D_once:
                D += 1
            if correct_once:
                correct += 1

        S, I = 0, 0
        for event in onsets:
            if event[2] == 'S':
                S += 1
            elif event[2] == 'I':
                I += 1

        # res = 'Case {0}'.format(i + 1) + '\n' + 'Correctly identified {0} out of {1} annotations'.format(correct, len(test_annotation)) + '\n' + 'False activation rate - {0}'.format(round(np.sum(consolidated_H) / consolidated_H.size, 6))
        res = 'Case {}'.format(anno) + '\n' + 'Error Rate is {}'.format(round((S + I + D) / len(test_annotation), 5)) + '\n' + 'True Positive Rate is {}'.format(
            round(correct / len(test_annotation), 5)) + '\n' + 'S: {0}, I: {1}, D: {2}, correct: {3}, N: {4}'.format(S, I, D, correct, len(test_annotation))
        errors.append((S + I + D) / len(test_annotation))
        print(res)
        outfile.write(res + '\n')
        total_correct += correct
        total_events += len(test_annotation)
    print(total_correct, total_events, total_correct / total_events)
    print(sum(errors) / len(errors))
