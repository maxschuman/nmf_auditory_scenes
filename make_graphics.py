from display_events import *

# display_events('test_sounds/test_trained_small.wav', 'trained_matrix_concatenated_sounds_small.pkl', (3, 1), 'graphics/small_test_events_None.png', list_of_effects=['Clear Throat', 'Cough', 'Door Slam'])

display_events('test_sounds/test_not_trained_small3.wav', 'trained_matrix_concatenated_sounds_small3.pkl', (3, 1), 'graphics/small_test3_events_not_trained_0.8.png', list_of_effects=['Cough', 'Key Drop', 'Phone'], threshold=0.8)