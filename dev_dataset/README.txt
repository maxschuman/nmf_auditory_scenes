IEEE AASP Challenge: Detection and Classification of Acoustic Scenes and Events 2016
http://www.cs.tut.fi/sgn/arg/dcase2016/

Task 2 - Synthetic audio sound event detection (SASED)
http://www.cs.tut.fi/sgn/arg/dcase2016/task-synthetic-sound-event-detection

Credits: Grégoire Lafay (IRCCYN, Ecole Cntrale de Nantes, France)

Contact: dcase-discussions@googlegroups.com

--

This README file documents the training and development data of the Synthetic Audio Sound Event Detection (SASED) task.

The training dataset is contained in the folder dcase2016_task2_train/
The development dataset is contained in the file dcase2016_task2_dev/

--

The training dataset is composed of mono recordings of isolated acoustic events typically found in an office environment. 11 classes are available:

* clearthroat (clearing throat)
* cough
* doorslam (slamming door)
* drawer
* keyboard (keyboard clicks)
* keysDrop (keys dropped on desk)
* knock (knocking on door)
* laughter
* pageturn (paper page turning)
* phone (vintage phone ringing)
* speech

Each class is represented by 20 recordings. Files are named according to the class name, i.e. classXXX.wav where XXX is a three-digit, unique and non-consecutive number. The audio format of the recordings is raw PCM, sampled at 44100 Hz, 16 bit (CD quality). All samples are normalized to -12 dBFS (true peak).

--

The development dataset has 18 simulated acoustic scenes built using the samples of the training dataset. The simulation process is controlled by three parameters:

* ebr: Event to Background Ratio
* nec: Number of active Events per Class
* poly: boolean indicating if the scene is polyphonic (poly=1; events may overlap each others), or monophonic (poly=0; events are not allowed to overlap each others)

Thus, each file name (.txt or .wav) is as follows: 'dev_1_ebr_XX_nec_Y_poly_Z', where XX={-6,0,6}, Z={0,1}, and Y={1,2,3} if Z=0 or Y={3,4,5} if Z=1.

Thus, the Event to Background Ratio varies from -12 to 0 dB, the scene can be monophonic or polyphonic. In the first case, the number of event varies from 1 to 3 and in the second from 3 to 5.
Note: the sound sources for the training and development sets are the same.

--

The folder 'annotation' in the development dataset contains the annotations for the aforementioned acoustic scenes as a plain ASCII file. Each .txt file contains information about the onset, offset, and event class for each event in the scene, separated by a tab, and is structured as follows:

onset1    offset1    EventID1
onset2    offset2    EventID2
...

where 'onset' refers to the onset time in seconds, 'offset' refers to the offset time in seconds since the beginning of the scene, and EventID is a string corresponding to the event class as shown in the list above.

--

Folder 'sound' in the development dataset contains the mono signals of the simulated acoustic scenes. The audio format of the simulated scenes is: raw PCM sampled at 44100 Hz, 16 bits (CD quality).

