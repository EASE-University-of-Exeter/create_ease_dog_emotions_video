
Author:  Steve North

License: AGPLv3 or later

License URI: http://www.gnu.org/licenses/agpl-3.0.en.html

Can: Commercial Use, Modify, Distribute, Place Warranty

Can't: Sublicence, Hold Liable

Must: Include Copyright, Include License, State Changes, Disclose Source

Copyright (c) 2019, The University of Exeter


create_ease_dog_emotions_video.py

Estimating Dog Emotions (particuarly for street dogs)

This uses a CSV output file from a DeepLabCut neural network's analysis of a video file (usually in the MP4 format).

Using a network trained to detect the facial landmarks in dogs.

This utility uses the canine facial landmarks to estimate expressions and emotions, based on existing taxonomy literature.

The distances between facial landmarks (forming a facial mesh) are used to predict specific expressions (brow raise, lip pull etc.), that are connected with specific emotions.

The resulting emotion charts are output to an MP4 video, that may be run in sync with the original video, as analysed by DeepLabCut 

(see: example_DeepLabCutandDogEmotions_side-by-side.mp4).

This code expects to find specific bodypart labels (facial landmarks), as column names in the input CSV file (see: data.csv):

mouth_corner_left

mouth_corner_right

inner_brow_left

inner_brow_right

upper_lip

lower_lip

nose_corner_left

nose_corner_right

pinna_ear_left

pinna_ear_right

The DeepLabCut neural network (also on GitHub) was trained using videography of street dogs from the University Of Exeter, UK, EASE Anthrozoology research group's 'Tails from the Streets' project (Using trans-species ethnography to document, understand and help mitigate the ‘stray dog problem’ in Europe:

https://socialsciences.exeter.ac.uk/ease/research/tailsfromthestreets/ ) and also example video clips of Facial Action Units and Action Descriptors from the Dog Facial Action Coding System (DogFACS):

Waller, B. M., Peirce, K., Caeiro, C. C., Scheider, L., Burrows, A. M., McCune, S. & Kaminski, J. 2013. Paedomorphic Facial Expressions Give Dogs a Selective Advantage. PLOS ONE, 8, 12, p.e82686.  http://dx.doi.org/10.1371/journal.pone.0082686. 

Waller, B. M. 2017. Dog Facial Action Coding System (DogFACS) [Online]. Available: http://dogfacs.com [Accessed 21 July 2017].

The network might require further raining (with as few as 10 frames), when encountering novel dog video.

What happens:
1. OPEN THE CSV FILE
2. MAP THE DATA TO A PANDAS DATAFRAME WITH NAMED COLUMNS
3. CALC THE DISTANCES BETWEEN SPECIFIC FACIAL LANDMARK LABELS AS ESTIMATED BY DEEPLABCUT NETWORK
4. NORMALISE DATA BY REMOVING OUTLIERS	
5. CALC AVERAGE DISTANCES (THIS VIDEO) FOR DISTANCE LINES BETWEEN SPECIFIC FACIAL LANDMARK LABELS	
6. SPECIFY THE FORMULAS FOR CALCULATING THE EMOTION CHART LINES
7. PLOT THE CHART LINES AND SAVE OUTPUTS: VIDEO, STILL IMAGE, FRAMES, CSV OUTPUT ETC.

Useage:

Ipython create_ease_dog_emotions_video.py data.csv

So, for example...

ipython create_growing_line_plot_video.py C:\Users\steve\Documents\GitHub\create_ease_dog_emotions_video\data.csv

Ipython C:\Users\steve\Documents\GitHub\create_ease_dog_emotions_video\create_ease_dog_emotions_video.py C:\Users\steve\Documents\GitHub\create_ease_dog_emotions_video\data.csv

ipython C:\Users\steve\Documents\GitHub\create_ease_dog_emotions_video\create_ease_dog_emotions_video.py "C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\Tales Vids\12-094136-services\12-094136-services_Segment_0_x264DeepCut_resnet50_ease_dog_emotions_1Mar13shuffle1_34000.csv"

Note: paths with spaces require speechmarks

Issue:

The output MP4 video for the line charts is at 25 FPS. If the original video file used as an input for DeepLabCut is not at 25 fps, then there will be a mismatch of durations.
The CSV file only lists the total number of frames (the same as the rows in the CSV file), but does not include the original rate. 

Workaround is to change the framerate of the output video to 25 fps, using keyframes to manually adjust how many of the available frames are used within a fixed duration (the length of the original video file).


