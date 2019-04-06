
'''
Author:  Steve North
Author URI: 
License: AGPLv3 or later
License URI: http://www.gnu.org/licenses/agpl-3.0.en.html
Can: Commercial Use, Modify, Distribute, Place Warranty
Can't: Sublicence, Hold Liable
Must: Include Copyright, Include License, State Changes, Disclose Source

Copyright (c) 2019, The University of Exeter
'''

create_ease_dog_emotions_video.py

This uses CSV output from DeepLabCut neural network's analysis of a video file.
Using a network trained to detect the facial landmarks in dogs.
This utility uses the canine facial landmarks to predict expressions and emotions, based on existing taxonomy literature.
The distances between facial landmarks are used to predict specific expressions (brow raise, lip pull etc.), that are connected with specific emotions.
The results emotion charts are output to an MP4 video, that may be run in sync with the original video, as analysed by DeepLabCut.

This code expects to find specific bodypart labels (facial landmarks), as column names in the input CSV file.

The network was trained using...

Might require retraining (with as few as 10 frames), when encountering novel dog video.

# What happens:
# 1. OPEN THE CSV FILE
# 2. MAP THE DATA TO A PANDAS DATAFRAME WITH NAMED COLUMNS
# 3. CALC THE DISTANCES BETWEEN SPECIFIC FACIAL LANDMARK LABELS AS ESTIMATED BY DEEPLABCUT NETWORK
# 4. NORMALISE DATA BY REMOVING OUTLIERS	
# 5. CALC AVERAGE DISTANCES (THIS VIDEO) FOR DISTANCE LINES BETWEEN SPECIFIC FACIAL LANDMARK LABELS	
# 6. SPECIFY THE FORMULAS FOR CALCULATING THE EMOTION CHART LINES
# 7. PLOT THE CHART LINES AND SAVE OUTPUTS: VIDEO, STILL IMAGE, FRAMES, CSV OUTPUT ETC.

Useage:

ipython create_growing_line_plot_video.py C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\create_ease_dog_emotions_video\data.csv

Ipython create_ease_dog_emotions_video.py data.csv

Ipython C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\create_ease_dog_emotions_video\create_ease_dog_emotions_video.py C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\create_ease_dog_emotions_video\data.csv

ipython C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\create_ease_dog_emotions_video\create_ease_dog_emotions_video.py "C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\Tales Vids\12-094136-services\12-094136-services_Segment_0_x264DeepCut_resnet50_ease_dog_emotions_1Mar13shuffle1_34000.csv"

Note: paths with spaces require speechmarks

ipython C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\create_ease_dog_emotions_video\create_ease_dog_emotions_video.py  "C:\Users\steve\Anaconda3\envs\ease_dog_emotions\dev\DogFAC vids\5\Clip 5DeepCut_resnet50_ease_dog_emotions_1Mar13shuffle1_38000.csv"

Issue:
The output MP4 video for the line charts is at 25 FPS. If the original video file used as an input for DeepLabCut is not at 25 fps, then there will be a mismatch of durations.
The CSV file only list the total number of frames (the same as the rows in the CSV file), but does not include the original rate. Workaround is to change the framerate of the output video to 25 fps.


