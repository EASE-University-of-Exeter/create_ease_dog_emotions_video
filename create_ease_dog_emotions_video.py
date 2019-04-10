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

# What happens:
# 1. OPEN THE CSV FILE
# 2. MAP THE DATA TO A PANDAS DATAFRAME WITH NAMED COLUMNS
# 3. CALC THE DISTANCES BETWEEN SPECIFIC FACIAL LANDMARK LABELS AS ESTIMATED BY DEEPLABCUT NETWORK
# 4. NORMALISE DATA BY REMOVING OUTLIERS	
# 5. CALC AVERAGE DISTANCES (THIS VIDEO) FOR DISTANCE LINES BETWEEN SPECIFIC FACIAL LANDMARK LABELS	
# 6. SPECIFY THE FORMULAS FOR CALCULATING THE EMOTION CHART LINES
# 7. PLOT THE CHART LINES AND SAVE OUTPUTS: VIDEO, STILL IMAGE, FRAMES, CSV OUTPUT ETC.


# libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import sys
from matplotlib.animation import FFMpegWriter
import math
from sklearn.preprocessing import MinMaxScaler

fn = sys.argv[1]

YpositionOfBottomChartLine = 30
YgapBetweenChartLines = 10

happy_line_default_y = YpositionOfBottomChartLine + (3* YgapBetweenChartLines)
scared_sad_default_y = YpositionOfBottomChartLine + (2* YgapBetweenChartLines)
angry_disgusted_default_y = YpositionOfBottomChartLine + YgapBetweenChartLines

suprised_default_y = YpositionOfBottomChartLine

threshold = 10
ears_apart_threshold = 5

#happy_chart_line_colour = '#8b02db'
#scared_sad_chart_line_colour = '#e5ed09'
#angry_disgusted_chart_line_colour = '#08ed18'
#suprised_chart_line_colour = '#f48e00'

happy_chart_line_colour = '#f20404'
scared_sad_chart_line_colour = '#26ce10'
angry_disgusted_chart_line_colour = '#1707f4'
suprised_chart_line_colour = '#f402d8'

numberOfParamsOutOfFiveRequiredTrueForHeadTiltConfirmation = 3 # x or more out of 5 required from: ears tilted, ears apart, brows tilted and mouth corners tilted

scaler = MinMaxScaler()

#if os.path.exists(fn):
    #print(os.path.basename(fn)) # just the filename without the path
    #print(fn)
    #print(os.path.dirname(fn))
    # file exists

############ 1. OPEN THE CSV FILE ##########################################	
	
csvFileDirName = os.path.dirname(fn)
csvFileNameWithoutExtension = os.path.basename(fn).split('.')[0] # get filename without extension
#print(csvFileDirName) 

############ 2. MAP THE DATA TO A PANDAS DATAFRAME WITH NAMED COLUMNS ##########################################	

# dataset
df = pd.read_csv(fn)  # store the csv file in dataframe df
#print(len(df.index))
df = df.iloc[2:len(df.index), np.r_[0,1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29]]  # subselect dataframe
df.columns = ['frame','mouth_corner_left_x','mouth_corner_left_y','mouth_corner_right_x','mouth_corner_right_y','inner_brow_left_x','inner_brow_left_y','inner_brow_right_x','inner_brow_right_y','upper_lip_x','upper_lip_y','lower_lip_x','lower_lip_y','nose_corner_left_x','nose_corner_left_y','nose_corner_right_x','nose_corner_right_y','pinna_ear_left_x','pinna_ear_left_y','pinna_ear_right_x','pinna_ear_right_y']
df = df.convert_objects(convert_numeric=True)

############ 3. CALC THE DISTANCES BETWEEN SPECIFIC FACIAL LANDMARK LABELS AS ESTIMATED BY DEEPLABCUT NETWORK ##########################################	

# Using Pythagoras' theorem (my maths teacher was right, I would need it!): the square of the hypotenuse is equal to the sum of the squares of the other two sides
# So, dist_between_any_two_facial_landmarks_as_estimated_by_neural_net = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
# This calculates the apparent 2D distance (nothing 3D going on here!) between pairs of significant canine facial landmarks, which change during DogFACS Facial Action Units and (hence) emotions 

calc_SEPARATION_mouth_corner_left_to_upper_lip = lambda row: math.sqrt( (row.mouth_corner_left_x - row.upper_lip_x)**2 + (row.mouth_corner_left_y - row.upper_lip_y)**2) # define a function for the new column

calc_SEPARATION_mouth_corner_right_to_upper_lip = lambda row: math.sqrt( (row.mouth_corner_right_x - row.upper_lip_x)**2 + (row.mouth_corner_right_y - row.upper_lip_y)**2) # define a function for the new column

calc_SEPARATION_pinna_ear_left_to_nose_corner_left = lambda row: math.sqrt( (row.pinna_ear_left_x - row.nose_corner_left_x)**2 + (row.pinna_ear_left_y - row.nose_corner_left_y)**2) # define a function for the new column

calc_SEPARATION_pinna_ear_right_to_nose_corner_right = lambda row: math.sqrt( (row.pinna_ear_right_x - row.nose_corner_left_x)**2 + (row.pinna_ear_right_y - row.nose_corner_left_y)**2) # define a function for the new column

calc_SEPARATION_pinna_ear_left_inner_brow_left = lambda row: math.sqrt( (row.pinna_ear_left_x - row.inner_brow_left_x)**2 + (row.pinna_ear_left_y - row.inner_brow_left_y)**2) # define a function for the new column

calc_SEPARATION_pinna_ear_right_inner_brow_right = lambda row: math.sqrt( (row.pinna_ear_right_x - row.inner_brow_right_x)**2 + (row.pinna_ear_right_y - row.inner_brow_right_y)**2) # define a function for the new column

calc_SEPARATION_upper_lip_to_lower_lip = lambda row: math.sqrt( (row.upper_lip_x - row.lower_lip_x)**2 + (row.upper_lip_y - row.lower_lip_y)**2) # define a function for the new column

calc_SEPARATION_pinna_ear_left_to_pinna_ear_right = lambda row: math.sqrt( (row.pinna_ear_left_x - row.pinna_ear_right_x)**2 + (row.pinna_ear_left_y - row.pinna_ear_right_y)**2) # define a function for the new column

calc_SEPARATION_pinna_ear_left_to_mouth_corner_left = lambda row: math.sqrt( (row.pinna_ear_left_x - row.mouth_corner_left_x)**2 + (row.pinna_ear_left_y - row.mouth_corner_left_y)**2) # define a function for the new column

calc_SEPARATION_pinna_ear_right_to_mouth_corner_right = lambda row: math.sqrt( (row.pinna_ear_right_x - row.mouth_corner_right_x)**2 + (row.pinna_ear_right_y - row.mouth_corner_right_y)**2) # define a function for the new column


data_SEPARATION_mouth_corner_left_to_upper_lip = df.apply(calc_SEPARATION_mouth_corner_left_to_upper_lip, axis=1) # get column data with an index
data_SEPARATION_mouth_corner_right_to_upper_lip = df.apply(calc_SEPARATION_mouth_corner_right_to_upper_lip, axis=1) # get column data with an index
data_SEPARATION_pinna_ear_left_to_nose_corner_left = df.apply(calc_SEPARATION_pinna_ear_left_to_nose_corner_left, axis=1) # get column data with an index
data_SEPARATION_pinna_ear_right_to_nose_corner_right = df.apply(calc_SEPARATION_pinna_ear_right_to_nose_corner_right, axis=1) # get column data with an index
data_SEPARATION_pinna_ear_left_inner_brow_left = df.apply(calc_SEPARATION_pinna_ear_left_inner_brow_left, axis=1) # get column data with an index
data_SEPARATION_pinna_ear_right_inner_brow_right = df.apply(calc_SEPARATION_pinna_ear_right_inner_brow_right, axis=1) # get column data with an index
data_SEPARATION_upper_lip_to_lower_lip = df.apply(calc_SEPARATION_upper_lip_to_lower_lip, axis=1) # get column data with an index
data_SEPARATION_pinna_ear_left_to_pinna_ear_right = df.apply(calc_SEPARATION_pinna_ear_left_to_pinna_ear_right, axis=1) # get column data with an index
data_SEPARATION_pinna_ear_left_to_mouth_corner_left = df.apply(calc_SEPARATION_pinna_ear_left_to_mouth_corner_left, axis=1) # get column data with an index
data_SEPARATION_pinna_ear_right_to_mouth_corner_right = df.apply(calc_SEPARATION_pinna_ear_right_to_mouth_corner_right, axis=1) # get column data with an index


df = df.assign(SEPARATION_mouth_corner_left_to_upper_lip=data_SEPARATION_mouth_corner_left_to_upper_lip.values) # assign values to column
df = df.assign(SEPARATION_mouth_corner_right_to_upper_lip=data_SEPARATION_mouth_corner_right_to_upper_lip.values) # assign values to column
df = df.assign(SEPARATION_pinna_ear_left_to_nose_corner_left=data_SEPARATION_pinna_ear_left_to_nose_corner_left.values) # assign values to column
df = df.assign(SEPARATION_pinna_ear_right_to_nose_corner_right=data_SEPARATION_pinna_ear_right_to_nose_corner_right.values) # assign values to column
df = df.assign(SEPARATION_pinna_ear_left_inner_brow_left=data_SEPARATION_pinna_ear_left_inner_brow_left.values) # assign values to column
df = df.assign(SEPARATION_pinna_ear_right_inner_brow_right=data_SEPARATION_pinna_ear_right_inner_brow_right.values) # assign values to column
df = df.assign(SEPARATION_upper_lip_to_lower_lip=data_SEPARATION_upper_lip_to_lower_lip.values) # assign values to column
df = df.assign(SEPARATION_pinna_ear_left_to_pinna_ear_right=data_SEPARATION_pinna_ear_left_to_pinna_ear_right.values) # assign values to column
df = df.assign(SEPARATION_pinna_ear_left_to_mouth_corner_left=data_SEPARATION_pinna_ear_left_to_mouth_corner_left.values) # assign values to column
df = df.assign(SEPARATION_pinna_ear_right_to_mouth_corner_right=data_SEPARATION_pinna_ear_right_to_mouth_corner_right.values) # assign values to column

############ 4. NORMALISE DATA BY REMOVING OUTLIERS ##########################################	

# Somewhere around here... need to identify outliers from the SEPARATION line columns, as there are skewing the means for these columns, making it difficult to spot real FX
# Outliers might be defined as such if they are more than 3 standard deviations away from the group mean?
# Can't delete outlier rows, as these are frames and will impact sync with corresponding video.
# But group mean must exclude outliers before it can be used to exclude outliers (!) How do you do that!!?
# This might help: https://stackoverflow.com/questions/27638743/pandas-replace-outliers-with-groupby-mean

#df.loc[df['SEPARATION_pinna_ear_left_to_pinna_ear_right'] > 1990, 'SEPARATION_pinna_ear_left_to_pinna_ear_right'] = 1

############ 5. CALC AVERAGE DISTANCES (THIS VIDEO) FOR DISTANCE LINES BETWEEN SPECIFIC FACIAL LANDMARK LABELS  ##########################################	

# Work out the averages for the lines linking the facial landmarks
average_for_this_video_SEPARATION_mouth_corner_left_to_upper_lip = df["SEPARATION_mouth_corner_left_to_upper_lip"].mean()
average_for_this_video_SEPARATION_mouth_corner_right_to_upper_lip = df["SEPARATION_mouth_corner_right_to_upper_lip"].mean()
average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left = df["SEPARATION_pinna_ear_left_to_nose_corner_left"].mean()
average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right = df["SEPARATION_pinna_ear_right_to_nose_corner_right"].mean()
average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left = df["SEPARATION_pinna_ear_left_inner_brow_left"].mean()
average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right = df["SEPARATION_pinna_ear_right_inner_brow_right"].mean()
average_for_this_video_SEPARATION_upper_lip_to_lower_lip = df["SEPARATION_upper_lip_to_lower_lip"].mean()
average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right = df["SEPARATION_pinna_ear_left_to_pinna_ear_right"].mean()
average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left = df["SEPARATION_pinna_ear_left_to_mouth_corner_left"].mean()
average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right = df["SEPARATION_pinna_ear_right_to_mouth_corner_right"].mean()


# Stick the averages in the dataframe - each column just repeats one value - just for useful reference when exported to CSV
df = df.assign(average_for_this_video_SEPARATION_mouth_corner_left_to_upper_lip = average_for_this_video_SEPARATION_mouth_corner_left_to_upper_lip)
df = df.assign(average_for_this_video_SEPARATION_mouth_corner_right_to_upper_lip = average_for_this_video_SEPARATION_mouth_corner_right_to_upper_lip)
df = df.assign(average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left = average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left)
df = df.assign(average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right = average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right)
df = df.assign(average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left = average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left)
df = df.assign(average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right = average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right)
df = df.assign(average_for_this_video_SEPARATION_upper_lip_to_lower_lip = average_for_this_video_SEPARATION_upper_lip_to_lower_lip)
df = df.assign(average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right = average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right)
df = df.assign(average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left = average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left)
df = df.assign(average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right = average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right)

#print ("average_for_this_video_SEPARATION_mouth_corner_left_to_upper_lip: {}".format(average_for_this_video_SEPARATION_mouth_corner_left_to_upper_lip))
#print ("average_for_this_video_SEPARATION_mouth_corner_right_to_upper_lip: {}".format(average_for_this_video_SEPARATION_mouth_corner_right_to_upper_lip))
#print ("average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left: {}".format(average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left))
#print ("average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right: {}".format(average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right))
#print ("average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left: {}".format(average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left))
#print ("average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right: {}".format(average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right))
#print ("average_for_this_video_SEPARATION_upper_lip_to_lower_lip: {}".format(average_for_this_video_SEPARATION_mouth_corner_left_to_upper_lip))
#print ("average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right: {}".format(average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right))
#print ("average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left: {}".format(average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left))
#print ("average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right: {}".format(average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right))

############ 6. SPECIFY THE FORMULAS FOR CALCULATING THE EMOTION CHART LINES  ##########################################	

# Initially, set all chart lines to their default min values
df = df.assign(happy = happy_line_default_y, scared_sad = scared_sad_default_y, angry_disgusted = angry_disgusted_default_y, suprised = suprised_default_y)

######################################## START HAPPY ######################################################################

# Happy: DECREASE in the length of the line (i) SEPARATION_pinna_ear_left_to_mouth_corner_left and / or (ii) SEPARATION_pinna_ear_right_to_mouth_corner_right, relative to the average.

#calc_happy = lambda row: happy_line_default_y + ( (average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left - threshold) - row.SEPARATION_pinna_ear_left_to_mouth_corner_left) if row.SEPARATION_pinna_ear_left_to_mouth_corner_left < (average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left - threshold) else happy_line_default_y + ( (average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right - threshold) - row.SEPARATION_pinna_ear_right_to_mouth_corner_right) if row.SEPARATION_pinna_ear_right_to_mouth_corner_right < (average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right - threshold) else happy_line_default_y

calc_happy = lambda row: happy_line_default_y + (YgapBetweenChartLines -2) + row.SEPARATION_pinna_ear_left_to_mouth_corner_left if row.SEPARATION_pinna_ear_left_to_mouth_corner_left < (average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left - threshold) else happy_line_default_y + ( (average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right - threshold) - row.SEPARATION_pinna_ear_right_to_mouth_corner_right) if row.SEPARATION_pinna_ear_right_to_mouth_corner_right < (average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right - threshold) else happy_line_default_y

data_happy = df.apply(calc_happy, axis=1) # get column data with an index

df = df.assign(happy=data_happy.values) # assign values to column

# Normalise (Min-Max scale) range of values to between 0.0 and 1.0, then add the desired Y position of the chart line, then  multiply by 10, 
# in order to keep peaks within space between each chart line

#df[['happy']] = happy_line_default_y + (scaler.fit_transform(df[['happy']]) * 10)
df[['happy']] = happy_line_default_y + (scaler.fit_transform(df[['happy']]) * YgapBetweenChartLines-2)

######################################## END HAPPY ######################################################################

######################################## START SCARED / SAD ######################################################################

#Scared/Sad: DECREASE in the length of the line (i) SEPARATION_pinna_ear_left_inner_brow_left and / or (ii) SEPARATION_pinna_ear_right_inner_brow_right, relative to the average.

#calc_scared_sad = lambda row: scared_sad_default_y + ( (average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left - threshold) - row.SEPARATION_pinna_ear_left_inner_brow_left) if row.SEPARATION_pinna_ear_left_inner_brow_left < (average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left - threshold) else scared_sad_default_y + ( (average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right - threshold) - row.SEPARATION_pinna_ear_right_inner_brow_right) if row.SEPARATION_pinna_ear_right_inner_brow_right < (average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right - threshold) else scared_sad_default_y

calc_scared_sad = lambda row: scared_sad_default_y + (YgapBetweenChartLines -2) + row.SEPARATION_pinna_ear_left_inner_brow_left if row.SEPARATION_pinna_ear_left_inner_brow_left < (average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left - threshold) else scared_sad_default_y + ( (average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right - threshold) - row.SEPARATION_pinna_ear_right_inner_brow_right) if row.SEPARATION_pinna_ear_right_inner_brow_right < (average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right - threshold) else scared_sad_default_y

data_scared_sad = df.apply(calc_scared_sad, axis=1) # get column data with an index

df = df.assign(scared_sad=data_scared_sad.values) # assign values to column

# Normalise (Min-Max scale) range of values to between 0.0 and 1.0, then add the desired Y position of the chart line, then  multiply by 10, 
# in order to keep peaks within space between each chart line

df[['scared_sad']] = scared_sad_default_y + (scaler.fit_transform(df[['scared_sad']]) * 10)

#df[['scared_sad']] = scared_sad_default_y + (scaler.fit_transform(df[['scared_sad']]) * 10)

df[['scared_sad']] = scared_sad_default_y + (scaler.fit_transform(df[['scared_sad']]) * YgapBetweenChartLines-2)


######################################## END SCARED / SAD ######################################################################

######################################## START ANGRY / DISGUSTED ######################################################################

#Angry/Disgusted: INCREASE in the length of the line (i) SEPARATION_upper_lip_to_lower_lip AND a significant, frame-to-frame DECREASE in the length of both the line SEPARATION_pinna_ear_left_to_nose_corner_left and the line SEPARATION_pinna_ear_right_to_nose_corner_right, relative to the average.

#calc_angry_disgusted = lambda row: angry_disgusted_default_y + ( (average_for_this_video_SEPARATION_upper_lip_to_lower_lip - threshold) - row.SEPARATION_upper_lip_to_lower_lip) if row.SEPARATION_upper_lip_to_lower_lip > (average_for_this_video_SEPARATION_upper_lip_to_lower_lip + threshold) and row.SEPARATION_pinna_ear_left_to_nose_corner_left < (average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left - threshold) and row.SEPARATION_pinna_ear_right_to_nose_corner_right < (average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right - threshold) else angry_disgusted_default_y

calc_angry_disgusted = lambda row: angry_disgusted_default_y + (YgapBetweenChartLines -2) + row.SEPARATION_upper_lip_to_lower_lip if row.SEPARATION_upper_lip_to_lower_lip > (average_for_this_video_SEPARATION_upper_lip_to_lower_lip + threshold) and row.SEPARATION_pinna_ear_left_to_nose_corner_left < (average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left - threshold) and row.SEPARATION_pinna_ear_right_to_nose_corner_right < (average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right - threshold) else angry_disgusted_default_y

data_angry_disgusted = df.apply(calc_angry_disgusted, axis=1) # get column data with an index

df = df.assign(angry_disgusted=data_angry_disgusted.values) # assign values to column

# Normalise (Min-Max scale) range of values to between 0.0 and 1.0, then add the desired Y position of the chart line, then  multiply by 10, 
# in order to keep peaks within space between each chart line

#df[['angry_disgusted']] = angry_disgusted_default_y + (scaler.fit_transform(df[['angry_disgusted']]) * 10)

df[['angry_disgusted']] = angry_disgusted_default_y + (scaler.fit_transform(df[['angry_disgusted']]) * YgapBetweenChartLines-2)

######################################## END ANGRY / DISGUSTED ######################################################################


######################################## START SUPRISED ######################################################################

#Surprised: at least any three out of the following five tests must be true, with all of the three coming from the same group - either #1 or #2, except for the ear to ear distance, which may be used in combination with either group: 
#INCREASE in the length of the line SEPARATION_pinna_ear_left_to_pinna_ear_right, relative to the average.

#Group #1
#pinna_ear_left is significantly decreased in its Y coordinate, relative to pinna_ear_right
#nose_corner_left is significantly decreased in its Y coordinate, relative to nose_corner_right
#mouth_corner_left is significantly decreased in its Y coordinate, relative to mouth_corner_right
#or
#Group #2
#pinna_ear_right is significantly decreased in its Y coordinate, relative to pinna_ear_left
#nose_corner_right is significantly decreased in its Y coordinate, relative to nose_corner_left
#mouth_corner_right is significantly decreased in its Y coordinate, relative to mouth_corner_left


calc_ears_tilted = lambda row: True if row.pinna_ear_left_y  < (row.pinna_ear_right_y - threshold) or row.pinna_ear_right_y  < (row.pinna_ear_left_y  - threshold) else False
data_ears_tilted = df.apply(calc_ears_tilted, axis=1) # get column data with an index
df = df.assign(ears_tilted=data_ears_tilted.values) # assign values to column

calc_brows_tilted = lambda row: True if row.inner_brow_left_y  < (row.inner_brow_right_y - threshold) or row.inner_brow_right_y  < (row.inner_brow_left_y  - threshold) else False
data_brows_tilted = df.apply(calc_brows_tilted, axis=1) # get column data with an index
df = df.assign(brows_tilted=data_brows_tilted.values) # assign values to column

calc_nose_tilted = lambda row: True if row.nose_corner_left_y  < (row.nose_corner_right_y - threshold) or row.nose_corner_right_y  < (row.nose_corner_left_y  - threshold) else False
data_nose_tilted = df.apply(calc_nose_tilted, axis=1) # get column data with an index
df = df.assign(nose_tilted=data_nose_tilted.values) # assign values to column

calc_mouth_tilted = lambda row: True if row.mouth_corner_left_y  < (row.mouth_corner_right_y - threshold) or row.mouth_corner_right_y  < (row.mouth_corner_left_y  - threshold) else False
data_mouth_tilted = df.apply(calc_mouth_tilted, axis=1) # get column data with an index
df = df.assign(mouth_tilted=data_mouth_tilted.values) # assign values to column

calc_ears_apart = lambda row: True if row.SEPARATION_pinna_ear_left_to_pinna_ear_right > (average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right + ears_apart_threshold) else False
data_ears_apart = df.apply(calc_ears_apart, axis=1) # get column data with an index
df = df.assign(ears_apart=data_ears_apart.values) # assign values to column

# Have to admit that adding in row.SEPARATION_pinna_ear_left_to_pinna_ear_right is just to give some variability to a flat line, when head tilted!
calc_suprised = lambda row: suprised_default_y + (YgapBetweenChartLines -2) + row.SEPARATION_pinna_ear_left_to_pinna_ear_right if sum([row.ears_apart,row.ears_tilted,row.brows_tilted,row.nose_tilted,row.mouth_tilted]) >= numberOfParamsOutOfFiveRequiredTrueForHeadTiltConfirmation else suprised_default_y

# Note: in above line, the sum() function works because Booleans in Python are kind of Integers...So, True == 1. 
# Therefore, if adding up: ears_apart + ears_tilted + brows_tilted + nose_tilted + mouth_tilted is greater than numberOfParamsOutOfFiveRequiredTrueForHeadTiltConfirmation
# ... (in this case 3), then 3 out of 5 being true is a good enough test to assume that the dog's head is tilted.

data_suprised = df.apply(calc_suprised, axis=1) # get column data with an index

df = df.assign(suprised=data_suprised.values) # assign values to column

# Normalise (Min-Max scale) range of values to between 0.0 and 1.0, then add the desired Y position of the chart line, then  multiply by 10, 
# in order to keep peaks within space between each chart line

df[['suprised']] = suprised_default_y + (scaler.fit_transform(df[['suprised']]) * YgapBetweenChartLines-2)

######################################## END SUPRISED ######################################################################



#print (df)

# plot

############ 7. PLOT THE CHART LINES AND SAVE OUTPUTS: VIDEO, STILL IMAGE, FRAMES, CSV OUTPUT ETC.  ##########################################	

figure,ax = plt.subplots()

plot_line_width = 2.0

happy_line, = ax.plot(df.frame,df.happy, happy_chart_line_colour, linestyle='-', marker='o', markersize=1, alpha=1.0, linewidth=plot_line_width, label='happy')
scared_sad_line, = ax.plot(df.frame,df.scared_sad, scared_sad_chart_line_colour, linestyle='-', marker='o', markersize=1, alpha=1.0, linewidth=plot_line_width, label='scared or sad')
angry_disgusted_line, = ax.plot(df.frame,df.angry_disgusted, angry_disgusted_chart_line_colour, linestyle='-', marker='o', markersize=1, alpha=1.0, linewidth=plot_line_width, label='angry or digusted')
suprised_line, = ax.plot(df.frame,df.suprised, suprised_chart_line_colour, linestyle='-', marker='o', markersize=1, alpha=1.0, linewidth=plot_line_width, label='suprised')

ax.legend()

plt.rcParams['savefig.facecolor']='black' # Messed about for ages here with ax.set_facecolor("black") and #ax.set_axis_bgcolor("black"), which didn't work. This works for video output, but only for JPG / PNG image when transparent=False. Not working for plt.show() (!)

plt.xlim(0,len(df.index))
plt.axis('off')
plt.box(False)
#plt.show()

figure.savefig(csvFileDirName + ".\\" + csvFileNameWithoutExtension + "_EMOTIONS_OUTPUT.jpg",bbox_inches='tight', transparent=False, dpi=300)

# Output entire dataframe as CSV
#df.to_csv(csvFileDirName + ".\\" + csvFileNameWithoutExtension + "_EMOTIONS_OUTPUT.csv", sep=',', encoding='utf-8')


# Customised CSV output, choosing columns and order
header = ["frame", "average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left", "SEPARATION_pinna_ear_left_to_mouth_corner_left","average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right", "SEPARATION_pinna_ear_right_to_mouth_corner_right", "happy","average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left", "SEPARATION_pinna_ear_left_inner_brow_left", "average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right", "SEPARATION_pinna_ear_right_inner_brow_right", "scared_sad", "average_for_this_video_SEPARATION_upper_lip_to_lower_lip","SEPARATION_upper_lip_to_lower_lip", "average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left","SEPARATION_pinna_ear_left_to_nose_corner_left", "average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right","SEPARATION_pinna_ear_right_to_nose_corner_right", "angry_disgusted", "average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right","SEPARATION_pinna_ear_left_to_pinna_ear_right", "ears_apart", "ears_tilted", "brows_tilted", "nose_tilted", "mouth_tilted", "suprised"]


df.to_csv(csvFileDirName + ".\\" + csvFileNameWithoutExtension + "_EMOTIONS_OUTPUT.csv", sep=',', encoding='utf-8', columns = header)

metadata = dict(title='DeepLabCut chart', artist='User',
                comment='Say something...')
				
writer = FFMpegWriter(fps=25, metadata=metadata)
#writer = FFMpegWriter(fps=len(df.index), metadata=metadata)

with writer.saving(figure, csvFileDirName + ".\\" + csvFileNameWithoutExtension + ".mp4", 100):
	for n in range(len(df.index) + 25): # Note: the +25 is only needed because I wanted to add an extra second at the end of the vid to get around an OpenShot bug!
		#print ("Creating frame %d of %d" % (n+1, len(df.index)))   
		#print(df.SEPARATION_mouth_corner_left_to_upper_lip[:n])
		happy_line.set_data(df.frame[:n],df.happy[:n]) # important! Don't forget that ':n' means everything upto n - this why this works... each video frame includes the new data point and all points up to the current point - if you put 'n', then each video frame would only include the current datapoint
		scared_sad_line.set_data(df.frame[:n],df.scared_sad[:n])
		angry_disgusted_line.set_data(df.frame[:n],df.angry_disgusted[:n])
		suprised_line.set_data(df.frame[:n],df.suprised[:n])
		writer.grab_frame()
		#if not os.path.exists(csvFileDirName + '\\frames\\'):   # umcomment this and next two lines to save individual frames
		  #os.makedirs(csvFileDirName + '\\frames\\')
		#figure.savefig(csvFileDirName + '\\frames\\Frame%03d.png' %n,bbox_inches='tight', transparent=True, dpi=300) 
		

#mouth_corner_left_x
#mouth_corner_left_y
#mouth_corner_right_x
#mouth_corner_right_y
#inner_brow_left_x
#inner_brow_left_y
#inner_brow_right_x
#inner_brow_right_y
#upper_lip_x
#upper_lip_y
#lower_lip_x
#lower_lip_y
#nose_corner_left_x
#nose_corner_left_y
#nose_corner_right_x
#nose_corner_right_y
#pinna_ear_left_x
#pinna_ear_left_y
#pinna_ear_right_x
#pinna_ear_right_y


# Separation lines between facial landmark labels:

#average_for_this_video_SEPARATION_mouth_corner_left_to_upper_lip
#average_for_this_video_SEPARATION_mouth_corner_right_to_upper_lip
#average_for_this_video_SEPARATION_pinna_ear_left_to_nose_corner_left
#average_for_this_video_SEPARATION_pinna_ear_right_to_nose_corner_right
#average_for_this_video_SEPARATION_pinna_ear_left_inner_brow_left
#average_for_this_video_SEPARATION_pinna_ear_right_inner_brow_right
#average_for_this_video_SEPARATION_upper_lip_to_lower_lip
#average_for_this_video_SEPARATION_pinna_ear_left_to_pinna_ear_right
#average_for_this_video_SEPARATION_pinna_ear_left_to_mouth_corner_left
#average_for_this_video_SEPARATION_pinna_ear_right_to_mouth_corner_right


#SEPARATION_mouth_corner_left_to_upper_lip
#SEPARATION_mouth_corner_right_to_upper_lip
#SEPARATION_pinna_ear_left_to_nose_corner_left
#SEPARATION_pinna_ear_right_to_nose_corner_right
#SEPARATION_pinna_ear_left_inner_brow_left
#SEPARATION_pinna_ear_right_inner_brow_right
#SEPARATION_upper_lip_to_lower_lip
#SEPARATION_pinna_ear_left_to_pinna_ear_right
#SEPARATION_pinna_ear_left_to_mouth_corner_left
#SEPARATION_pinna_ear_right_to_mouth_corner_right


#Labelled points for posture estimation with DeepLabCut:

#mouth_corner_left
#mouth_corner_right
#inner_brow_left
#inner_brow_right
#upper_lip
#lower_lip
#nose_corner_left
#nose_corner_right
#pinna_ear_left
#pinna_ear_right
	


 

	
	
	


