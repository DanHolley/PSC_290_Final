
# coding: utf-8

# Translational studies of nonhuman subjects are vital to an improved understanding of human anxiety disorders.  Correlational 
# data between subjects' behavior under certain contexts and the functional and structural connectivity of those subjects' threat-
# responsive neural substrates may offer particularly keen insight into the ways that the our brains give rise to, and reinforce, 
# anxious behaviors.  In such translational studies, subjects' behavior is often scored by human observers.  For example, in studies 
# of nonhuman primate species, researchers are often interested in freezing behavior (i.e., a period of immobility lasting longer 
# than 3 seconds, during which the subject attends to a threatening stimulus) as a measure of anxious temperament.  While this method 
# of hand scoring has unquestionably contributed to an improved understanding of animal behavior and its translational correlates, 
# computer-based methods may be able to obviate hand scoring and augment behavioral studies by improving the speed and accuracy of 
# the scoring process.  Furthermore, a precise, objective, computer-based tool would potentially allow researchers to replace the 
# ecologically questionable 3-second rule by which freezing is currently defined with a less problematic measure (e.g., total number 
# of immobile versus mobile frames, overall percent of time spent immobilized, etc.).
# 
# Here, we present a novel computer-automated tool for behavioral scoring in experimental paradigms featuring contexts designed to 
# elicit freezing, for instance the human intruder paradigm.  The tool analyzes the video in a stepwise, frame-by-frame fashion.  
# Its underlying function is grounded in the principle that a delta of uncommon pixels can be obtained between any two contiguous 
# video frames, and that the delta will increase when an animal is in motion.  Our tool exploits this principle to generate one-
# dimensional Gaussian mixture models and two-dimensional k-means clustering models of immobility versus mobility for each subject.  
# From these models, motion and freezing can be inferred, and then quantified under thresholds specified by the researcher.

# In[2]:


# If memory/processing speed becomes an issue, may need to launch Jupyter from terminal with 
# "jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000"

# Necessary imports:

import os
import imageio
import itertools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scipy import ndimage, misc, stats
from scipy.misc import imsave
from scipy.ndimage.filters import minimum_filter
from scipy.ndimage.filters import maximum_filter
from matplotlib.font_manager import FontProperties
from sklearn.mixture import GMM
from sklearn.cluster import KMeans
from scipy.stats import mode

# http://zulko.github.io/moviepy/
from moviepy.editor import *
import pygame
get_ipython().magic(u'matplotlib inline')
imageio.plugins.ffmpeg.download()


# In[3]:


### STYLE ###

# set graph size:
size = (6, 3.5)

# set distance between header and top of graph
header_space = 0.88 


# In[4]:


# Create a list of video files to score.
# VIDEOS MUST FIRST BE STORED IN YOUR JUPYTER ENVIRONMENT!

video_files = [VideoFileClip("motion_test.mov"), VideoFileClip("test_vid_2.mov"), VideoFileClip("test_vid_4.mov"), VideoFileClip("paprika3.mov")]
#video_files = [VideoFileClip("motion_test.mov"), VideoFileClip("paprika3.mov"), VideoFileClip("test_vid_4.mov"), VideoFileClip("test_vid_5.mov")]


# To avoid the problem of changes in color patterns as an animal moves about its enclosure, we convert each video to greyscale 
# using the following function.

# In[5]:


# RGB to grayscale function:
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


# Next, we step through each video to obtain various difference scores between contiguous frames.  Currently, we use 
# sum of squared difference (SSD) scores and coefficient of determination (r squared) scores for this purpose.

# In[6]:


### GET FRAME-BASED, STEPWISE SIMILARITY SCORES ###

sum_sq_diff_list = []

r_squared_values_list = []

# Loop over each video to be scored:
for video in video_files:
    sum_sq_diff = []
    r_squared_values = []
    prev_frame = []
    total_frames = 0
    
    # Within each video, loop over individual frames, comparing each frame to its preceding frame:
    for frame in video.iter_frames(fps=30):
        this_frame = rgb2gray(frame)
        if prev_frame != []:
            total_frames += 1
            
            # Get the delta between contiguous frames:
            diff_frame = this_frame-prev_frame
            r_squared = (np.corrcoef(np.ravel(this_frame), np.ravel(prev_frame))[0,1])**2
            r_squared_values.append(r_squared)
            
            # Square each delta to eliminate negative values:
            diff_frame_sq = np.power(diff_frame, 2)
            sum_diff_frame_sq = np.sum( diff_frame_sq )
            sum_sq_diff.append( sum_diff_frame_sq )
            
            # Get the r-squared (or Pearson's) correlation coefficient between each frame:
            # ::: Do something to diff_frame to get this value; store that value a float variable :::
            # ::: Append the value to the "pearson = []" list :::
            
        prev_frame = this_frame
        
    # Append the full list of squared deltas to a master list of 
    # squared deltas for each of the n videos to be scored:
    sum_sq_diff_list.append(sum_sq_diff)
    r_squared_values_list.append(r_squared_values)
    print total_frames
    
    # Plot some contiguous frames and delta frames from each video;
    # note that the deltas (rightmost column) between these frames are extremely small:
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(this_frame)
    axarr[1].imshow(prev_frame)
    axarr[2].imshow(diff_frame)


# In[7]:


### PLOT AND STORE RAW, NORMALIZED, AND LOG-TRANSFORMED/NORMALIZED R-SQUARED VALUES ###

palette = sns.cubehelix_palette(n_colors = len(r_squared_values_list), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

count = 0

normalized_r2_values = []

log_transformed_normalized_r2_values = []

# outliers_list = [] # TURN ON IF OPTIONAL OUTLIER FILTER (BELOW) IS ACTIVE

# Iterate over the list of lists of r-squared values:
for values in r_squared_values_list:
    
    temp_normalized_values = []
    temp_normalized_values_2 = []
    temp_log_transformed_normalized_values = []
    
    # Iterate over each r-squared value, to get/store normalized and log-transformed/normalized values:
    for value in values:
        norm_value = (1 - ((value-min(values))/min(values))) * 100
        temp_normalized_values.append(norm_value)
        
    for value in temp_normalized_values:
        norm_value_2 = value - min(temp_normalized_values)
        temp_normalized_values_2.append(norm_value_2)

    count_2 = 0
    for value in temp_normalized_values_2:
        # Get log-transformed normalized values:
        log_transformed = np.log(temp_normalized_values_2[count_2] + 1 - min(temp_normalized_values_2)) # Constant added to avoid negative values
        if np.isfinite(log_transformed) == True:
            temp_log_transformed_normalized_values.append(log_transformed)
        else:
            temp_log_transformed_normalized_values.append(0)
        count_2 += 1
        
    # Plot histograms of the normalized r-squared values:
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3), sharex=False, sharey=False, subplot_kw={'adjustable': 'box-forced'})
    fig.suptitle('R-Squared Values: Video {}'.format(count + 1), fontsize=15)
    ax = axes.ravel()

    # Unadjusted r-squared scores:
    ax[0].set_ylabel('frames', fontsize = 15)
    ax[0].set_xlabel('raw r-squared value', fontsize = 15)
    ax[0].hist(values, edgecolor = "black", bins=25, color = palette[count]);
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlim(left=min(values), right=max(values))

    # Normalized r-squared scores:
    ax[1].set_xlabel('normalized r-squared value', fontsize = 15)
    ax[1].hist(temp_normalized_values_2, edgecolor = "black", bins=25, color = palette[count]);
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlim(left=min(temp_normalized_values_2), right=max(temp_normalized_values_2))
    
    # Log-transformed/normalized r-squared scores:
    plot_me = temp_log_transformed_normalized_values - min(temp_log_transformed_normalized_values)
    #print np.median
    ax[2].set_xlabel('log-transformed normalized\nr-squared value', fontsize = 15)
    ax[2].hist(plot_me, edgecolor = "black", bins=25, color = palette[count]);
    ax[2].set_ylim(bottom=0)
    ax[2].set_xlim(left=0, right=max(plot_me))
    ax[2].axvline(x = np.median(plot_me), c = "r", dashes = [1, 2], label = "mean:\n{:.2f}".format(np.mean(plot_me)))
    ax[2].legend(bbox_to_anchor=(1.02, 1.03), loc="upper right", borderaxespad=0., fontsize = 14, frameon=True, framealpha=1, edgecolor = "inherit", handletextpad=.5)
    
    # Store each list of normalized r-squared values outside the loop for later use:
    normalized_r2_values.append(temp_normalized_values_2)
    log_transformed_normalized_r2_values.append(temp_log_transformed_normalized_values)
        
    count += 1


# Next we apply sliding minimum and maximum filters to the frames.  These filters step through the frames in groupings of 
# size n (specified by the user) and find the smallest and largest deltas, respectively, within any n-frame group. A large 
# difference between minimum and maximum filter scores suggests that the subject is moving. 

# In[8]:


# Several lists generated in the following for loop will need to be saved as lists of sub-lists for later use:
sum_sq_diff_nparray_list = []
min_vectors_list = []
max_vectors_list = []
diff_vectors_list = []
my_ordered_array_list = []
log_transformed_normalized_r2_values_x10 = []

for list_of_values in log_transformed_normalized_r2_values:
    temp_storage = []
    for value in list_of_values:
        temp_storage.append(value * 10)
    log_transformed_normalized_r2_values_x10.append(temp_storage)

count = 0

for sum_of_sq_differences in sum_sq_diff_list:
    min_vectors = []    
    n = 10 # This can be changed to alter sensitivity of the filter. A lower value would be LESS sensitive to motion.
    sum_sq_diff_nparray = np.array(sum_of_sq_differences)
    # Normalize SSD:
    sum_sq_diff_nparray = ((sum_sq_diff_nparray)/max(sum_sq_diff_nparray))*100

    # Use a sliding window of n-frames to calculate the highest and lowest
    # deltas among those frames. Store those values as max and min vectors.
    # Large divergence in max/min values indicates motion.
    
    min_vector = minimum_filter(sum_sq_diff_nparray, size=n, mode='constant')
    #max_vector = minimum_filter(sum_sq_diff_nparray*-1, size=n, mode='constant')*-1
    max_vector = maximum_filter(sum_sq_diff_nparray, size=n, mode='constant')
    
    diff_vector = max_vector - min_vector

    my_ordered_array = np.arange(0,len(sum_of_sq_differences))

    # Generate plots:
    fig = plt.figure()
    fig.set_size_inches(12, 3.5)
    fig.suptitle('Frame-by-Frame Filter Scores: Video {}'.format(count + 1), fontsize = 15)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=header_space)

    ax.set_xlabel('frame', fontsize = 15)
    ax.set_ylabel('filter scores', fontsize = 15)

    plt.plot(my_ordered_array, min_vector, 'orange', label = "minimum filter", alpha = .7, linewidth = 1)
    plt.plot(my_ordered_array, max_vector, 'g-', label = "maximum filter", alpha = .7, linewidth = 1)
    plt.plot(my_ordered_array, diff_vector, label = "difference between\nmax and min filters", alpha = 1, linewidth = 1)

    # These next two lines of code can be turned on to see how some other metrics map to the filter scores:
#     plt.plot(my_ordered_array, normalized_r2_values[count], 'r-', label = "normalized\nr-squared values", alpha = .5, linewidth = 2)
#     plt.plot(my_ordered_array, log_transformed_normalized_r2_values_x10[count], 'orange', label = "log-norm r-squared\nvalues * 10", alpha = .5, linewidth = 2)

    ax.legend(bbox_to_anchor=(1.08, .5), loc = "center", borderaxespad=0., fontsize = 14, frameon=True, framealpha=1, edgecolor = "inherit", handletextpad=.5)
    
    # Store necessary data in their respective lists for later use:
    sum_sq_diff_nparray_list.append(sum_sq_diff_nparray)
    min_vectors_list.append(min_vector)
    my_ordered_array_list.append(my_ordered_array)
    max_vectors_list.append(max_vector)
    diff_vectors_list.append(diff_vector)
    
    count += 1


# In[9]:


### CATEGORIZE FREEZING AND MOTION USING MINIMUM VECTOR SCORES AS A CUTOFF ###

freezing_list = []

#freezing_threshold_list = [] # We can reactivate this line if we set a dynamic freezing threshold

count = 0

for min_vector in min_vectors_list:
    # Create a min_vector-length array of zeros:
    freezing = np.zeros_like(diff_vectors_list[count])
    
    # Filter-out noise. 
    # Lower values would be more sensitive to motion.
    # Give it a try by changing the threshold to a 
    # different value and re-examining the plots.
    freezing_threshold = 4 # Usually set to 4
    freezing[diff_vectors_list[count] < np.mean(diff_vectors_list[count])] = max(diff_vectors_list[count])

    # Generate plots:
    fig = plt.figure()
    fig.set_size_inches(12, 3.5)
    fig.suptitle('Filtered Motion Score: Video {}'.format(count +1), fontsize=15)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=header_space)

    ax.set_xlabel('frame', fontsize = 15)
    ax.set_ylabel('filter scores', fontsize = 15)

#     plt.plot(my_ordered_array_list[count], min_vector, 'orange', label = "minimum filter", alpha = .7)
#     plt.plot(my_ordered_array_list[count], max_vectors_list[count], 'g-', label = "maximum filter", alpha = .7)
    plt.plot(my_ordered_array_list[count], diff_vectors_list[count], label = "difference between\nmax and min filters", alpha = 1, linewidth = 1)
    plt.plot(my_ordered_array_list[count], freezing, "black", label = '"freezing" score', linewidth = 1)
    plt.axhline(y = np.mean(diff_vectors_list[count]), color = "r", dashes = [1, 2], label = "mean difference score: {:.2f}".format(np.mean(diff_vectors_list[count])))

    ax.legend(bbox_to_anchor=(1.15, .5), loc = "center", borderaxespad=0., fontsize = 14, frameon=True, framealpha=1, edgecolor = "inherit", handletextpad=.5)
    
    # Store freezing values for each of the n videos as a list of sub-lists, for later use:
    freezing_list.append(freezing)
    
    #freezing_threshold_list.append(freezing_threshold) # We can reactivate this line if we set a dynamic freezing threshold
    
    count += 1


# Next, we create some histograms of various scores and log-transformed scores for each video, so that we can visualize 
# the distribution of scores.  A subject whose normalized scores tend to cluster around zero doesn't move very much 
# compared to a subject whose normalized scores push further right.  From that, we might infer that a low-scoring subject 
# exhibits more freezing behavior, which could be interpreted as indicative of an "anxious" phenotype. 

# In[10]:


# Plot the maximum vector scores from each video as a histogram to visualize
# the distribution of scores that indicate freezing/motion: 

palette = sns.cubehelix_palette(n_colors = len(max_vectors_list), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

count = 0

max_vector_logs = []

red_palette = 10*["#F5C1C0", "#F7CECD", "#F9DADA", "#FBE7E6"]

for max_vector in max_vectors_list:

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3), sharex=False, sharey=False, subplot_kw={'adjustable': 'box-forced'})
    fig.suptitle('Maximum Filter Scores: Video {}'.format(count + 1), fontsize=15)
    fig.subplots_adjust(top=.85)
    ax = axes.ravel()

    # Raw max filter scores:
    ax[0].set_ylabel('frames', fontsize = 15)
    ax[0].set_xlabel('raw max filter scores', fontsize = 15)
    ax[0].hist(max_vector, edgecolor = "black", bins=10, color = palette[count]);
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlim(left=min(max_vector), right=max(max_vector))

    # Log-transformed max filter scores:
    max_vector_log = np.log(max_vector + 1 - min(max_vector))
    max_vector_logs.append(max_vector_log)
    ax[1].set_xlabel('log-transformed max filter scores', fontsize = 15)
    ax[1].hist(max_vector_log, edgecolor = "black", bins=10, color = palette[count]);
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlim(left=min(max_vector_log), right=max(max_vector_log))
    ax[1].axvline(x = np.median(max_vector_log), c = "r", dashes = [1, 2], label = "median:\n{:.2f}".format(np.median(max_vector_log)))
    ax[1].legend(bbox_to_anchor=(1.02, 1.035), loc="upper right", borderaxespad=0., fontsize = 14, frameon=True, framealpha=1, edgecolor = "inherit", handletextpad=.5)

    # Overlay comparison of log-transformed max filter scores and log-transformed normalized r-squared values:
    ax[2].set_xlabel('log-transformed max filter scores with\nlog-transformed r-squared overlay', fontsize = 14.5)
    ax[2].hist(max_vector_log, edgecolor = "black", bins=10, color = palette[count]);
    ax[2].hist(log_transformed_normalized_r2_values[count], bins=10, color = "r", alpha = .2) #- (0.05*count));
    ax[2].set_ylim(bottom=0)
    ax[2].set_xlim(left=0)
    patch = mpatches.Patch(facecolor=red_palette[1], edgecolor = "black", label="log-transformed\nr-squared values")
    ax[2].legend(handles=[patch], fontsize = 12, edgecolor = "inherit", framealpha=1, bbox_to_anchor=(1.20, 1.07), loc="upper right", frameon = True)

    count += 1


# In[11]:


# Plot the max-min difference scores from each video as a histogram to visualize
# the distribution of scores that indicate freezing/motion: 

palette = sns.cubehelix_palette(n_colors = len(max_vectors_list), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

count = 0

diff_vector_logs = []

red_palette = 10*["#F5C1C0", "#F7CECD", "#F9DADA", "#FBE7E6"]

for diff_vector in diff_vectors_list:

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3), sharex=False, sharey=False, subplot_kw={'adjustable': 'box-forced'})
    fig.suptitle('Maximum Filter Scores: Video {}'.format(count + 1), fontsize=15)
    fig.subplots_adjust(top=.85)
    ax = axes.ravel()

    # Raw max-min difference scores scores:
    ax[0].set_ylabel('frames', fontsize = 15)
    ax[0].set_xlabel('max-min difference scores', fontsize = 15)
    ax[0].hist(diff_vector, edgecolor = "black", bins=10, color = palette[count]);
    ax[0].set_ylim(bottom=0)
    ax[0].set_xlim(left=min(diff_vector), right=max(diff_vector))

    # Log-transformed max-min difference scores:
    diff_vector_log = np.log(diff_vector + 1 - min(diff_vector))
    diff_vector_logs.append(diff_vector_log)
    ax[1].set_xlabel('log-transformed max-min\ndifference scores', fontsize = 15)
    ax[1].hist(diff_vector_log, edgecolor = "black", bins=10, color = palette[count]);
    ax[1].set_ylim(bottom=0)
    ax[1].set_xlim(left=min(diff_vector_log), right=max(diff_vector_log))
    ax[1].axvline(x = np.median(diff_vector_log), c = "r", dashes = [1, 2], label = "median:\n{:.2f}".format(np.median(diff_vector_log)))
    ax[1].legend(bbox_to_anchor=(1.02, 1.035), loc="upper right", borderaxespad=0., fontsize = 14, frameon=True, framealpha=1, edgecolor = "inherit", handletextpad=.5)

    # Overlay comparison of log-transformed max-min difference scores and log-transformed normalized r-squared values:
    ax[2].set_xlabel('log-transformed max-min difference scores\nwith log-transformed r-squared overlay', fontsize = 14.5)
    ax[2].hist(diff_vector_log, edgecolor = "black", bins=10, color = palette[count]);
    ax[2].hist(log_transformed_normalized_r2_values[count], bins=10, color = "r", alpha = .2) #- (0.05*count));
    ax[2].set_ylim(bottom=0)
    ax[2].set_xlim(left=0)
    patch = mpatches.Patch(facecolor=red_palette[1], edgecolor = "black", label="log-transformed\nr-squared values")
    ax[2].legend(handles=[patch], fontsize = 12, edgecolor = "inherit", framealpha=1, bbox_to_anchor=(1.20, 1.07), loc="upper right", frameon = True)

    count += 1


# Now that we have visualized the data a bit, we can move on to fitting the data to some models.  The first such model 
# is a probabilistic, one-dimensional Gaussian mixture model (GMM).  Imagine that you have some data, and that you 
# believe that the data arose from two different groups; however, it may be difficult to determine which data originated 
# from which group.  GMM can aid inthis determination by algorithmically fitting the data to a specified number of 
# underlying Gaussians.  Here, we will first determine the "right" number of Gaussians through AIC and BIC comparisons 
# of models ranging from 1 to 15 Gaussians.  The winning model's BIC score will be retained for downstream use (i.e., 
# actually plotting the GMM).  Although there will often be convergence of AIC and BIC scores, where there is not, BIC 
# is preferred because of its greater stringency (i.e., BIC penalizes the addition of Gaussians more heavily than AIC does).

# In[44]:


# ::: Moving forward, I should probably run this 10-20 times and select the modal BIC value for plotting
# ::: and downstream use, to hedge against the prospect of a lower-probability BIC score winning out.

### MODEL COMPARISONS ###
def model_comparison(data):
        
    count = 1

    min_aics = []
    
    global min_bics # stored as a global variable to facilitate its use in the subsequent GMM generator (next cell)
    min_bics = []

    # Iterate over list of lists (data):
    for datum in data:

        # Reshape the data to facilitate Gaussian modeling:
        datum.shape = (datum.shape[0],1)

        # Test models for each video against a range of GMMs with between 1 and 15 underlying Gaussians:
        n_estimators = np.arange(1, 16)
        clfs = [GMM(n, n_iter=1000).fit(datum) for n in n_estimators]
        aics = [clf.aic(datum) for clf in clfs]
        bics = [clf.bic(datum) for clf in clfs]

        #Grab the lowest AIC and BIC scores from each video's model trials.
        # The lowest scores correspond to the optimal number of Gaussians to use
        # per video, so we can grab each of those by indexing the aics/bics lists
        # for lowest values (and then adding 1 to account for zero-indexing):
        min_aics.append(aics.index(min(aics))+1)
        min_bics.append(bics.index(min(bics))+1)

        # Plot the AIC and BIC scores:
        fig = plt.figure() 
        ax = fig.add_subplot(111)  

        fig.suptitle("Comparing 1-Dimensional n-Gaussian Models\n via AIC and BIC Testing: Video {}".format(count), fontsize=16)

        fig.subplots_adjust(top=header_space - .05)

        plt.plot(n_estimators, aics, label= "Lowest AIC = {:.2f} at {} Gaussians".format(min(aics), aics.index(min(aics))+1))
        plt.plot(n_estimators, bics, label= "Lowest BIC = {:.2f} at {} Gaussians".format(min(bics), bics.index(min(bics))+1))
        plt.legend(fontsize = 14, bbox_to_anchor=(1.49, 0.5), loc="center", frameon = False)
        plt.axvline(x = aics.index(min(aics))+1, color = "blue", ls = "dotted")
        plt.axvline(x = bics.index(min(bics))+1, color = "orange", ls = "dotted")

        ax = plt.gca()

        ax.set_xlabel("Number of Gaussians", fontsize = 15)
        ax.set_ylabel("AIC/BIC Score", fontsize = 15)

        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

        count += 1

    print min_aics
    print min_bics
    
model_comparison(diff_vector_logs)


# Next we will fit a GMM to each each video using the lowest (i.e., winning) BIC score, which we determined and 
# have carried over from the previous cell.  The leftmost Gaussian in any model represents the data least indicative 
# of movement; the rightmost Gaussian represents the data most indicative of movement.  Subjects that have clear 
# motion/immobility patterns, then, might be expected to have a somewhat bimodal output, here. 

# In[63]:


### PLOT GAUSSIAN MIXTURE MODEL W/ LOWEST BIC SCORE PER VIDEO ###
# http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.mixture.GMM.html

def Gaussian_mixture_model(data, posterior_probability_check = None):
    
    palette = sns.cubehelix_palette(n_colors = len(data), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

    count = 0

    # Save some values to use later:
    xpdf_list = []
    density_list = []
    clf_list = []
        
    # Iterate over data set to generate histograms and composite Gaussians
    for datum in data:
        
        # Reshape the data to facilitate Gaussian modeling:
        datum.shape = (datum.shape[0],1)

        # Set the Gaussian mixed model to fit 2 Gaussians;
        # conduct 300 iterations to determine the best fit:
        clf = GMM(min_bics[count], n_iter = 300).fit(datum)
        clf_list.append(clf)

        # Reshape the data to facilitate probability density polotting:
        xpdf = np.linspace(-2, 7, 1000)
        xpdf.shape = (xpdf.shape[0], 1)
        xpdf_list.append(xpdf)

        density = np.exp(clf.score(xpdf))
        density_list.append(density)

        # Generate plots:
        fig = plt.figure()

        plt.hist(datum, bins = 10, normed = True, color = palette[count], alpha = 1)
        plt.plot(xpdf, density, "r-")
        plt.xlim(-2, 7)
        fig.set_size_inches(12, 3.5)

        fig.suptitle("1-Dimensional Gaussian Mixture Model:\nFreezing and Motion Frames: Video {}".format(count + 1), fontsize=16)

        fig.subplots_adjust(top=header_space - .05)

        ax = plt.gca()

        ax.set_xlabel("distribution of {} data points".format(len(datum)), fontsize = 15)
        ax.set_ylabel("probability density", fontsize = 15)

        ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
        ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)

        patch = mpatches.Patch(facecolor="#D8D8D8", edgecolor = "red", linewidth = "1.5", label="Winning model:\n{} Underlying Gaussians\nBIC: {:.2f}".format(min_bics[count], clf.bic(datum)))
        plt.legend(handles=[patch], fontsize = 14, bbox_to_anchor=(1, 0.5), loc="center", framealpha = 1, borderaxespad=0., frameon = True, edgecolor = "inherit", handletextpad=.5)

        count += 1
        
        # Check the posterior probability of any number falling within each model's n Gaussians:
        
        posteriors = []
        
        def posterior_prob_check(num):
            # Check to see whether the user has entered the optional argument posterior_probability_check
            if posterior_probability_check != None:
                # Iterate over each clf.predict_proba np.array:
                for i in clf.predict_proba(num):
                    # Pull each value out of its respective array and append it to posteriors = []
                    for k in i:
                        posteriors.append(k)
            else:
                pass
            
        posterior_prob_check(posterior_probability_check)
                
        # Iterate over each underlying Gaussian distribution, plotting and printing summary statistics for each:
        print "Video {}:\n".format(count)
        for i in range(clf.n_components):
            pdf = clf.weights_[i] * stats.norm(clf.means_[i, 0], np.sqrt(clf.covars_[i, 0])).pdf(xpdf)
            print "Gaussian {}:".format(i+1), "AUC: {:.3f};".format(clf.weights_[i]), "Mean: %.4f;" % round(clf.means_[i], 4), "Covariance: %.3f;" % round(clf.covars_[i], 3), "Posterior probability of {}:".format(posterior_probability_check), "%.4f" % round(posteriors[i], 3)
            plt.fill(xpdf, pdf, facecolor = "grey", edgecolor= None, alpha=0.5)
            plt.xlim(-2, 7);
        print ""
        plt.show()
        print ""
            
Gaussian_mixture_model(diff_vector_logs, .75)


# REVISIT THIS! (Dan's note to self...) In the previous cell, perhaps I can take ALL values and calculate 
# their posterior probability of falling within the left-most Gaussian, and call THAT the probability of 
# freezing for any given frame (???)

# Now that we've plotted one-dimensional GMM, we will move on to two-dimensional k-means modeling (KMM), 
# which, like GMM, algorithmically fits around n specified centroids on a two-dimensional Cartesian plane.  Again, we'll use the winning BIC score to determine the "right" number of centroids.

# First, we will scatterplot some similarity values to check for divergence in their correlation.  
# As long as they're not highly correlated, they should be useful for KMM.

# In[137]:


### SCATTER-PLOT SOME SIMILARITY VALUES ###

palette = sns.cubehelix_palette(n_colors = len(max_vector_logs), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

count = 0

ssd_logs = []

# Get log-transformed SSD values:
for i in sum_sq_diff_list:
    ssd_logs_temp = []
    for k in i:
        ssd_logs_temp.append(np.log(k))
    ssd_logs.append(ssd_logs_temp)

for i in range(len(max_vector_logs)):
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 3), sharex=False, sharey=False, subplot_kw={'adjustable': 'box-forced'})
    fig.suptitle('Similarity Score Scatter Plots: Video {}'.format(count + 1), fontsize=15)
    fig.subplots_adjust(top=.8)
    ax = axes.ravel()

    ax[0].set_ylabel('log max filter score', fontsize = 12)
    ax[0].set_xlabel("SSD score", fontsize = 12)
    ax[0].scatter(ssd_logs[i], max_vector_logs[i], color = ["green", "orange"], s = 10, alpha = .25)

    ax[1].set_xlabel("log normal r-squared value", fontsize = 12)
    ax[1].set_ylabel('log max filter score', fontsize = 12)
    ax[1].scatter(log_transformed_normalized_r2_values[i], max_vector_logs[i], color = ["green", "blue"], s = 10, alpha = .25)

    ax[2].set_xlabel('log normal r-squared value', fontsize = 12)
    ax[2].set_ylabel("SSD score", fontsize = 12)
    ax[2].scatter(log_transformed_normalized_r2_values[i], ssd_logs[i], color = ["blue", "orange"], s = 10, alpha = .25)

    count += 1


# Now we must prepare the data for KMM (which, unlike GMM, requires at least x and y coordinates) by 
# zipping our similarity-value lists together.

# In[138]:


### ZIPPING SSD, LOG-NORM MAX FILTER, AND LOG-NORM R2 LISTS FOR K-MEANS CLUSTERING ###

ssd_finite = []
log_max_filter_finite = []
log_r2_finite = []

zipped_ssd_max = []
zipped_r2_max = []

zipped_lists = []

zipped_list_names = []

# Before zipping, we need to drop NaNs and infinite values; otherwise, they error-out later:
for i in ssd_logs:
    temp_ssd = []
    for j in i:
        if (np.isnan(j) == False and np.isfinite(j) == True):
            temp_ssd.append(j)
        else:
            pass
    ssd_finite.append(temp_ssd)

for i in max_vector_logs:
    temp_max_filter = []
    for j in i:
        if (np.isnan(j) == False and np.isfinite(j) == True):
            temp_max_filter.append(j)
        else:
            pass
    log_max_filter_finite.append(temp_max_filter)
    
for i in log_transformed_normalized_r2_values:
    temp_log_r2 = []
    for j in i:
        if (np.isnan(j) == False and np.isfinite(j) == True):
            temp_log_r2.append(j)
        else:
            pass
    log_r2_finite.append(temp_log_r2)

# Iterate over and zip the finite, NaN-free lists:
for i in range(len(ssd_logs)):
    zipped_ssd_max.append(zip(ssd_finite[i], log_max_filter_finite[i]))
zipped_lists.append(zipped_ssd_max)
zipped_list_names.append(["SSD", "log max filter score"])
    
for i in range(len(ssd_logs)):
    zipped_r2_max.append(zip(log_r2_finite[i], log_max_filter_finite[i]))
zipped_lists.append(zipped_r2_max)
zipped_list_names.append(["log-norm r-squared values", "log max filter score"])


# With the lists zipped, we can now fit KMM models to our data, with each video exhibiting the number of 
# centroids (i.e., groups within which the data best fit) determined by our BIC model comparisons a few cells back. 

# In[139]:


colors = ["#A9E5BB", "#00B0CB", '#F7DB2A', "#DB0000", "#7F5484"] * 10

count = 0

count_2 = 0

optimal_peaks_expanded = optimal_peaks * 10

zipped_list_names_expanded = zipped_list_names * 10

video_number = []

for k, v in enumerate(video_files):
    video_number.append(k + 1)
    
video_number_expanded = video_number * 10

for zipped_list in zipped_lists:
        
    for values in zipped_list:
        
        kmeans = KMeans(n_clusters = optimal_peaks_expanded[count])
        kmeans.fit(values)

        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        fig = plt.figure() 
        ax = fig.add_subplot(111)

        fig.suptitle("K-Means Clustering: {} Centroids (Video {})".format(optimal_peaks_expanded[count], video_number_expanded[count]), fontsize=14)
                
        if count_2 <= 3:
            x_lab = 0
        else:
            x_lab = 1
        
        for i in range(len(values)):
            plt.plot(values[i][0], values[i][1], colors[labels[i]], marker = ".", markersize = 8, alpha = 1, zorder = 1)
            ax.set_xlabel("{}".format(zipped_list_names_expanded[x_lab][0]), fontsize = 14)
            ax.set_ylabel("{}".format(zipped_list_names_expanded[0][1]), fontsize = 14)
            sns.despine()
        count_2 += 1

        # Show the centroids:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker = ".", color = "lightgrey", linewidths = 1.5, edgecolors = "black", s = 400, zorder = 2, label="k-means\ncentroid")
        plt.legend(fontsize = 12, edgecolor = "inherit", framealpha=0.5, bbox_to_anchor=(1.25, 0.5), loc="center", frameon = True)

        count += 1


# From here, researchers must determine how to interpret the models.  One way might be to consider all of the 
# data in the leftmost n-Gaussians in each video's winning GMM model to be indicative of freezing. Another might 
# be to use each video's KMM model's lower-leftmost centroid as the "freezing" cluster.  We are still exploring 
# options for the most sound interpretation method.

# # DIAGNOSTIC TOOLS (NO NEED TO GRADE BEYOND THIS POINT)
# 

# In[ ]:


# View a specified frame:
def frame_check(video, frame_number):
    display = VideoFileClip(video)
    return display.ipython_display(t=frame_number/30)
    
frame_check("motion_test.mov", 190)


# In[ ]:


# Map natural log values to their respective time points:

count = 0

for log in max_vector_logs:
    # Iterate over freezing_list to obtain and temp. store nlog values for plot overlays
    freezing_logs = []
    for freezing in freezing_list:
        freezing_log = np.log(freezing + 1 - min(freezing))
        freezing_logs.append(freezing_log)

    fig = plt.figure()
    fig.suptitle("Log-Transformed Scores (Video {})".format(count + 1), fontsize = 15)
    fig.set_size_inches(size)

    fig.subplots_adjust(top=header_space)
    
    ax = plt.gca()
    ax.set_xlabel("frame", fontsize = 15)
    ax.set_ylabel("natural log value", fontsize = 15)

    plt.plot(my_ordered_array_list[count], log, label = "motion score")
    plt.plot(my_ordered_array_list[count], freezing_logs[count], "g-", label = '"freezing" score (filtered)')

    ax.legend(bbox_to_anchor=(1.34, 0.5), loc="center", borderaxespad=0., fontsize = 14, frameon=False, handletextpad=.5)
    
    count += 1


# ### Back Converting Natural Logs to their Antilogs (DIAGNOSTIC PURPOSES ONLY)

# In[ ]:


### VERIFICATION THAT OUR NLOG CONVERSIONS ARE REVERSIBLE ###

# Map data BACK to video by un-logging.
# Need to ensure that the un-logging goes back far enough to get timestamped data, so that
# the data can be mapped to the video to see whether the mean of each distribution is a viable
# binary threshold for categorizing the animal as moving/not moving

logs_to_back_convert = max_vector_logs

# 		If you were given ln(x) = 1.3 then x = inverse natural log of 1.3 or the natural
# 		antilog of 1.3 or 
# 		x = e1.3=3.669

# The exponential function is e^x where e is a mathematical constant 
# called Euler's number, approximately 2.718281.

print min(max_vector)
print max(max_vector)
print max_vector[1]
test_log = np.log(max_vector[1] + 1 - min(max_vector))
print test_log
antilog = np.exp(test_log) - (1 - min(max_vector))
print antilog

unlogged_list = []

def unlogged(natural_logs):
    for log in natural_logs:
        unlogged_list.append(np.exp(log) - (1 - min(max_vector)))
    return(0)

unlogged(logs_to_back_convert)

# So, back-conversion works fine, and we can preserve order,
# but I need to think a bit more about what this has bought us.


# ## Video exporting begins here.
# ## Currently a lower priority--WE DO NOT NEED TO WORK ON THIS FOR PSC 290
# ## Eventually, need to work out a way to do this task iteratively.

# In[ ]:


video_to_export = VideoFileClip("motion_test.mov")


# In[ ]:


test_mask = freezing
test_mask = np.insert(test_mask, 0, 0.0) 
print test_mask[4]
def test_func(t):
    return test_mask[int(round(t*30))]*100 #How did we arrive at this?


# In[ ]:


print test_mask.shape


# In[ ]:


from moviepy.editor import *
from moviepy.video.tools.drawing import circle

clip = VideoFileClip("motion_test.mov", audio=False).add_mask()
clip2 = VideoFileClip("motion_test.mov", audio=False).add_mask()

w,h = clip.size

# This mask creates a persistent black circle.
clip.mask.get_frame = lambda t: circle(screensize=(clip.w,clip.h),
                                       center=(clip.w/10,clip.h/5),
                                       radius=50,col1=0, col2=1, blur=4)

# The mask is a circle with vanishing radius r(t) = 800-200*t that 
# conditionally "erases" the persistent circle when motion is detected.
clip2.mask.get_frame = lambda t: circle(screensize=(clip.w,clip.h),
                                       center=(clip.w/10,clip.h/5),
                                       radius=test_func(t),
                                       col1=1, col2=0, blur=4)

final = CompositeVideoClip([clip, clip2], size = clip.size)

final.write_videofile("motion_test_outcomeII.mp4")
