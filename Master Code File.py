
# coding: utf-8

# In[39]:
####hola!!!!
#Hola senora!

# If memory/processing speed becomes an issue, may need to launch Jupyter from terminal with 
# "jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000"

# Necessary imports:

import imageio
import itertools
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from scipy import ndimage, misc, stats
from scipy.ndimage.filters import minimum_filter
from matplotlib.font_manager import FontProperties
from sklearn.mixture import GMM
#from sklearn.cluster import KMeans # Probably won't need this one, which is for k-means clustering in 2-d space

# http://zulko.github.io/moviepy/
from moviepy.editor import *
import pygame
get_ipython().magic(u'matplotlib inline')
imageio.plugins.ffmpeg.download()


# In[3]:


### Style ###

# set graph size:
size = (6, 3.5)

# set distance between header and top of graph
header_space = 0.88 


# In[4]:


# RGB to grayscale function:
# https://stackoverflow.com/questions/12201577/how-can-i-convert-an-rgb-image-into-grayscale-in-python

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


# In[5]:


# Create a list of video files to score.
# VIDEOS MUST FIRST BE STORED IN YOUR JUPYTER ENVIRONMENT!

video_files = [VideoFileClip("motion_test.mov"), VideoFileClip("test_vid_2.mov"), VideoFileClip("test_vid_4.mov"), VideoFileClip("test_vid_5.mov")]


# In[6]:


sum_sq_diff_list = []

# Loop over each video to be scored:
for video in video_files:
    sum_sq_diff = []
    prev_frame = []
    total_frames = 0
    
    # Within each video, loop over individual frames, comparing each frame to its preceding frame:
    for frame in video.iter_frames(fps=30):
        this_frame = rgb2gray(frame)
        if prev_frame != []:
            total_frames += 1
            
            # Get the delta between contiguous frames:
            diff_frame = this_frame-prev_frame
            
            # Square each delta to eliminate negative values:
            diff_frame_sq = np.power(diff_frame, 2)
            sum_diff_frame_sq = np.sum( diff_frame_sq )
            sum_sq_diff.append( sum_diff_frame_sq )
            
        prev_frame = this_frame
        
    # Append the full list of squared deltas to a master list of 
    # squared deltas for each of the n videos to be scored:
    sum_sq_diff_list.append(sum_sq_diff)
    print total_frames
    
    # Plot some contiguous frames and delta frames from each video:
    f, axarr = plt.subplots(1,3)
    axarr[0].imshow(this_frame)
    axarr[1].imshow(prev_frame)
    axarr[2].imshow(diff_frame)
    
#print sum_sq_diff_list


# In[7]:


# Several lists generated in the following for loop will need to be saved as lists of sub-lists for later use:
sum_sq_diff_nparray_list = []
min_vectors_list = []
max_vectors_list = []
my_ordered_array_list = []

count = 0

for sum_of_sq_differences in sum_sq_diff_list:
    min_vectors = []    
    n = 10 # This can be changed to alter sensitivity of the filter. A lower value would be LESS sensitive to motion.
    sum_sq_diff_nparray = np.array(sum_of_sq_differences)
    sum_sq_diff_nparray = ((sum_sq_diff_nparray)/max(sum_sq_diff_nparray))*100

    # Use a sliding window of n-frames to calculate the highest and lowest
    # deltas among those frames. Store those values as max and min vectors.
    # Large divergence in max/min values indicates motion.
    min_vector = minimum_filter(sum_sq_diff_nparray, size=n, mode='constant')
    max_vector = minimum_filter(sum_sq_diff_nparray*-1, size=n, mode='constant')*-1

    print np.mean(min_vector)
    my_ordered_array = np.arange(0,len(sum_of_sq_differences))

    # Generate plots:
    fig = plt.figure()
    fig.set_size_inches(size)
    fig.suptitle('Frame-by-Frame Filter Scores (Video {})'.format(count + 1), fontsize = 15)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=header_space)

    ax.set_xlabel('frame', fontsize = 15)
    ax.set_ylabel('filter scores', fontsize = 15)

    plt.plot(my_ordered_array, min_vector, 'bx', label = "minimum filter")
    plt.plot(my_ordered_array, max_vector, 'gx', label = "maximum filter")
    
    ax.legend(bbox_to_anchor=(1.25, 0.5), loc="center", borderaxespad=0., fontsize = 14, frameon=False, handletextpad=.5)
    
    # Store necessary data in their respective lists for later use:
    sum_sq_diff_nparray_list.append(sum_sq_diff_nparray)
    min_vectors_list.append(min_vector)
    my_ordered_array_list.append(my_ordered_array)
    max_vectors_list.append(max_vector)
    
    count += 1


# ### Conceptual problem 1:
# 
# ### What is the "right" value for n in the previous cell? Are there characteristics of each video that can be recursively fed into this line to dynamically set an appropriate n, for instance relative to an animal's average speed?
# 

# In[8]:


freezing_list = []

#freezing_threshold_list = [] # We can reactivate this line if we set a dynamic freezing threshold

count = 0

for min_vector in min_vectors_list:
    # Create a min_vector-length array of zeros:
    freezing = np.zeros_like(min_vector)
    
    # Filter-out noise. 
    # Lower values would be more sensitive to motion.
    # Give it a try by changing the threshold to a 
    # different value and re-examining the plots.
    freezing_threshold = 4
    freezing[min_vector < freezing_threshold] = 1*max(min_vector)

    # Generate plots:
    fig = plt.figure()
    fig.set_size_inches(size)
    fig.suptitle('Filtered Motion Score (Video {})'.format(count +1), fontsize=15)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=header_space)

    ax.set_xlabel('frame', fontsize = 15)
    ax.set_ylabel('filter scores', fontsize = 15)

    plt.plot(my_ordered_array_list[count], min_vector, 'b-', label = "minimum vector")
    plt.plot(my_ordered_array_list[count], freezing, "g-", label = '"freezing" score')

    ax.legend(bbox_to_anchor=(1.25, 0.5), loc="center", borderaxespad=0., fontsize = 14, frameon=False, handletextpad=.5)
    
    # Store freezing values for each of the n videos as a list of sub-lists, for later use:
    freezing_list.append(freezing)
    
    #freezing_threshold_list.append(freezing_threshold) # We can reactivate this line if we set a dynamic freezing threshold
    
    count += 1


# In[9]:


# Plot the maximum vector scores from each video as a histogram to visualize
# the ostensibly bimodal distribution of scores that indicate freezing/motion: 

palette = sns.cubehelix_palette(n_colors = len(max_vectors_list), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

count = 0

for max_vector in max_vectors_list:
    fig = plt.figure()
    fig.suptitle('Distribution of Maximum Filter Scores (Video{})'.format(count + 1), fontsize=15)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=header_space)
    fig.set_size_inches(size)

    ax.set_xlabel('maximum filter score', fontsize = 15)
    ax.set_ylabel('frames', fontsize = 15)

    plt.hist(max_vector, bins=50, color = palette[count]);
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    
    count += 1


# In[13]:


### LOG TRANSFORMED PLOTS ###
# Transforming the data to their natural logs will facilitate better visualization and analysis.

max_vector_logs = []

palette = sns.cubehelix_palette(n_colors = len(max_vectors_list), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

count = 0

for max_vector in max_vectors_list:
    # Log-transform the data with an added constant to avoid negative values:
    max_vector_log = np.log(max_vector + 1 - min(max_vector))
    max_vector_logs.append(max_vector_log)

    # Generate log-transformed plots:
    fig = plt.figure()
    fig.suptitle('Distribution of Log-Transformed Maximum Filter Scores (Video {})'.format(count + 1), fontsize=15)
    fig.set_size_inches(size)

    ax = fig.add_subplot(111)
    fig.subplots_adjust(top=header_space)

    ax.set_xlabel('log-transformed maximum filter score', fontsize = 15)
    ax.set_ylabel('frames', fontsize = 15)

    plt.hist(max_vector_log, color = palette[count])
    plt.gca().set_ylim(bottom=0)
    plt.gca().set_xlim(left=0)
    
    count += 1


# In[392]:


palette = sns.cubehelix_palette(n_colors = len(max_vectors_list), start = 2.8, rot = -.1, dark = .25, light = .75, reverse = True)

count = 0

# Save some values to use later:
xpdf_list = []
density_list = []
clf_list = []

# Set the number of Gaussians to be used in our model:
peaks = 3

# Iterate over max_vector_logs to generate histograms and composite Gaussians
for max_vector_log in max_vector_logs:
    
    # Reshape the data to facilitate Gaussian modeling:
    max_vector_log.shape = (max_vector_log.shape[0],1)
    
    # Set the Gaussian mixed model to fit 2 Gaussians;
    # conduct 300 iterations to determine the best fit:
    clf = GMM(peaks, n_iter = 300).fit(max_vector_log)
    clf_list.append(clf)
    
    # Reshape the data to facilitate probability density polotting:
    xpdf = np.linspace(-2, 7, 1000)
    xpdf.shape = (xpdf.shape[0], 1)
    xpdf_list.append(xpdf)
    
    density = np.exp(clf.score(xpdf))
    density_list.append(density)

    # Generate plots:
    fig = plt.figure()

    plt.hist(max_vector_log, 10, normed = True, color = palette[count], alpha = 1)
    plt.plot(xpdf, density, "r-")
    plt.xlim(-2, 7)
    fig.set_size_inches(size)
    
    fig.suptitle("1-Dimensional Gaussian Mixture Model:\nFreezing and Motion Frames (Video {})".format(count + 1), fontsize=16)

    fig.subplots_adjust(top=header_space - .05)

    ax = plt.gca()

    ax.set_xlabel("Log-Transformed Maximum Filter Score", fontsize = 15)
    ax.set_ylabel("Probability Density", fontsize = 15)
    
    ax.tick_params(axis = 'both', which = 'major', labelsize = 12)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 12)
        
    patch = mpatches.Patch(facecolor="#D8D8D8", edgecolor = "red", linewidth = "1.5", label="{} Underlying Gaussians".format(peaks))
    plt.legend(handles=[patch], fontsize = 14, bbox_to_anchor=(1.35, 0.5), loc="center", frameon = False)
    
    count += 1
        
    # Iterate over each underlying Gaussian distribution, plotting and printing summary statistics for each:
    print "Video {}:\n".format(count)
    for i in range(clf.n_components):
        pdf = clf.weights_[i] * stats.norm(clf.means_[i, 0], np.sqrt(clf.covars_[i, 0])).pdf(xpdf)
        print "Gaussian {}:".format(i+1), "AUC: {:.3f};".format(clf.weights_[i]), "Mean: %.4f;" % round(clf.means_[i], 4), "Covariance: %.3f" % round(clf.covars_[i], 3)
        plt.fill(xpdf, pdf, facecolor = "grey", edgecolor= None, alpha=0.5)
        plt.xlim(-2, 7);
    
    print "Sum of AUC weights:", sum(clf.weights_)
    print "AIC: {:.2f}".format(clf.aic(max_vector_log))
    print "BIC: {:.2f}\n".format(clf.bic(max_vector_log))

    # Check the posterior probability of any number falling within each model's n Gaussians:
    def posterior_prob_check(num): 
        for i in clf.predict_proba(num):
            count2 = 1
            for k in i:
                print "Posterior probability of {} falling under Gaussian {} = {:.4f}".format(num, count2, k)
                count2 += 1
                
    posterior_prob_check(0.5)
    print ""
    #break
    
          
# get weights, means, sd for each; get AIC (should be for each composite, I think?):
# http://ogrisel.github.io/scikit-learn.org/sklearn-tutorial/modules/generated/sklearn.mixture.GMM.html


# In[393]:


### MODEL COMPARISONS ###

count = 1

min_aics = []
min_bics = []

# Iterate over the lists of log-transformed max vector scores:
for max_vector_log in max_vector_logs:
        
    # Reshape the data to facilitate Gaussian modeling:
    max_vector_log.shape = (max_vector_log.shape[0],1)
    
    # Test models for each video against a range of GMMs with between 1 and 10 underlying Gaussians:
    n_estimators = np.arange(1, 10)
    clfs = [GMM(n, n_iter=1000).fit(max_vector_log) for n in n_estimators]
    aics = [clf.aic(max_vector_log) for clf in clfs]
    bics = [clf.bic(max_vector_log) for clf in clfs]
    
    #Grab the lowest AIC and BIC scores from each video's model trials.
    # The lowest scores correspond to the optimal number of Gaussians to use
    # per video, so we can grab each of those by indexing the aics/bics lists
    # for lowest values (and then adding 1 to account for zero-indexing):
    min_aics.append(aics.index(min(aics))+1)
    min_bics.append(bics.index(min(bics))+1)

    # Plot the AIC and BIC scores:
    fig = plt.figure() 
    ax = fig.add_subplot(111)  
    
    fig.suptitle("Comparing n-Gaussian Models\n via AIC and BIC Testing: (Video {})".format(count), fontsize=16)

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


# In[15]:


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
    
    
#################################################################################
#################################################################################
########## NO NEED TO PROCEED PAST THIS POINT FOR OUR PSC 290 PROJECT############
#################################################################################
#################################################################################



# ## Back Converting Natural Logs to their Antilogs:

# In[10]:


### THERE IS NO NEED TO RUN THIS CODE. THIS IS JUST VERIFICATION THAT OUR NLOG CONVERSIONS ARE REVERSIBLE. ###

# # Map data BACK to video by un-logging.
# # Need to ensure that the un-logging goes back far enough to get timestamped data, so that
# # the data can be mapped to the video to see whether the mean of each distribution is a viable
# # binary threshold for categorizing the animal as moving/not moving

# logs_to_back_convert = max_vector_logs

# # 		If you were given ln(x) = 1.3 then x = inverse natural log of 1.3 or the natural
# # 		antilog of 1.3 or 
# # 		x = e1.3=3.669

# # The exponential function is e^x where e is a mathematical constant 
# # called Euler's number, approximately 2.718281.

# print min(max_vector)
# print max(max_vector)
# print max_vector[1]
# test_log = np.log(max_vector[1] + 1 - min(max_vector))
# print test_log
# antilog = np.exp(test_log) - (1 - min(max_vector))
# print antilog

# unlogged_list = []

# def unlogged(natural_logs):
#     for log in natural_logs:
#         unlogged_list.append(np.exp(log) - (1 - min(max_vector)))
#     return(0)

# unlogged(logs_to_back_convert)

# # So, back-conversion works fine, and we can preserve order,
# # but I need to think a bit more about what this has bought us.


# In[38]:


# freezing_logs = []

# for freezing in freezing_list:
#     freezing_log = np.log(freezing + 1 - min(freezing))
#     freezing_logs.append(freezing_log)
    
# len(freezing_logs)
# print len(freezing_logs[1])

# print freezing_logs[1]
# print max_vector_logs[1]


# ## Video exporting begins here.
# ## Currently a lower priority. 
# ## Eventually, need to work out a way to do this task iteratively.

# In[31]:


video_to_export = VideoFileClip("motion_test.mov")


# In[17]:


test_mask = freezing
test_mask = np.insert(test_mask, 0, 0.0) 
print test_mask[4]
def test_func(t):
    return test_mask[int(round(t*30))]*100 #How did we arrive at this?


# In[33]:


print test_mask.shape


# In[18]:


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


# In[128]:


a = 5
b = 10
print "1: ", "a + b" # Python treats this as a literal string of characters, not as variables or math.
print "2: ", a + b # Python treats this as a math operation performed on two variables, which store integers.
print "3: ", a, "a + b" # Here, we're concatenating a variable (which stores an integer) and a character string.
print "4: ", str(a + b) # Here, we're doing a math operation on two variables, then storing the output as a string.

