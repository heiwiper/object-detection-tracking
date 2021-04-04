#                     .---------------.
# --------------------| Configuration |--------------------
#                     '---------------'
# Choose one of the 7 videos :
# 1 ==> backdoor
# 2 ==> bungalows
# 3 ==> highway
# 4 ==> office
# 5 ==> pedestians
# 6 ==> peopleInShade
# 7 ==> PETS2006
VIDEO = 3
#
#                       .-----------.
# ----------------------| Detection |----------------------
#                       '-----------'
# There are 2 detection algorithms :
# 1 ==> bakcground substraction
# 2 ==> HOG
DETECTION_ALGO = 2

# Parameters for the first detection algorithm :
#
# How many frames should we wait before performing
# background substraction
BG_HIST_THRESHOLD = 100
#
# Background substraction variable THRESHOLD
BG_THRESHOLD = 20
#
# If the area of a contour is below this value
# the contour is rejected, else it is accepted
AREA_THRESHOLD = 400

# Parameters for the second detection algorithm :
#
# There are two models :
# 1 ==> Car
# 2 ==> Pedestrian
HOG_MODEL = 1
#
# At which scale does the matching box change
SCALE_FACTOR = 1.05
#
# Model 1 has one extra parameter : 
# Minimum matching neighbors
MIN_NEIGHBORS = 13
#
# Model 2 has two extra parameters :
# Matching window padding
PADDING = (4, 4)
#
# Matching window size
WIN_STRIDE = (4, 4)
#                     .----------.
# --------------------| Tracking |--------------------
#                     '----------'
#
# There are 6 tracking algorithms :
# 1 ==> MOSSE
# 2 ==> KFC
# 3 ==> MIL
# 4 ==> CSRT
# 5 ==> GOTURN
# 6 ==> MedianFlow
TRACKING_ALGO = 1
#
# This variable specifies how many frames should we
# discard the object for not being detected or tracked
TRACK_HIST_THRESHOLD = 30

# -------------------------------------
