# Date: 2020-06-09
# Description: path planning algorithms and user interface
#-----------------------------------------------------------------------------

# Load external modules
import cv2 # OpenCV
import matplotlib.pyplot as plt # Matplotlib plotting functionality
from matplotlib.patches import Patch # graphics object
import numpy as np # Numpy toolbox
import os # access to Windows OS
from PIL import ImageTk,Image # TkInter-integrated image display functionality 
import tkinter as tk # TkInter UI backbone
import xlwt # Excel write functionality

# Hardcoded directory paths
dirPvars = '../vars/' # persistent variables directory

# Load persistent variables
h_npz_settings = np.load(dirPvars+'settings.npz',allow_pickle=True)
softwareName = str(h_npz_settings['softwareName'])
figSize = list(h_npz_settings['figSize'])
guiColor_black = h_npz_settings['guiColor_black']
guiColor_white = h_npz_settings['guiColor_white']
guiColor_offwhite = h_npz_settings['guiColor_offwhite']
guiColor_darkgreen = h_npz_settings['guiColor_darkgreen']
guiColor_cherryred = h_npz_settings['guiColor_cherryred']
guiFontSize_large = h_npz_settings['guiFontSize_large']
guiFontSize_small = h_npz_settings['guiFontSize_small']
guiFontType_normal = h_npz_settings['guiFontType_normal']
guiFontType_uniform = h_npz_settings['guiFontType_uniform']

#-----------------------------------------------------------------------------