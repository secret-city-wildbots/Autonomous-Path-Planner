# Date: 2019-09-30
# Author: Secret City Wildbots
# Description: allows the user to load an image of the field and generate a 
#              planned path for the robot to follow
#------------------------------------------------------------------------------

# Load external libraries
import cv2 # OpenCV
import matplotlib.pyplot as plt # plotting functionality
import numpy as np # math operations
from pylab import ginput # get user mouse inputs
import tkinter as tk # file UI
root = tk.Tk() # suppress the default window
root.withdraw() # suppress the default window
from tkinter import filedialog # file UI
#from tkinter import messagebox # file UI

# Change Environment settings (some are only applicable if running in an iPython console)
try:
    from IPython import get_ipython # needed to run magic commands
    ipython = get_ipython() # needed to run magic commands
    ipython.magic('matplotlib qt') # display figures in a separate window
except: pass
plt.rcParams.update({'font.size': 24}) # change the default font size for plots

#------------------------------------------------------------------------------

# Constants
fieldx = 54.0 # [feet] field length
fieldy = 27.0 # [feet] field width
res = 100 # controls the resolution of the displayed image
maxVel = 25 # [ft/s] maximum robot velocity

#------------------------------------------------------------------------------

def textEntryWidget(title):
    """ Creates a blocking text entry widget
    Args:
        title: name of the entry box
    Returns:
        user_input: the user input as a string
    Saves:
        None
    """
    
    global user_input
    def widgetClose(*args):
        global user_input
        user_input = entryField.get()
        textwindow.quit()
        textwindow.destroy()
    
    # Open the text entry widget
    textwindow = tk.Tk()
    textwindow.title(title)
    textwindow.geometry('400x50')
    textwindow.configure(background='#%02x%02x%02x' % ((245,245,245)))
    
    # Configure to handle use of the Windows close button
    textwindow.protocol('WM_DELETE_WINDOW',widgetClose)
    
    # Create, pack, and display the widget
    entryField = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    entryField.pack()
    entryField.focus_set()
    textwindow.mainloop()
    
    return user_input

def safePtSelection(h_fig,ax,mkrType,mkrSize):
    """" Allows the user to select points on a figure window without closing the window failing silently
    Args:
        h_fig: handle for the relevant figure
        ax: handle for the relevant figure axis
        mkrType: shape/style of the point display marker
        mkrSize: size of the point display marker 
    Returns:
        pts: 2D array of selected points with each row of format [x,y]
        h_pts: handle for the plotted points
    """

    # Watch for window close event
    global flag_abort
    flag_abort = False
    def actionWindowClose(evt):
        global flag_abort
        flag_abort = True
    h_fig.canvas.mpl_connect('close_event', actionWindowClose) 
    
    # Let the user select the requested number of points
    ptxs = [] # initialize
    ptys = [] # initialize
    vels = [] # initialize
    while(flag_abort==False):
        pt = ginput(1,show_clicks=True,timeout=0.25) # check for click
        if(len(pt)!=0): 
            
            # Ask for the velocity
            vel = textEntryWidget('Target Velocity (ft/s)')
            try: vel = np.float(vel)
            except: vel = 0.0
            
            # Append the user inputs
            ptxs.append(pt[0][0])
            ptys.append(pt[0][1])
            vels.append(vel)
            
            # Determine the color for the waypoint dot
            ptColor = plt.cm.plasma(vel/maxVel)
            ptColor = np.array([ptColor[0],ptColor[1],ptColor[2]])
            ax.scatter(pt[0][0],pt[0][1],color=ptColor,marker=mkrType,s=mkrSize)
            
    # Reshape list into 2D numpy array
    
    return ptxs, ptys, vels

#------------------------------------------------------------------------------

# Load the field image
path_I = filedialog.askopenfilename(initialdir='',title="Select a Cropped Field Diagram",filetypes=[('Images','*.jpg *.jpeg *.png *.tif *.tiff')]);
I = cv2.imread(path_I,cv2.IMREAD_GRAYSCALE) # load the selected image 
I = cv2.resize(I,(int(res*fieldx),int(res*fieldy))) # resize the image

# Correct image color
I_color = np.zeros((I.shape[0],I.shape[1],3),np.uint8)
if(len(I.shape)==2):
    I_color[:,:,0] = I # set red channel
    I_color[:,:,1] = I # set green channel
    I_color[:,:,2] = I # set blue channel
else:
    I_color[:,:,0] = I[:,:,2] # set red channel
    I_color[:,:,1] = I[:,:,1] # set green channel
    I_color[:,:,2] = I[:,:,0] # set blue channel

# Create the figure
h_fig = plt.figure('Wildbot''s Autonomous Path Planner',[15,15],facecolor='w')
ax = plt.gca() # get current axis
ax.axis('image') # set axis type
ax.set_xlim((0,I.shape[1]))        
ax.set_ylim((I.shape[0],0))

# Generate tick marks and grid lines
numyticks = 5 # number of ticks to show along the x-axis
numxticks = 10 # number of ticks to show along the y-axis
scale = I.shape[0]/fieldy # [pix/ft] recover the scaling between the image and real life
xticks_num = (np.round(np.linspace(0,I.shape[1],numxticks+1)/scale)).astype(int)
yticks_num = (np.round(np.linspace(0,I.shape[0],numyticks+1)/scale)).astype(int)
xticks_str = []
yticks_str = []
for i in range(0,len(xticks_num),1):
    xticks_str.append(str(xticks_num[i]))
for i in range(0,len(yticks_num),1):
    yticks_str.append(str(yticks_num[i])) 
yticks_str = np.flip(yticks_str,0)
ax.set_xticklabels(xticks_str)
ax.set_yticklabels(yticks_str)
xstep = (scale*fieldx/(numxticks))
ystep = (scale*fieldy/(numyticks))
xticklocs = np.zeros((numxticks+1),float)
yticklocs = np.zeros((numyticks+1),float)
xticklocs[0:numxticks] = np.arange(0,I.shape[1]-0.1,xstep)
yticklocs[0:numyticks] = np.arange(0,I.shape[0]-0.1,ystep)
xticklocs[numxticks] = I.shape[1]
yticklocs[numyticks] = I.shape[0]
ax.set_xticks(xticklocs,minor=False)
ax.set_yticks(yticklocs,minor=False)
plt.grid(b=None,which='major',axis='both',color=np.array([0.8,0.8,0.8]))

# Label the axes    
plt.xlabel('Length (ft)')
plt.ylabel('Width (ft)')
        
# Display the image        
h_im = plt.imshow(I_color,extent=[0,I.shape[1],I.shape[0],0])
plt.title('Left-Click to Select Waypoints',fontsize=24)
plt.draw()

# Have the user select the waypoints
[ptxs,ptys,vels] = safePtSelection(h_fig,ax,'o',100)
print(ptxs)
print(ptys)
print(vels)

# Convert the selected point coordinates to real units
#*** remember to flip y value


#*** also need to specify robot orientation along with velocity


