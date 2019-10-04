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
import scipy.interpolate as interp
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

def textEntryWidget(title,posx_init,posy_init):
    """ Creates a blocking text entry widget
    Args:
        title: name of the entry box
    Returns:
        user_input: the user input as a string
    Saves:
        None
    """
    
    global posx, posy, velocity, orientation
    def widgetClose(*args):
        global posx, posy, velocity, orientation
        posx = entryFieldPosX.get()
        posy = entryFieldPosY.get()
        velocity = entryFieldVelocity.get()
        orientation = entryFieldOrientation.get()
        textwindow.quit()
        textwindow.destroy()
    
    # Open the text entry widget
    textwindow = tk.Tk()
    textwindow.title(title)
    textwindow.geometry('400x400')
    textwindow.configure(background='#%02x%02x%02x' % ((245,245,245)))
    
    # Configure to handle use of the Windows close button
    textwindow.protocol('WM_DELETE_WINDOW',widgetClose)
    
    # Create, pack, and display the widget
    entryFieldPosX = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    entryFieldPosY = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    entryFieldVelocity = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    entryFieldOrientation = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    labelFieldPosX = tk.Label(textwindow,text='X Position (in)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),height=1,width=30,anchor='w')
    labelFieldPosY = tk.Label(textwindow,text='Y Position (in)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),height=1,width=30,anchor='w')  
    labelFieldVelocity = tk.Label(textwindow,text='Velocity (ft/s)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),height=1,width=30,anchor='w')
    labelFieldOrientation = tk.Label(textwindow,text='Orientation (deg)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),height=1,width=30,anchor='w')         
    labelFieldPosX.pack()                              
    entryFieldPosX.pack()
    labelFieldPosY.pack()
    entryFieldPosY.pack()
    labelFieldVelocity.pack()
    entryFieldVelocity.pack()
    labelFieldOrientation.pack()
    entryFieldOrientation.pack()
    entryFieldPosX.focus_set()
    posx_init = posx_init/scale_pi;
    posy_init = (fieldy_pixels-posy_init)/scale_pi;
    entryFieldPosX.insert(0,'%0.2f' %(posx_init))
    entryFieldPosY.insert(0,'%0.2f' %(posy_init))
    textwindow.mainloop()
    
    return posx, posy, velocity, orientation

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
    oris = [] # initialize
    while(flag_abort==False):
        pt = ginput(1,show_clicks=True,timeout=0.25) # check for click
        if(len(pt)!=0): 
            
            # Ask for the velocity
            [posx,posy,vel,ori] = textEntryWidget('Target Velocity (ft/s)',pt[0][0],pt[0][1])
            try: 
                posx = np.float(posx)
                posx = posx*scale_pi
            except: posx = pt[0][0]
            try: 
                posy = np.float(posy)
                posy = fieldy_pixels-posy*scale_pi
            except: posy = pt[0][1]
            try: vel = np.float(vel)
            except: vel = 0.0
            try: ori = np.float(ori)
            except: ori = 'ahead'
            
            # Append the user inputs
            ptxs.append(posx)
            ptys.append(posy)
            vels.append(vel)
            oris.append(ori)
            
            # Determine the color for the waypoint dot
            ptColor = plt.cm.plasma(vel/maxVel)
            ptColor = np.array([ptColor[0],ptColor[1],ptColor[2]])
            ax.scatter(posx,posy,color=ptColor,marker=mkrType,s=mkrSize)
            
    return ptxs, ptys, vels, oris

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
scale_pf = I.shape[0]/fieldy # [pix/ft] recover the scaling between the image and real life
scale_pi = scale_pf/12 # [pix/inches]
fieldy_pixels = I.shape[0] # [pixels] width of the field in pixels
xticks_num = (np.round(np.linspace(0,I.shape[1],numxticks+1)/scale_pf)).astype(int)
yticks_num = (np.round(np.linspace(0,I.shape[0],numyticks+1)/scale_pf)).astype(int)
xticks_str = []
yticks_str = []
for i in range(0,len(xticks_num),1):
    xticks_str.append(str(xticks_num[i]))
for i in range(0,len(yticks_num),1):
    yticks_str.append(str(yticks_num[i])) 
yticks_str = np.flip(yticks_str,0)
ax.set_xticklabels(xticks_str)
ax.set_yticklabels(yticks_str)
xstep = (scale_pf*fieldx/(numxticks))
ystep = (scale_pf*fieldy/(numyticks))
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
[ptxs,ptys,vels,oris] = safePtSelection(h_fig,ax,'o',100)

# Convert the selected point coordinates to real units
ptxs = np.array(ptxs)/scale_pi # [in]
ptys = (fieldy_pixels-np.array(ptys))/scale_pi # [in]
vels = 12.0*np.array(vels) # [in/s]

# Show waypoints
print(ptxs)
print(ptys)
print(vels)
print(oris)

# Cubic spline interpolation
interp.CubicSpline()

#*** also need to specify robot orientation along with velocity









