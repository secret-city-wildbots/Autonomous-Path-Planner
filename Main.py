# Date: 2019-10-4
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
maxVel = 15 # [ft/s] maximum robot velocity

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
[ptxs_way,ptys_way,vels_way,oris_way] = safePtSelection(h_fig,ax,'o',100)

# Convert the selected point coordinates to real units
ptxs_way = np.array(ptxs_way)/scale_pi # [in]
ptys_way = (fieldy_pixels-np.array(ptys_way))/scale_pi # [in]
vels_way = 12.0*np.array(vels_way) # [in/s]

#ptxs_way = np.array([ 64.11,  96.94, 148.74, 180.65, 196.83, 195.  , 195.  , 195.  ,
#       214.87, 243.54, 258.33, 264.81])
#ptys_way = np.array([190.  , 190.  , 190.  , 204.4 , 229.37, 247.86, 264.51, 284.86,
#       295.96, 284.4 , 256.19, 226.13])
#vels_way = 12*np.array([0., 2., 10., 15., 15., 15., 8., 8., 8., 8., 4., 0.])

#------------------------------------------------------------------------------

# Insert dense points along a piece-wise linear interpolation
delta_s = 1 # [inches] step length along the path
vels_segs = [] # [in/s] the dense v points listed per segment
ptxs_segs = [] # [inches] the dense x points listed per segment
ptys_segs = [] # [inches] the dense y points listed per segment
for i in range(0,len(ptxs_way)-1,1):
    
    # Initialize
    ptxs_seg = []
    ptys_seg = []
    vels_seg = []
    
    # Calculate the length of the path segment
    max_s = np.sqrt(((ptys_way[i+1]-ptys_way[i])**2)+((ptxs_way[i+1]-ptxs_way[i])**2)) 
    
    # Calculate the steps for the current path segment 
    gamma = np.arctan2((ptys_way[i+1]-ptys_way[i]),(ptxs_way[i+1]-ptxs_way[i]))
    delta_x = delta_s*np.cos(gamma)
    delta_y = delta_s*np.sin(gamma)
    delta_v = (vels_way[i+1]-vels_way[i])/max_s
    
    # Include the current waypoint
    ptxs_seg.append(ptxs_way[i]) 
    ptys_seg.append(ptys_way[i]) 
    vels_seg.append(vels_way[i])
    
    # Step the inserted points along the path
    total_s = 0 # tracks total travel along the current path segment
    
    while((total_s+delta_s)<max_s):
        
        ptxs_seg.append(ptxs_seg[-1]+delta_x)
        ptys_seg.append(ptys_seg[-1]+delta_y)
        vels_seg.append(vels_seg[-1]+delta_v)
        total_s += delta_s 
        
    # Store the dense points segement-wise (needed for smoothing)
    ptxs_segs.append(ptxs_seg)
    ptys_segs.append(ptys_seg)
    vels_segs.append(vels_seg)
        
#------------------------------------------------------------------------------
    
# Smooth the path
r_min = 12 # [inches] turn radius at v = 0
r_max = 100 # [inches] turn radius at v = maxVel
ptxs_smooth = []
ptys_smooth = []
vels_smooth = []
aNum_start = 0 # start point for the current segment
for segNum in range(0,(len(ptxs_segs)-1),1):
    
    # Initialize
    flag_corner = False # False if not reached a corner yet
    
    # Calculate the minimum turn radius as function of velocity
    v_wayb = vels_way[segNum+1]
    r = v_wayb*(r_max-r_min)/(maxVel*12) + r_min
    max_sa = np.sqrt(((ptys_way[segNum+1]-ptys_way[segNum])**2)+((ptxs_way[segNum+1]-ptxs_way[segNum])**2)) # length of segment
    max_sb = np.sqrt(((ptys_way[segNum+2]-ptys_way[segNum+1])**2)+((ptxs_way[segNum+2]-ptxs_way[segNum+1])**2)) # length of next segment
    r = min(r,min(0.9*max_sa,0.9*max_sb)) # prevent the radius from exceeding the segment length
    
    # Calculate (Q-I) gamma 
    gamma = np.arctan2(np.abs((ptys_way[segNum+1]-ptys_way[segNum])),np.abs((ptxs_way[segNum+1]-ptxs_way[segNum])))
    
    # Calculate the perpendicular offsets from the current segment
    delta_x = r*np.sin(gamma)
    delta_y = r*np.cos(gamma)
    
    # Determine the slope of the current line segment
    try:
        m = (ptys_way[segNum+1]-ptys_way[segNum])/(ptxs_way[segNum+1]-ptxs_way[segNum])
        if(m>=0.0): seg_slope = 'positive'
        else: seg_slope = 'negative'
    except: seg_slope = 'positive' # catch vertical slope
    
    for aNum in range(aNum_start,len(ptxs_segs[segNum]),1):
        
        # Calculate the centers (h,k) of the correct tangent circles
        xa = ptxs_segs[segNum][aNum] # current y point in the current segment
        ya = ptys_segs[segNum][aNum] # current x point in the current segment
        va = vels_segs[segNum][aNum] # current velocity in the current segment
        if(seg_slope=='positive'):
            h1 = xa + delta_x
            k1 = ya - delta_y
            h2 = xa - delta_x
            k2 = ya + delta_y
        elif(seg_slope=='negative'):
            h1 = xa + delta_x
            k1 = ya + delta_y
            h2 = xa - delta_x
            k2 = ya - delta_y
            
        for bNum in range(0,len(ptxs_segs[segNum+1]),1):
            
            # Calculate distance from a point in the next segment to the circle centers
            xb = ptxs_segs[segNum+1][bNum]
            yb = ptys_segs[segNum+1][bNum]
            d1 = np.sqrt(((h1-xb)**2)+((k1-yb)**2))
            d2 = np.sqrt(((h2-xb)**2)+((k2-yb)**2))
            
            # Check if the point in the next segment is inside of one of the tangent circles
            if((d1<=r)|(d2<=r)):
                
                # Determine the number of points which must be smoothed
                numPtsSmooth = (len(ptxs_segs[segNum]) - aNum) + (bNum+1)
                
                # Check which candidate circle is the correct one
                if(d1<=r):
                    h = h1
                    k = k1
                else:
                    h = h2
                    k = k2
                    
                # Calculate the angles at intercept
                phia = np.arctan2((ya-k),(xa-h))
                if(phia<0): phia += 2.0*np.pi
                phib = np.arctan2((yb-k),(xb-h))
                if(phib<0): phib += 2.0*np.pi
                
                # Choose the interior angle
                if(phia>=phib):
                    delta_phi1 = phia-phib
                    delta_phi2 = 2.0*np.pi-phia+phib
                    if(delta_phi1<=delta_phi2):
                        delta_phi = delta_phi1
                        arcDir = 'clockwise'
                    else:
                        delta_phi = delta_phi2
                        arcDir = 'anticlockwise'
                else:
                    delta_phi1 = phib-phia
                    delta_phi2 = 2.0*np.pi-phib+phia
                    if(delta_phi1<=delta_phi2):
                        delta_phi = delta_phi1
                        arcDir = 'anticlockwise'
                    else:
                        delta_phi = delta_phi2
                        arcDir = 'clockwise'
                
                # Calculate the points along the arc
                phi_step = delta_phi/(numPtsSmooth-1)
                phis = []
                if(arcDir=='clockwise'):
                    # Initerior is clockwise
                    phi_moving = phia
                    for q in range(0,numPtsSmooth,1):
                        phis.append(phi_moving)
                        phi_moving -= phi_step
                else:
                    # Interior is anti-clockwise
                    phi_moving = phia
                    for q in range(0,numPtsSmooth,1):
                        phis.append(phi_moving)
                        phi_moving += phi_step
                
                # Smooth points in current segment
                phis_index = 0
                for i in range(aNum,len(ptxs_segs[segNum]),1):
                    xa_smooth = h + r*np.cos(phis[phis_index])
                    ya_smooth = k + r*np.sin(phis[phis_index])
                    ptxs_smooth.append(xa_smooth)
                    ptys_smooth.append(ya_smooth)
                    va = vels_segs[segNum][i]
                    vels_smooth.append(va)
                    phis_index += 1
                
                # Smooth points in next segment
                for i in range(0,(bNum+1),1):
                    xb_smooth = h + r*np.cos(phis[phis_index])
                    yb_smooth = k + r*np.sin(phis[phis_index])
                    ptxs_smooth.append(xb_smooth)
                    ptys_smooth.append(yb_smooth)
                    vb = vels_segs[segNum+1][i]
                    vels_smooth.append(vb)
                    phis_index += 1
                
                # Set corner flag
                flag_corner = True
                break
                
        # Add the current point in the current segment to the smoothed path if it's chill
        if(flag_corner==False):
            ptxs_smooth.append(xa)
            ptys_smooth.append(ya)
            vels_smooth.append(va)
            aNum_start = 0 # start the next segment at the beginning
        else: 
            aNum_start = bNum+1 # start the next segment after the radius
            break

# Include the last line segment in the smoothed path
for i in range(aNum_start,len(ptxs_segs[-1]),1):
    ptxs_smooth.append(ptxs_segs[-1][i])
    ptys_smooth.append(ptys_segs[-1][i])
    vels_smooth.append(0.0)
ptxs_smooth.append(ptxs_way[-1])
ptys_smooth.append(ptys_way[-1])
vels_smooth.append(0.0)



#------------------------------------------------------------------------------

# Limit velocity changes to max acceleration

#------------------------------------------------------------------------------

# Calculate the distance and time, and orientation along the path
#***

#------------------------------------------------------------------------------

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
plt.draw()

# Display the waypoints
for i in range(0,len(ptxs_way),1):
    ax.scatter(ptxs_way[i]*scale_pi,fieldy_pixels-(ptys_way[i]*scale_pi),color='g',marker='x',s=100)

# Display the path
for i in range(0,len(ptxs_smooth),1):
    ptColor = plt.cm.plasma(vels_smooth[i]/(12*maxVel))
    ptColor = np.array([ptColor[0],ptColor[1],ptColor[2]])
    ax.scatter(ptxs_smooth[i]*scale_pi,fieldy_pixels-(ptys_smooth[i]*scale_pi),color=ptColor,marker='.',s=50)

#------------------------------------------------------------------------------



























