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
maxVel = 100 # [ft/s] maximum robot velocity

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

## Create the figure
#h_fig = plt.figure('Wildbot''s Autonomous Path Planner',[15,15],facecolor='w')
#ax = plt.gca() # get current axis
#ax.axis('image') # set axis type
#ax.set_xlim((0,I.shape[1]))        
#ax.set_ylim((I.shape[0],0))
#
## Generate tick marks and grid lines
#numyticks = 5 # number of ticks to show along the x-axis
#numxticks = 10 # number of ticks to show along the y-axis
#scale_pf = I.shape[0]/fieldy # [pix/ft] recover the scaling between the image and real life
#scale_pi = scale_pf/12 # [pix/inches]
#fieldy_pixels = I.shape[0] # [pixels] width of the field in pixels
#xticks_num = (np.round(np.linspace(0,I.shape[1],numxticks+1)/scale_pf)).astype(int)
#yticks_num = (np.round(np.linspace(0,I.shape[0],numyticks+1)/scale_pf)).astype(int)
#xticks_str = []
#yticks_str = []
#for i in range(0,len(xticks_num),1):
#    xticks_str.append(str(xticks_num[i]))
#for i in range(0,len(yticks_num),1):
#    yticks_str.append(str(yticks_num[i])) 
#yticks_str = np.flip(yticks_str,0)
#ax.set_xticklabels(xticks_str)
#ax.set_yticklabels(yticks_str)
#xstep = (scale_pf*fieldx/(numxticks))
#ystep = (scale_pf*fieldy/(numyticks))
#xticklocs = np.zeros((numxticks+1),float)
#yticklocs = np.zeros((numyticks+1),float)
#xticklocs[0:numxticks] = np.arange(0,I.shape[1]-0.1,xstep)
#yticklocs[0:numyticks] = np.arange(0,I.shape[0]-0.1,ystep)
#xticklocs[numxticks] = I.shape[1]
#yticklocs[numyticks] = I.shape[0]
#ax.set_xticks(xticklocs,minor=False)
#ax.set_yticks(yticklocs,minor=False)
#plt.grid(b=None,which='major',axis='both',color=np.array([0.8,0.8,0.8]))
#
## Label the axes    
#plt.xlabel('Length (ft)')
#plt.ylabel('Width (ft)')
#        
## Display the image        
#h_im = plt.imshow(I_color,extent=[0,I.shape[1],I.shape[0],0])
#plt.title('Left-Click to Select Waypoints',fontsize=24)
#plt.draw()
#
## Have the user select the waypoints
#[ptxs_way,ptys_way,vels_way,oris_way] = safePtSelection(h_fig,ax,'o',100)
#
## Convert the selected point coordinates to real units
#ptxs_way = np.array(ptxs_way)/scale_pi # [in]
#ptys_way = (fieldy_pixels-np.array(ptys_way))/scale_pi # [in]
#vels_way = 12.0*np.array(vels_way) # [in/s]
#
## Show waypoints
#print(ptxs_way)
#print(ptys_way)
#print(vels_way)
#print(oris_way)

ptxs_way = np.array([ 64.11,  96.94, 148.74, 180.65, 196.83, 195.  , 195.  , 195.  ,
       214.87, 243.54, 258.33, 264.81])
ptys_way = np.array([190.  , 190.  , 190.  , 204.4 , 229.37, 247.86, 264.51, 284.86,
       295.96, 284.4 , 256.19, 226.13])
vels_way = 12*np.array([0., 2., 10., 10., 10., 9., 8., 7., 6., 10., 5., 0.])

#------------------------------------------------------------------------------

# Insert dense points along a piece-wise linear interpolation
delta_s = 1 # [inches] step length along the path
vels_dense = [] # [in/s] dense v points along the original path
ptxs_segs = [] # [inches] the dense x points listed per segment
ptys_segs = [] # [inches] the dense y points listed per segment

for i in range(0,len(ptxs_way)-1,1):
    
    ptxs_seg = []
    ptys_seg = []
    
    delta_x = ptxs_way[i+1]-ptxs_way[i]
    delta_y = ptys_way[i+1]-ptys_way[i]
    angleboi = np.arctan2(delta_y,delta_x)
    delta_s_tally = 0
    dist_x = np.square(ptxs_way[i+1] - ptxs_way[i])
    dist_y = np.square(ptys_way[i+1] - ptys_way[i])
    pts_dist = np.sqrt(dist_x + dist_y)
 
    while delta_s_tally < pts_dist:
        delta_s_tally = delta_s_tally + delta_s
        delta_little_x = np.cos(angleboi)*delta_s_tally
        delta_little_y = delta_s_tally*np.sin(angleboi)
        
        current_x = ptxs_way[i] + delta_little_x
        current_y = ptys_way[i] + delta_little_y
        
        ptxs_seg.append(current_x)
        ptys_seg.append(current_y)
        #*** fake velocity
        vels_dense.append(0.0)
        

    
    
    
    # Store the dense points segement-wise (needed for smoothing)
    ptxs_segs.append(ptxs_seg)
    ptys_segs.append(ptys_seg)
    
#------------------------------------------------------------------------------
    
# Smooth the path

r = 24 # [inches] minimum turn radius *** could scale as f(v)

ptxs_smooth = []
ptys_smooth = []
vels_smooth = []
aNum_start = 0 # start point for the current segment
slope = ""
circleNum = -1
arc_ptsx = []
arc_ptsy = []
for segNum in range(0,(len(ptxs_segs)-1),1):
    
    # Initialize
   
    m_y = ptys_way[segNum + 1] - ptys_way[segNum]
    m_x = ptxs_way[segNum + 1] - ptxs_way[segNum]
    
    delta_x = ptxs_way[segNum+1]-ptxs_way[segNum]
    delta_y = ptys_way[segNum+1]-ptys_way[segNum]
    
    angleboi = np.arctan2(delta_y,delta_x)
    delta_x_circle = np.sin(angleboi) * r 
    delta_y_circle = np.cos(angleboi) * r
    
    if m_x == 0: 
        slope = "undef"
    else:
        m = m_y / m_x
        if m > 0:
            slope = "positive"
        if m < 0:
            slope = "negative"
        if m == 0:
            slope = "0"
    for point in range(aNum_start, len(ptxs_segs[segNum]) - 1, 1):
       
        
        if slope == "positive":
            circle1_x = ptxs_segs[segNum][point] + delta_x_circle
            circle1_y = ptys_segs[segNum][point] - delta_y_circle
            circle2_x = ptxs_segs[segNum][point] - delta_x_circle
            circle2_y = ptys_segs[segNum][point] + delta_y_circle
        elif slope == "negative":
            circle1_x = ptxs_segs[segNum][point] + delta_x_circle
            circle1_y = ptys_segs[segNum][point] + delta_y_circle
            circle2_x = ptxs_segs[segNum][point] - delta_x_circle
            circle2_y = ptys_segs[segNum][point] - delta_y_circle
        elif slope == "0":
            circle1_x = ptxs_segs[segNum][point]
            circle1_y = ptys_segs[segNum][point] + delta_y_circle
            circle2_x = ptxs_segs[segNum][point]
            circle2_y = ptys_segs[segNum][point] - delta_y_circle
        elif slope == "undef":
            circle1_x = ptxs_segs[segNum][point] + delta_x_circle
            circle1_y = ptys_segs[segNum][point]
            circle2_x = ptxs_segs[segNum][point] - delta_x_circle
            circle2_y = ptys_segs[segNum][point] 

        
        for g in range(0, len(ptxs_segs[segNum + 1]) - 1, 1):
         
            c_dist_x_1 = np.square(ptxs_segs[segNum + 1][g] - circle1_x)
            c_dist_y_1 = np.square(ptys_segs[segNum + 1][g] - circle1_y)
            c_dist_1 = np.sqrt(c_dist_x_1 + c_dist_y_1)
            
            c_dist_x_2 = np.square(ptxs_segs[segNum + 1][g] - circle2_x)
            c_dist_y_2 = np.square(ptys_segs[segNum + 1][g] - circle2_y)
            c_dist_2 = np.sqrt(c_dist_x_2 + c_dist_y_2)
            
        ptCount = 0
        flag_corner = False # False if not reached a corner yet
        if((c_dist_1 <= r)|(c_dist_2 <= r)):
            flag_corner = True
            
            if(c_dist_1 <= r):
                circleNum = 1
                h  = circle1_x
                k = circle1_y
            elif (c_dist_2 <= r):
                circleNum = 2
                h = circle2_x
                k = circle2_y
            pointx = ptxs_segs[segNum][point]
            pointy = ptys_segs[segNum][point]
#            numofpoints = (len(ptxs_segs[segNum]) - segNum) + (g+1)
            numofpoints =  (len(ptxs_segs[segNum]) - point) + g
            
            
#            for z in range(point, len(ptxs_segs[segNum]) - 1, 1):
#                ptCount = ptCount + 1
#            print(2*ptCount)
##                
#            
#            for b in range(0,ptCount,1):
#                vels_smooth.append(1000)
#                ptxs_smooth.append(ptxs_segs[segNum + 1][b])
#                ptys_smooth.append(ptys_segs[segNum + 1][b])
#                aNum_start = ptCount - 1
            
            phia = np.arctan2(ptys_segs[segNum][point] - k, ptxs_segs[segNum][point] - h)
            if(phia<0): phia += 2.0*np.pi
            phib = np.arctan2(ptys_segs[segNum + 1][g] - k, ptxs_segs[segNum + 1][g] -h)
            if(phib<0): phib += 2.0*np.pi
            
#            phia = phia * (180 / np.pi)
#            phib = phib * (180 / np.pi)   
       
   
            
            if(phia >= phib):
                delta_phi1 = phia - phib
                delta_phi2 = (2.0*np.pi) - phia + phib
                if(delta_phi1<=delta_phi2):
                    delta_phi = delta_phi1
                    arcDir = "clockwise"
                else:
                    delta_phi = delta_phi2
                    arcDir = "counter clockwise"
            else:
                
                delta_phi1 = phib - phia
                delta_phi2 = (2.0*np.pi) - phib + phia
                if(delta_phi1 <= delta_phi2):
                    delta_phi = delta_phi1
                    arcDir = "counter clockwise"
                else:
                    delta_phi = delta_phi2
                    arcDir = "clockwise"
                
            
           
            tally_phi = 0
            arc_ptx = []
            arc_pty = []
#            phia = phia * (np.pi / 180)
#            phib = phib * (np.pi / 180)
#            delta_phi1 = delta_phi1 *(np.pi / 180)
#            delta_phi2 = delta_phi2 * (np.pi / 180)
#            delta_phi = delta_phi * (np.pi /180)
            phi_step = delta_phi / numofpoints
            print(arcDir)
            
            if(arcDir == "clockwise"):
                
#                for h in range(int(phia), int(phib), int(phi_step)):
#                    phi_c = np.pi - (phia - tally_phi)
#                    delta_arc_x = r * np.cos(phi_c)
#                    delta_arc_y = r * np.sin(phi_c)
#                    
#                    arc_x = h + delta_arc_x
#                    arc_y = k + delta_arc_y
#                
#                    arc_ptx.append(arc_x)
#                    arc_pty.append(arc_y)
#                    
#                    tally_phi = tally_phi + phi_step
                
                for e in range(0, numofpoints, 1):
                    
                    phi_c = (phia - tally_phi)
                    delta_arc_x = r * np.cos(phi_c)
                    delta_arc_y = r * np.sin(phi_c)
                    
                    arc_x = h + delta_arc_x
                    arc_y = k + delta_arc_y
                  
                    tally_phi = tally_phi + phi_step
                    arc_ptx.append(arc_x)
                    arc_pty.append(arc_y)
                    
                    
                 
                
            elif(arcDir == "counter clockwise"):
                for f in range(0,2 * numofpoints, 1):
                     
                    phi_c = np.pi - (phia + tally_phi)
                    delta_arc_x = r * np.cos(phi_c)
                    delta_arc_y = r * np.sin(phi_c)
        
        
                    arc_x = h - delta_arc_x
                    arc_y = k + delta_arc_y
                    
                    arc_ptx.append(arc_x)
                    arc_pty.append(arc_y)
                    tally_phi = tally_phi + phi_step
                    
            arc_ptsx.append(arc_ptx)
            arc_ptsy.append(arc_pty)
            ptCount = 0
    
            for a in range(0, len(ptxs_segs[segNum]) - point, 1):
                    
                vels_smooth.append(1000)
                ptxs_smooth.append(arc_ptx[a])
                ptys_smooth.append(arc_pty[a])
                    
                ptCount = ptCount + 1
            
            for b in range(0,ptCount,1):
                vels_smooth.append(1000)
                ptxs_smooth.append(arc_ptx[b])
                ptys_smooth.append(arc_pty[b])
                aNum_start = (ptCount )- 1
                     
            break

        else:
            vels_smooth.append(0)
        
        
        
        if(flag_corner == False):
            ptxs_smooth.append(ptxs_segs[segNum][point])
            ptys_smooth.append(ptys_segs[segNum][point])
            aNum_start = 0
        else:
            aNum_start = ptCount - 1  
            
        

            
            
        # Calculate distance from a point in the next segment to the circle centers
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            

# Include the last line segment in the smoothed path
for i in range(aNum_start,len(ptxs_segs[-1]),1):
    ptxs_smooth.append(ptxs_segs[-1][i])
    ptys_smooth.append(ptys_segs[-1][i])
    vels_smooth.append(0.0)
ptxs_smooth.append(ptxs_way[-1])
ptys_smooth.append(ptys_way[-1])
vels_smooth.append(0.0)



#------------------------------------------------------------------------------

# Calculate the distance, time, and curvature along the path
#***

#------------------------------------------------------------------------------

#*** should also plot original waypoints

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

# Display the path
for i in range(0,len(ptxs_smooth),1):
    ptColor = plt.cm.plasma(vels_smooth[i]/(12*maxVel))
    ptColor = np.array([ptColor[0],ptColor[1],ptColor[2]])
    ax.scatter(ptxs_smooth[i]*scale_pi,fieldy_pixels-(ptys_smooth[i]*scale_pi),color=ptColor,marker='.',s=50)

#------------------------------------------------------------------------------



























