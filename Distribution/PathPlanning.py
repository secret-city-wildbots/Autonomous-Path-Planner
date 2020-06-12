# Date: 2020-06-11
# Description: path planning algorithms and user interface
#-----------------------------------------------------------------------------

# Load external modules
import cv2 # OpenCV
import matplotlib.pyplot as plt # Matplotlib plotting functionality
import numpy as np # Numpy toolbox
import pandas # data handling toolbox
import sys # access to Windows OS
import tkinter as tk # TkInter UI backbone

# Load remaining user modules
import GeneralSupportFunctions as gensup # general support functions

# Hardcoded directory paths
dirPvars = '../vars/' # persistent variables directory

# Load persistent variables
h_npz_settings = np.load(dirPvars+'settings.npz',allow_pickle=True)
softwareName = str(h_npz_settings['softwareName'])
figSize = list(h_npz_settings['figSize'])
dispRes = float(h_npz_settings['dispRes'])
guiColor_black = h_npz_settings['guiColor_black']
guiColor_white = h_npz_settings['guiColor_white']
guiColor_offwhite = h_npz_settings['guiColor_offwhite']
guiColor_darkgreen = h_npz_settings['guiColor_darkgreen']
guiColor_cherryred = h_npz_settings['guiColor_cherryred']
guiColor_hotgreen = h_npz_settings['guiColor_hotgreen']
guiColor_hotyellow = h_npz_settings['guiColor_hotyellow']
guiFontSize_large = h_npz_settings['guiFontSize_large']
guiFontSize_small = h_npz_settings['guiFontSize_small']
guiFontType_normal = h_npz_settings['guiFontType_normal']
guiFontType_uniform = h_npz_settings['guiFontType_uniform']

#-----------------------------------------------------------------------------

def generatePath(path):
    """
    Generates a smooth path based on a set of waypoints
    Args:
        path: robot path 
    Returns:
        path: the updated robot path
    """
    
    # Extract information from the path
    ways_x = path.ways_x
    ways_y = path.ways_y
    ways_v = path.ways_v
    ways_o = path.ways_o
    
    # Insert dense points along a piece-wise linear interpolation
    vels_segs = [] # [in/s] the dense v points listed per segment
    oris_segs = [] # [degrees] the dense orientation points list per segment
    ptxs_segs = [] # [inches] the dense x points listed per segment
    ptys_segs = [] # [inches] the dense y points listed per segment
    for i in range(0,path.numWayPoints()-1,1):
        
        # Initialize
        ptxs_seg = []
        ptys_seg = []
        vels_seg = []
        oris_seg = []
        
        # Calculate the length of the path segment
        max_s = np.sqrt(((ways_y[i+1]-ways_y[i])**2)+((ways_x[i+1]-ways_x[i])**2)) 
        
        # Calculate the steps for the current path segment 
        gamma = np.arctan2((ways_y[i+1]-ways_y[i]),(ways_x[i+1]-ways_x[i]))
        delta_x = path.step_size*np.cos(gamma)
        delta_y = path.step_size*np.sin(gamma)
        delta_v = (ways_v[i+1]-ways_v[i])/max_s
        if(ways_o[i+1]>ways_o[i]):
            delta_oseg1 = ways_o[i+1]-ways_o[i]
            delta_oseg2 = 360 - delta_oseg1
            if(delta_oseg1<delta_oseg2): delta_o = delta_oseg1/max_s
            else: delta_o = -delta_oseg2/max_s
        else:
            delta_oseg1 = ways_o[i]-ways_o[i+1]
            delta_oseg2 = 360 - delta_oseg1
            if(delta_oseg1<delta_oseg2): delta_o = -delta_oseg1/max_s
            else: delta_o = delta_oseg2/max_s
        
        # Include the current waypoint
        ptxs_seg.append(ways_x[i]) 
        ptys_seg.append(ways_y[i]) 
        vels_seg.append(ways_v[i])
        oris_seg.append(ways_o[i])
        
        # Step the inserted points along the path
        total_s = 0 # tracks total travel along the current path segment
        while((total_s+path.step_size)<max_s):
            
            ptxs_seg.append(ptxs_seg[-1]+delta_x)
            ptys_seg.append(ptys_seg[-1]+delta_y)
            vels_seg.append(vels_seg[-1]+delta_v)
            oris_seg.append((oris_seg[-1]+delta_o)%360)
            total_s += path.step_size 
            
        # Store the dense points segement-wise (needed for smoothing)
        ptxs_segs.append(ptxs_seg)
        ptys_segs.append(ptys_seg)
        vels_segs.append(vels_seg)
        oris_segs.append(oris_seg)
            
    # Smooth the path
    ptxs_smooth = []
    ptys_smooth = []
    vels_smooth = []
    oris_smooth = []
    aNum_start = 0 # start point for the current segment
    for segNum in range(0,(len(ptxs_segs)-1),1):
        
        # Initialize
        flag_corner = False # False if not reached a corner yet
        
        # Calculate the minimum turn radius as function of velocity
        v_wayb = ways_v[segNum+1]
        r = v_wayb*(path.radius_max-path.radius_min)/(path.v_max) + path.radius_min
        max_sa = np.sqrt(((ways_y[segNum+1]-ways_y[segNum])**2)+((ways_x[segNum+1]-ways_x[segNum])**2)) # length of segment
        max_sb = np.sqrt(((ways_y[segNum+2]-ways_y[segNum+1])**2)+((ways_x[segNum+2]-ways_x[segNum+1])**2)) # length of next segment
        r = min(r,min(0.9*max_sa,0.9*max_sb)) # prevent the radius from exceeding the segment length
        
        # Calculate (Q-I) gamma 
        gamma = np.arctan2(np.abs((ways_y[segNum+1]-ways_y[segNum])),np.abs((ways_x[segNum+1]-ways_x[segNum])))
        
        # Calculate the perpendicular offsets from the current segment
        delta_x = r*np.sin(gamma)
        delta_y = r*np.cos(gamma)
        
        # Determine the slope of the current line segment
        numer = (ways_y[segNum+1]-ways_y[segNum])
        denom = (ways_x[segNum+1]-ways_x[segNum])
        if(denom!=0): 
            m = numer/denom
            if(m>=0.0): seg_slope = 'positive'
            else: seg_slope = 'negative'
        else:
            seg_slope = 'positive' # catch vertical slope
        
        for aNum in range(aNum_start,len(ptxs_segs[segNum]),1):
            
            # Calculate the centers (h,k) of the correct tangent circles
            xa = ptxs_segs[segNum][aNum] # current y point in the current segment
            ya = ptys_segs[segNum][aNum] # current x point in the current segment
            va = vels_segs[segNum][aNum] # current velocity in the current segment
            oa = oris_segs[segNum][aNum] # current orientation in the current segment
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
                        oa = oris_segs[segNum][i]
                        oris_smooth.append(oa)
                        phis_index += 1
                    
                    # Smooth points in next segment
                    for i in range(0,(bNum+1),1):
                        xb_smooth = h + r*np.cos(phis[phis_index])
                        yb_smooth = k + r*np.sin(phis[phis_index])
                        ptxs_smooth.append(xb_smooth)
                        ptys_smooth.append(yb_smooth)
                        vb = vels_segs[segNum+1][i]
                        vels_smooth.append(vb)
                        ob = oris_segs[segNum+1][i]
                        oris_smooth.append(ob)
                        phis_index += 1
                    
                    # Set corner flag
                    flag_corner = True
                    break
                    
            # Add the current point in the current segment to the smoothed path if it's chill
            if(flag_corner==False):
                ptxs_smooth.append(xa)
                ptys_smooth.append(ya)
                vels_smooth.append(va)
                oris_smooth.append(oa)
                aNum_start = 0 # start the next segment at the beginning
            else: 
                aNum_start = bNum+1 # start the next segment after the radius
                break
    
    # Include the last line segment and the last waypoint in the smoothed path
    for i in range(aNum_start,len(ptxs_segs[-1]),1):
        ptxs_smooth.append(ptxs_segs[-1][i])
        ptys_smooth.append(ptys_segs[-1][i])
        vels_smooth.append(vels_segs[-1][i])
        oris_smooth.append(oris_segs[-1][i])
    ptxs_smooth.append(ways_x[-1])
    ptys_smooth.append(ways_y[-1])
    vels_smooth.append(ways_v[-1])
    oris_smooth.append(ways_o[-1])
    
    # Calculate the distance and time, and orientation along the path
    dsts_smooth = [] # initialize
    tims_smooth = [] # initialize
    dsts_smooth.append(0.0) # first point
    tims_smooth.append(0.0) # first point
    d_total = 0 # initialize
    t_total = 0 # initialize
    for i in range(0,len(ptxs_smooth)-1,1):
        
        # Pull the coordinates for current baby segment
        xa = ptxs_smooth[i]
        xb = ptxs_smooth[i+1]
        ya = ptys_smooth[i]
        yb = ptys_smooth[i+1]
        va = vels_smooth[i]
        vb = vels_smooth[i+1]
        
        # Calculate the distance along the path
        d = np.sqrt(((xb-xa)**2)+((yb-ya)**2))
        d_total += d
        dsts_smooth.append(d_total)
        
        # Calculate the times along the path
        v_avg = 0.5*(va+vb)
        t = d/v_avg
        t_total += t
        tims_smooth.append(t_total)
        
        # # Calculate the along-path orientations [180,-180]
        # theta = 180.0*np.arctan2((yb-ya),(xb-xa))/np.pi
        
    # Update the path
    path.updateSmoothPath(ptxs_smooth,ptys_smooth,vels_smooth,oris_smooth,dsts_smooth,tims_smooth)
    
    return path

#-----------------------------------------------------------------------------

def popupPtData(path,x_prior,y_prior):
    """
    Creates a pop-up window that allows a user to create or edit a waypoint
    Args:
        path: robot path 
        x_prior: (pixels) the x coordinate of the user's mouse click
        y_prior: (pixels) the y coordinate of the user's mouse click
    Returns:
        path: the updated robot path
    """
    
    # Configure waypoint selection
    [x_init,y_init,v_init,o_init,way_index] = path.configureWayPoint(x_prior,y_prior)
    
    # Define button callbacks
    def actionClose(*args):
        
        # userEntry = entry0.get()
        popwindow.unbind('<Return>')
        popwindow.quit()
        popwindow.destroy()
        
    def actionSave(*args):
        
        # Check entries for errors
        flags = True
        [x_way,flags] = gensup.safeTextEntry(flags,textFields[0]['field'],'float',vmin=0.0,vmax=path.field_x_real)
        [y_way,flags] = gensup.safeTextEntry(flags,textFields[1]['field'],'float',vmin=0.0,vmax=path.field_y_real)
        [v_way,flags] = gensup.safeTextEntry(flags,textFields[2]['field'],'float',vmin=0.0,vmax=path.v_max/12.0)
        [o_way,flags] = gensup.safeTextEntry(flags,textFields[3]['field'],'float',vmin=0.0,vmax=360.0)
    
        # Save the error-free entries in the correct units
        if(flags):
             
            # Add way point
            path.addWayPoint(x_way,y_way,v_way*12,o_way,way_index)
            
            # Close the popup
            actionClose()
        
    def actionDelete(*args):
        
        # Remove a way point
        path.removeWayPoint(way_index)
        actionClose()
    
    # Define the popup window
    popwindow = tk.Toplevel()
    popwindow.title('Waypoint')
    windW = 250 # window width
    windH = 425 # window height 
    popwindow.geometry(str(windW)+'x'+str(windH))
    popwindow.configure(background=guiColor_offwhite)
    popwindow.resizable(width=False, height=False)
    
    # Set the initial window location
    popwindow.geometry("+{}+{}".format(int(0.5*(popwindow.winfo_screenwidth()-windW)),int(0.5*(popwindow.winfo_screenheight()-windH))))
    
    # Configure to handle use of the Windows close button
    popwindow.protocol('WM_DELETE_WINDOW',actionClose)
    
    # Define the fields
    fieldNames = ['X Position (in)',
                  'Y Position (in)',
                  'Velocity (ft/s)',
                  'Orientation (deg) [0-360]']
    defaults = [x_init,
                y_init,
                v_init,
                o_init]
    
    # Set up the elements
    textFields = []
    for i in range(0,len(fieldNames),1):
        [title,field] = gensup.easyTextField(popwindow,windW,fieldNames[i],str(defaults[i]))
        textFields.append({'title': title, 'field': field})
    if(way_index==-1): buttonName = 'Create'
    else: buttonName = 'Edit'
    buttonSave = tk.Button(popwindow,text=buttonName,fg=guiColor_black,bg=guiColor_hotgreen,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionSave)
    popwindow.bind('<Return>',actionSave)
    buttonDelete = tk.Button(popwindow,text='Delete',fg=guiColor_black,bg=guiColor_hotyellow,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionDelete)
    
    # Place all elements
    for i in range(0,len(textFields),1):
        textFields[i]['title'].pack(fill='both')
        textFields[i]['field'].pack()
    buttonSave.pack(pady=10)
    buttonDelete.pack(pady=10)
    
    # Run the GUI
    popwindow.mainloop()
    
    return path

#-----------------------------------------------------------------------------

def loadRobot(file_robot,scale_pi):
    """
    Loads the robot model
    Args:
        file_robot: file path to the robot model
        scale_pi: (pix/in) unit conversion factor for the field
    Returns:
        I_robot: image of the robot model
    """

    try:

        # Load the robot model
        I_robot = cv2.imread(file_robot,cv2.IMREAD_COLOR)
        I_robot = gensup.convertColorSpace(I_robot) # fix image coloring
    
        # Recover the robot dimensions
        filename = file_robot.split('/')[-1]
        filename = filename.split('.')[0]
        dimensions = filename.split('_')
        robot_x = float(dimensions[1].replace('-','.')) # (in) robot length
        robot_y = float(dimensions[2].replace('-','.')) # (in) robot width
        robot_x = int(np.round(robot_x*scale_pi,0)) # (pix)
        robot_y = int(np.round(robot_y*scale_pi,0)) # (pix)
    
        # Resize the robot model
        I_robot = cv2.resize(I_robot,(robot_x,robot_y)) # resize the image
        
        # Pad the robot model for later rotation
        pad_v = I_robot.shape[0]//2
        pad_h = I_robot.shape[1]//2
        I_robot = cv2.copyMakeBorder(I_robot,pad_v,pad_v,pad_h,pad_h,cv2.BORDER_CONSTANT,value=[255,255,255])
        
    except:
        
        # default robot model
        I_robot = 255*np.ones((10,10,3),np.uint8)

    return I_robot

#-----------------------------------------------------------------------------

def overlayRobot(I_in,I_robot_in,scale_pi,field_y_pixels,center_x,center_y,theta):
    """
    Overlays the robot model on top of the field image
    Args:
        I_in: image of the field map
        I_robot_in: image of the robot model
        scale_pi: (pix/in) unit conversion factor for the field
        field_y_pixels: (pix) width of the field
        center_x: (in) x coordinate of the robot center
        center_y: (in) y coordinate of the robot center
        theta: (deg) rotation angle for the robot
    Returns:
        I: fused image of the field map and the robot model
    """
    
    # Handle OpenCV ghost assignments
    I = np.copy(I_in)
    I_robot = np.copy(I_robot_in)
    
    # Convert the coordinates into pixels
    center_x = scale_pi*center_x
    center_y = field_y_pixels - scale_pi*center_y
    tx = center_x - 0.5*I_robot.shape[1] # shift for the center of the robot
    ty = center_y - 0.5*I_robot.shape[0] # shift for the center of the robot
    tx = int(np.round(tx,0))
    ty = int(np.round(ty,0))
    
    # Rotate the robot model about its center
    M_rot = cv2.getRotationMatrix2D((I_robot.shape[1]//2,I_robot.shape[0]//2),theta,1.0)
    I_robot = cv2.warpAffine(I_robot,M_rot,(I.shape[1],I.shape[0]))
    
    # Translate the robot model
    M_trans = np.float32([[1,0,tx],[0,1,ty]])
    I_robot = cv2.warpAffine(I_robot,M_trans,(I.shape[1],I.shape[0]))
    
    # Find the background pixel locations and make them transparent in the robot image
    locs_background = (I_robot[:,:,0]>=225) & (I_robot[:,:,1]>=225) & (I_robot[:,:,2]>=225)
    I_robot[locs_background,0] = 0
    I_robot[locs_background,1] = 0
    I_robot[locs_background,2] = 0
    
    # Find the foreground pixel locations and make them transparent in the field image
    locs_foreground = (I_robot[:,:,0]>=30) | (I_robot[:,:,1]>=30) | (I_robot[:,:,2]>=30)
    I[locs_foreground,0] = 0
    I[locs_foreground,1] = 0
    I[locs_foreground,2] = 0
    
    # Fuse the two images
    I = I + I_robot
    
    return I

#-----------------------------------------------------------------------------

def definePath(path,file_I,file_robot):
    """
    Allows the user to define an autonomous path
    Args:
        path: robot path
        file_I: file path to the field map image
        file_robot: file path to the robot model
    Returns:
        None
    """
    
    # Load the field image
    I = cv2.imread(file_I,cv2.IMREAD_COLOR) # load the selected image 
    I = cv2.resize(I,(int(dispRes*path.field_x_real),int(dispRes*path.field_y_real))) # resize the image
    I = gensup.convertColorSpace(I) # fix image coloring
    
    # Calculate the scaling 
    path.fieldScale(I)
    
    # Load the robot model
    I_robot = loadRobot(file_robot,path.scale_pi)
    
    # Display the field image
    [h_fig,h_im,ax,_] = gensup.smartRealImageDisplay(I,[path.field_x_real/12.0,path.field_y_real/12.0],'Field Map',
                                                      bannertext='Right-Click to Select Waypoints',flag_grid=True,
                                                      origin='bottomleft',x_real='X: Down Field',y_real='Y: Side Field',
                                                      units='ft')
    
    # Define global variables
    global flag_abort
    flag_abort = False
    global flag_risingEdge
    flag_risingEdge = False
    global add_state
    add_state = 2 # intialize in this state to display a loaded path
    
    # Watch for window close event
    def actionWindowClose(evt):
        global flag_abort
        flag_abort = True
    h_fig.canvas.mpl_connect('close_event', actionWindowClose) 
    
    # Set up mouse button click callbacks
    def mouseClick(event):
        global flag_risingEdge,add_state
        
        sys.stdout.flush()
        button = 'RIGHT'
        if(str(event.button).find(button)!=-1):
            if(flag_risingEdge==False):
                flag_risingEdge = True
                if(add_state==0):
                
                    # Block additional popups
                    add_state = 1
                    
                    # Retrieve information about the selected point
                    [x_prior,y_prior] = event.xdata, event.ydata 
                    
                    # Display popup window
                    popupPtData(path,x_prior,y_prior)
                    add_state = 2
                
    # Set up mouse button unclick callbacks
    def mouseUnClick(event):
        global flag_risingEdge
        sys.stdout.flush()
        button = 'RIGHT'
        if(str(event.button).find(button)!=-1):
            if(flag_risingEdge==True):
                flag_risingEdge = False
    
    # Connect the mouse button callbacks
    plt.connect('button_press_event',mouseClick)
    plt.connect('button_release_event',mouseUnClick)
    
    # Blocking loop
    h_ways = None
    h_smooths = None
    hs_ori = []
    while(flag_abort==False):
        plt.pause(0.1)
        if(add_state==2):
            
            # Remove the previous plots
            try: 
                
                h_ways.remove()
                h_smooths.remove()
                for h in hs_ori: h[0].remove()
                hs_ori = []
            except: pass
        
            if(path.numWayPoints()>0):
        
                # Add the robot model at the starting waypoint
                I_fused = overlayRobot(I,I_robot,path.scale_pi,path.field_y_pixels,path.ways_x[0],path.ways_y[0],path.ways_o[0])
                
            if(path.numWayPoints()>1):
        
                # Add the robot model at the ending waypoint
                I_fused = overlayRobot(I_fused,I_robot,path.scale_pi,path.field_y_pixels,path.ways_x[-1],path.ways_y[-1],path.ways_o[-1])
                
            if(path.numWayPoints()>0):    
                
                # Display the robot at the starting and end points
                h_im.remove()
                h_im = ax.imshow(I_fused,vmin=0,vmax=255,interpolation='none',extent=[0,I.shape[1],I.shape[0],0])  
        
            if(path.numWayPoints()>0):
            
                # Update the way point plots
                h_ways = ax.scatter(path.scale_pi*np.array(path.ways_x),(path.field_y_pixels)*np.ones((len(path.ways_y)),float) - path.scale_pi*np.array(path.ways_y),
                                    facecolors='none',edgecolors='r',marker='o',s=400)
            
            if(path.numWayPoints()>1):
                
                # Calculate the path
                path = generatePath(path)
                
                # Display the path metrics
                details_start = 'start at (%0.2f in, %0.2f in, %0.0f°)' %(path.smooths_x[0],path.smooths_y[0],path.smooths_o[0])
                details_end = 'end at (%0.2f in, %0.2f in, %0.0f°) predicted travel time: %0.2f s' %(path.smooths_x[-1],path.smooths_y[-1],path.smooths_o[-1],path.total_t)
                plt.xlabel('X: Down Field\n%s\n%s' %(details_start,details_end))
                
                # Display the smooth path
                ptColors = []
                for i in range(0,path.numSmoothPoints(),1):
                    ptColor = plt.cm.plasma(path.smooths_v[i]/path.v_max)
                    ptColors.append(np.array([ptColor[0],ptColor[1],ptColor[2]]))
                h_smooths = ax.scatter(path.scale_pi*np.array(path.smooths_x),
                                       (path.field_y_pixels)*np.ones((len(path.smooths_y)),float) - path.scale_pi*np.array(path.smooths_y),
                                       color=np.array(ptColors),marker='.',s=200)
                
                # Display the orientation overlays
                for i in range(0,path.numSmoothPoints(),1):
                    xa = path.smooths_x[i]*path.scale_pi
                    ya = path.field_y_pixels-(path.smooths_y[i]*path.scale_pi)
                    oa = np.pi*path.smooths_o[i]/180.0
                    xb = xa + (path.step_size*path.scale_pi)*np.cos(oa)
                    yb = ya - (path.step_size*path.scale_pi)*np.sin(oa)
                    hs_ori.append(plt.plot(np.array([xa,xb]),np.array([ya,yb]),color=np.array(ptColors[i])))
            
            # Reset
            add_state = 0 
            
    if(path.numWayPoints()>1):
        
        # Ask the user if they would like to save the path
        filename = gensup.popupTextEntry('name the path or leave blank to not save',path.loaded_filename)
        
        if(filename!=''):
            
            # Save the .csv file
            nComp = max(0,path.numSmoothPoints()-path.numWayPoints())
            df = pandas.DataFrame(data={"Distance (in)": path.smooths_d,
                                        "Time (s)": path.smooths_t,
                                        "X (in)": path.smooths_x,
                                        "Y (in)": path.smooths_y,
                                        "Velocity (in/s)": path.smooths_v,
                                        "Orientation (deg)": path.smooths_o,
                                        "Way X (in)": path.ways_x + nComp*[''],
                                        "Way Y (in)": path.ways_y + nComp*[''],
                                        "Way Velocity (in/s)": path.ways_v + nComp*[''],
                                        "Way Orientation (deg)": path.ways_o + nComp*['']})
            df.to_csv("../robot paths/%s.csv" %(filename), sep=',',index=False)
            
            # Save an image of the path planning figure
            h_fig.savefig('../robot paths/%s.jpg' %(filename))