# Date: 2021-02-12
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
    
    # Characterize the turns
    ways_clock = [None]
    ways_gamma = [None]
    ways_dbt = [None]
    for i in range(1,path.numWayPoints()-1,1):
        
        # Get the coordinates of the relevant waypoints
        ax = path.ways_x[i-1]
        ay = path.ways_y[i-1]
        bx = path.ways_x[i]
        by = path.ways_y[i]
        cx = path.ways_x[i+1]
        cy = path.ways_y[i+1]
        
        # Calculate the angle between the segments
        dotproduct = (ax-bx)*(cx-bx)+(ay-by)*(cy-by)
        mag_ba = np.sqrt(((ax-bx)**2)+((ay-by)**2))
        mag_bc = np.sqrt(((cx-bx)**2)+((cy-by)**2))
        theta = np.arccos(dotproduct/(mag_ba*mag_bc))
        theta_half = theta/2
        
        # Calculate the angle from the x-axis to the line bisecting the segments
        alpha_bc = np.arctan2((cy-by),(cx-bx))
        crossproduct = (ax-bx)*(cy-by)-(ay-by)*(cx-bx)
        if(crossproduct<0):
            clock = 1
            gamma = alpha_bc+theta_half
        else: 
            clock = -1
            gamma = alpha_bc-theta_half
        ways_gamma.append(gamma)
        ways_clock.append(clock)
        
        # Calculate transition distance along the linear segments
        dbt = path.ways_R[i]/np.tan(theta_half)
        ways_dbt.append(dbt)
        
    # The last waypoint never has any smoothing
    ways_clock.append(None)
    ways_gamma.append(None)
    ways_dbt.append(None)
    
    # Generate the dense position points
    idxs_smooth = [1] # the waypoint the smooth point is trying to reach
    ptxs_smooth = [path.ways_x[0]]
    ptys_smooth = [path.ways_y[0]]
    tchs_smooth = [0]
    section = 'middle'
    rx = None
    ry = None
    kappa = None
    for i in range(0,path.numWayPoints()-1,1):
        
        # Get the coordinates of the relevant waypoints
        ax = path.ways_x[i]
        ay = path.ways_y[i]
        bx = path.ways_x[i+1]
        by = path.ways_y[i+1]
        
        # Get the relevant turn information
        dat_beg = ways_dbt[i]
        clock_beg = ways_clock[i]
        R_beg = path.ways_R[i]
        delta_kappa_beg = path.step_size/R_beg
        dbt_end = ways_dbt[i+1]
        gamma_end = ways_gamma[i+1]
        clock_end = ways_clock[i+1]
        R_end = path.ways_R[i+1]
        delta_kappa_end = path.step_size/R_end
        
        # Reset the next segment flag
        flag_nxt = False
        
        # Check that the waypoints aren't closer than the step size
        dw = np.sqrt(((bx-ax)**2)+((by-ay)**2))
        if(dw>path.step_size):
            timeout = 0
            while(True):
                
                # Check and update the timeout counter
                if(timeout>10000): raise Exception('no solution')
                timeout += 1
                
                if(section=='begarc'):
                    
                    # Find the arc center
                    # has not moved from previous arcend
                
                    # Determine the angle to start at
                    # pick up where you left off from previous arcend
                    
                    # Calculate the transition point
                    alpha_ab = np.arctan2((by-ay),(bx-ax))
                    tx = ax+dat_beg*np.cos(alpha_ab)
                    ty = ay+dat_beg*np.sin(alpha_ab)
                
                    # Calculate the next point on the arc and increment
                    nx = rx+R_beg*np.cos(kappa)
                    ny = ry+R_beg*np.sin(kappa)
                    kappa += clock_beg*delta_kappa_beg
                    
                    # Check for the transition of the arc
                    dotproduct = (nx-rx)*(tx-rx)+(ny-ry)*(ty-ry)
                    mag_rn = np.sqrt(((nx-rx)**2)+((ny-ry)**2))
                    mag_rt = np.sqrt(((tx-rx)**2)+((ty-ry)**2))
                    theta = np.arccos(dotproduct/(mag_rn*mag_rt))
                    if(theta<=delta_kappa_beg): 
                        section = 'middle'
                
                if(section=='middle'):
                
                    # Calculate the next point
                    alpha_ab = np.arctan2((by-ay),(bx-ax))
                    dx = path.step_size*np.cos(alpha_ab)
                    dy = path.step_size*np.sin(alpha_ab)
                    nx = ptxs_smooth[-1]+dx
                    ny = ptys_smooth[-1]+dy
                    
                    # Calculate remaining linear distance to the next way point
                    dab = np.sqrt(((bx-nx)**2)+((by-ny)**2))
                    
                    # Check if we are at the end of this segment
                    if(dab<=path.step_size): flag_nxt = True
                    
                    # Check if we should start arcing
                    if(not flag_nxt):
                        if(dbt_end is not None):
                            if(dab<=dbt_end): 
                                kappa = None
                                section = 'endarc'
                       
                if(section=='endarc'):
                    
                    # Find the arc center
                    dbr = np.sqrt((dbt_end**2)+(R_end**2))
                    rx = bx+dbr*np.cos(gamma_end)
                    ry = by+dbr*np.sin(gamma_end)
                    
                    # Determine the angle to start at
                    if(kappa is None):
                        kappa = np.arctan2((ny-ry),(nx-rx))
                    
                    # Calculate the next point on the arc and increment
                    nx = rx+R_end*np.cos(kappa)
                    ny = ry+R_end*np.sin(kappa)
                    kappa += clock_end*delta_kappa_end
                    
                    # Check for the end of the arc
                    dotproduct = (nx-rx)*(bx-rx)+(ny-ry)*(by-ry)
                    mag_rn = np.sqrt(((nx-rx)**2)+((ny-ry)**2))
                    mag_rb = np.sqrt(((bx-rx)**2)+((by-ry)**2))
                    theta = np.arccos(dotproduct/(mag_rn*mag_rb))
                    if(theta<=delta_kappa_end): 
                        section = 'begarc'
                        flag_nxt = True
                        
                # Insert the next point in the path
                idxs_smooth.append(i+1)
                ptxs_smooth.append(nx)
                ptys_smooth.append(ny)
                if(flag_nxt):
                    if(path.ways_T[i+1]>0.5): tchs_smooth.append(1)
                    else: tchs_smooth.append(0)
                else: tchs_smooth.append(0)
                
                # Check if this segment of the path is complete
                if(flag_nxt): break
                    
        else: section = 'middle'            
        
        # The next dense point is the next waypoint if there was no arc 
        if(section=='middle'):
            idxs_smooth.append(min(i+2,path.numWayPoints()-1))
            ptxs_smooth.append(path.ways_x[i+1])
            ptys_smooth.append(path.ways_y[i+1])
            tchs_smooth.append(0)
            
    # Calculate the distances along the path and for each segment
    ds_seg = 0
    wayNum = 1
    dsts_segs = [0]
    dsts_smooth = [0]
    for i in range(1,len(ptxs_smooth),1):
        
        # Calculate the the change in distance along the path since the last smooth point
        ax = ptxs_smooth[i-1]
        ay = ptys_smooth[i-1]
        bx = ptxs_smooth[i]
        by = ptys_smooth[i]
        ds = np.sqrt(((bx-ax)**2)+((by-ay)**2))
        
        # Total distance along the path
        ds_total = dsts_smooth[-1] + ds
        dsts_smooth.append(ds_total)
        
        # Distance along the path in the current segment
        ds_seg += ds
        
        # Save the distances for each segment
        if(idxs_smooth[i]!=wayNum): 
            dsts_segs.append(ds_seg)
            ds_seg = 0
            wayNum += 1
            
    # Store the distance for the final segment
    dsts_segs.append(ds_seg)
    
    # Calculate the smooth velocities
    wayNum = 0
    v_running = path.ways_v[0] # intialize to start velocity
    vels_smooth = [v_running]
    for i in range(1,len(ptxs_smooth),1):
        
        if(idxs_smooth[i]!=wayNum):
            
            # Determine the requested delta v
            wayNum += 1
            v_goal = path.ways_v[wayNum]
            delta_v_req = v_goal-v_running
            
            # Travel distance available in this segment
            delta_s_seg = dsts_segs[wayNum] 
            
            # Handle acceleration limiting
            if(delta_v_req>0):
                
                # Calculate the maximum delta v permitted
                delta_v_max = -v_running+np.sqrt((v_running**2)+(2*path.a_max*delta_s_seg))
                
                # Accelerate as fast as possible if we missed the previous velocity waypoint
                if(v_running<path.ways_v[wayNum-1]):
                    delta_v_okay = delta_v_max 
                else:
                    delta_v_okay = min(delta_v_req,delta_v_max)
                
            else: 
                
                # No decceleration limit
                delta_v_okay = delta_v_req
                
            # Normalize the delta v per unit distance
            delta_v_per_s = delta_v_okay/delta_s_seg
        
        # Update the robot's current velocity
        delta_s = dsts_smooth[i]-dsts_smooth[i-1]
        v_running += delta_v_per_s*delta_s
        if(delta_v_req>0): v_running = min(v_running,v_goal)
        else: v_running = max(v_running,v_goal)
        vels_smooth.append(v_running)
       
    # Impliment instantaneous acceleration/decceleration below 12.0 in/s
    for i in range(1,len(ptxs_smooth)-1,1):
        if(idxs_smooth[i]==idxs_smooth[i+1]): 
            vels_smooth[i] = max(12.0,vels_smooth[i])
            
    # Calculate the smooth times
    t_total = 0 # total time that the robot has been traveling
    tims_smooth = [t_total]
    for i in range(1,len(ptxs_smooth),1):
        
        # Calculate the time elapsed in the current step
        delta_s = dsts_smooth[i]-dsts_smooth[i-1]
        v_avg = 0.5*(vels_smooth[i]+vels_smooth[i-1])
        delta_t = delta_s/v_avg
        
        # Update the total time
        t_total += delta_t
        tims_smooth.append(t_total)
    
    # Calculate the smooth orientations
    wayNum = 0
    o_running = path.ways_o[0] # intialize to start orientation
    oris_smooth = [o_running]
    for i in range(1,len(ptxs_smooth),1):
        
        if(idxs_smooth[i]!=wayNum):
            
            # Determine the requested delta o
            wayNum += 1
            o_goal = path.ways_o[wayNum]
            if(o_goal>o_running):
                delta_oseg1 = o_goal-o_running
                delta_oseg2 = 360 - delta_oseg1
                if(delta_oseg1<delta_oseg2): delta_o = delta_oseg1
                else: delta_o = -delta_oseg2
            else:
                delta_oseg1 = o_running-o_goal
                delta_oseg2 = 360 - delta_oseg1
                if(delta_oseg1<delta_oseg2): delta_o = -delta_oseg1
                else: delta_o = delta_oseg2
                
            # Travel distance avalaible in this segment
            delta_s_seg = dsts_segs[wayNum] 
                
            # Normalize the delta o per unit distance
            delta_o_per_s = delta_o/delta_s_seg
            
        # Update the robot's current orientation
        delta_s = dsts_smooth[i]-dsts_smooth[i-1]
        o_running += delta_o_per_s*delta_s
        oris_smooth.append(o_running%360)
    
    # Update the path
    path.updateSmoothPath(ptxs_smooth,ptys_smooth,vels_smooth,oris_smooth,dsts_smooth,tims_smooth,tchs_smooth)
    
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
    [x_init,y_init,v_init,o_init,R_init,T_init,way_index] = path.configureWayPoint(x_prior,y_prior)
    if(way_index!=-1): way_order = way_index
    else: way_order = path.numWayPoints()
    
    # Define button callbacks
    def actionClose(*args):
        
        # userEntry = entry0.get()
        popwindow.unbind('<Return>')
        popwindow.quit()
        popwindow.destroy()
        
    def actionSave(*args):
        
        # Check entries for errors
        flags = True
        [O_way,flags] = gensup.safeTextEntry(flags,textFields[0]['field'],'int',vmin=0,vmax=path.numWayPoints())
        [x_way,flags] = gensup.safeTextEntry(flags,textFields[1]['field'],'float',vmin=0.0,vmax=path.field_x_real)
        [y_way,flags] = gensup.safeTextEntry(flags,textFields[2]['field'],'float',vmin=0.0,vmax=path.field_y_real)
        [v_way,flags] = gensup.safeTextEntry(flags,textFields[3]['field'],'float',vmin=path.v_min/12.0,vmax=path.v_max/12.0)
        [o_way,flags] = gensup.safeTextEntry(flags,textFields[4]['field'],'float',vmin=0.0,vmax=360.0)
        [R_way,flags] = gensup.safeTextEntry(flags,textFields[5]['field'],'float',vmin=3*path.step_size)
        [T_way,flags] = gensup.safeTextEntry(flags,textFields[6]['field'],'bool')
        if(T_way): T_way = 1
        else: T_way = 0
    
        # Save the error-free entries in the correct units
        if(flags):
             
            # Add way point
            path.addWayPoint(x_way,y_way,v_way*12,o_way,R_way,T_way,way_index,O_way)
            
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
    windH = 700 # window height 
    popwindow.geometry(str(windW)+'x'+str(windH))
    popwindow.configure(background=guiColor_offwhite)
    popwindow.resizable(width=False, height=False)
    
    # Set the initial window location
    popwindow.geometry("+{}+{}".format(int(0.5*(popwindow.winfo_screenwidth()-windW)),int(0.5*(popwindow.winfo_screenheight()-windH))))
    
    # Configure to handle use of the Windows close button
    popwindow.protocol('WM_DELETE_WINDOW',actionClose)
    
    # Define the fields
    fieldNames = ['Order',
                  'X Position (in)',
                  'Y Position (in)',
                  'Velocity (ft/s)',
                  'Orientation (deg) [0-360]',
                  'Turn Radius (in)',
                  'Touch this Point']
    defaults = [way_order,
                x_init,
                y_init,
                v_init,
                o_init,
                R_init,
                T_init>0.5]
    
    # Set up the elements
    textFields = []
    for i in range(0,len(fieldNames),1):
        [title,field] = gensup.easyTextField(popwindow,windW,fieldNames[i],str(defaults[i]))
        textFields.append({'title': title, 'field': field})
    if(way_index==-1): buttonName = 'Create'
    else: buttonName = 'Edit'
    if(way_index==-1): textFields[0]['field'].configure(state=tk.DISABLED)
    buttonSave = tk.Button(popwindow,text=buttonName,fg=guiColor_black,bg=guiColor_hotgreen,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionSave)
    popwindow.bind('<Return>',actionSave)
    buttonDelete = tk.Button(popwindow,text='Delete',fg=guiColor_black,bg=guiColor_hotyellow,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionDelete)
    
    # Place all elements
    for i in range(0,len(textFields),1):
        textFields[i]['title'].pack(fill='both')
        textFields[i]['field'].pack()
    buttonSave.pack(pady=20)
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
                    try:
                        popupPtData(path,x_prior,y_prior)
                        add_state = 2
                    except: add_state=0 # ignore
                
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
                try: 
                    path = generatePath(path)
                    flag_solution = True
                except: 
                    tk.messagebox.showerror('4265 Path Planner','No pathing solution found, please modify the waypoints')
                    flag_solution = False
                
                if(flag_solution):
                
                    # Display the path metrics
                    details_start = 'start at (%0.2f in, %0.2f in, %0.0f°)' %(path.smooths_x[0],path.smooths_y[0],path.smooths_o[0])
                    details_end = 'end at (%0.2f in, %0.2f in, %0.0f°) predicted travel time: %0.2f s' %(path.smooths_x[-1],path.smooths_y[-1],path.smooths_o[-1],path.total_t)
                    plt.xlabel('X: Down Field\n%s\n%s' %(details_start,details_end))
                    
                    # Display the smooth path
                    ptColors = []
                    for i in range(0,path.numSmoothPoints(),1):
                        if(path.smooths_T[i]>0.5):
                            ptColor = [1,0,0] # color "must touch points red"
                        else: ptColor = plt.cm.plasma(path.smooths_v[i]/path.v_max)
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
                                        "Touch": path.smooths_T,
                                        "Way X (in)": path.ways_x + nComp*[''],
                                        "Way Y (in)": path.ways_y + nComp*[''],
                                        "Way Velocity (in/s)": path.ways_v + nComp*[''],
                                        "Way Orientation (deg)": path.ways_o + nComp*[''],
                                        "Way Turn Radius (in)": path.ways_R + nComp*[''],
                                        "Touch this Point": path.ways_T + nComp*['']})
            df.to_csv("../robot paths/%s.csv" %(filename), sep=',',index=False)
            
            # Save an image of the path planning figure
            h_fig.savefig('../robot paths/%s.jpg' %(filename))