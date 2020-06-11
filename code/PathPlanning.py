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
import sys # access to Windows OS
import tkinter as tk # TkInter UI backbone
import xlwt # Excel write functionality

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

def popupPtData(path,x_prior,y_prior):
    """
    ***
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
        [v_way,flags] = gensup.safeTextEntry(flags,textFields[2]['field'],'float',vmin=1.0,vmax=path.v_max/12.0)
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
    buttonSave = tk.Button(popwindow,text='Apply',fg=guiColor_black,bg=guiColor_hotgreen,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionSave)
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

def definePath(path,file_I):
    """
    ***
    """
    
    # Load the field image
    I = cv2.imread(file_I,cv2.IMREAD_UNCHANGED) # load the selected image 
    I = cv2.resize(I,(int(dispRes*path.field_x_real),int(dispRes*path.field_y_real))) # resize the image
    I = gensup.convertColorSpace(I) # fix image coloring
    
    # Calculate the scaling 
    path.fieldScale(I)
    
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
    add_state = 0
    
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
    while(flag_abort==False):
        plt.pause(0.1)
        if(add_state==2):
            
            # Remove the previous plots
            try: h_ways.remove()
            except: pass
            
            # Update the way point plots
            h_ways = ax.scatter(path.scale_pi*np.array(path.ways_x),(path.field_y_pixels)*np.ones((len(path.ways_y)),float) - path.scale_pi*np.array(path.ways_y),
                                facecolors='none',edgecolors='r',marker='o',s=400)
            # Reset
            add_state = 0 
    
    




#-----------------------------------------------------------------------------