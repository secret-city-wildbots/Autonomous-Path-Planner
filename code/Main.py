# Date: 2020-10-01
# Description: a path planner for FRC 2020
#-----------------------------------------------------------------------------

# Versioning information
versionNumber = '1.2.0' # breaking.major-feature-add.minor-feature-or-bug-fix
versionType = 'beta' # options are "beta" or "release"
print('Loading v%s...' %(versionNumber))

# Ignore future and depreciation warnings when not in development
import warnings 
if(versionType=='release'):
    warnings.simplefilter(action='ignore', category=FutureWarning)
    warnings.simplefilter(action='ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=UserWarning)
    
# Hardcoded directory paths
dirPvars = '../vars/' # persistent variables directory
    
# Check the specified operating system
import numpy as np # Numpy toolbox
try: ostype = str(np.load(dirPvars+'ostype.npy'))
except: ostype = None
    
# Get system configuration  
import os # operating system interactions
cpus = os.cpu_count() # number of logical cores
import psutil # needed to query system memory
ram = int(psutil.virtual_memory().free//1e9)
from win32api import GetSystemMetrics # system configuration and details
scrnW = GetSystemMetrics(0) # [pixels] width of the computer monitor
scrnH = GetSystemMetrics(1) # [pixels] height of the computer monitor

# Report system configuration
print('\nOperating System: %s' %(ostype))
print('Monitor resolution: %ip x %ip' %(scrnW,scrnH))
print('CPUs detected: %i' %(cpus))
print('RAM available: %i GB \n' %(ram))

#-----------------------------------------------------------------------------

# Perform initial installation if necessary
import VersioningControl as version # functions to handle software updates
if(versionType=='release'):
    
    # Automatic installation
    version.install()

    # Update the readme file
    flag_upgraded = version.upgrade(versionNumber)
    
    # Restart the software if the software has been upgraded
    if(flag_upgraded): raise Exception('restart required')
    
else: flag_upgraded = False

#-----------------------------------------------------------------------------

# Load remaining external modules
import cv2 # OpenCV
import math # additional math functionality
import matplotlib # Matplotlib module
import matplotlib.pyplot as plt # Matplotlib plotting functionality
import pandas # data handling toolbox
import tkinter as tk # TkInter UI backbone
from tkinter import filedialog # TkInter file browsers
from tkinter import messagebox # TkInter popup windows

# Load remaining user modules
import GeneralSupportFunctions as gensup # general support functions
import PathPlanning as plan # pathe planner

# Load persistent variables
h_npz_settings = np.load(dirPvars+'settings.npz',allow_pickle=True)
softwareName = str(h_npz_settings['softwareName'])
recognizedImageExtensions = np.ndarray.tolist(h_npz_settings['recognizedImageExtensions'])
guiColor_black = h_npz_settings['guiColor_black']
guiColor_white = h_npz_settings['guiColor_white']
guiColor_offwhite = h_npz_settings['guiColor_offwhite']
guiColor_hotpink = h_npz_settings['guiColor_hotpink']
guiFontSize_large = h_npz_settings['guiFontSize_large']
guiFontSize_small = h_npz_settings['guiFontSize_small']
guiFontType_normal = h_npz_settings['guiFontType_normal']
guiFontType_uniform = h_npz_settings['guiFontType_uniform']

#-----------------------------------------------------------------------------

# Close previous windows
cv2.destroyAllWindows()
plt.close('all')

# Change Environment settings (some are only applicable if running in an iPython console)
try:
    from IPython import get_ipython # needed to run magic commands
    ipython = get_ipython() # needed to run magic commands
    ipython.magic('matplotlib qt') # display figures in a separate window
except: pass
plt.rcParams.update({'font.size': 24}) # change the default font size for plots
plt.rcParams.update({'figure.max_open_warning': False}) # disable warning about too opening too many figures - don't need that kind of negativity 
plt.rcParams['keymap.quit'] = '' # disable matplotlib hotkeys
plt.rcParams['keymap.save'] = '' # disable matplotlib hotkeys
if(ostype=='Linux'): matplotlib.use('TkAgg') # call the correct backend when installed on Linux

# Adjust dpi and font settings based on screen resolution
minScrnDim = min(scrnW,scrnH) # smallest screen dimension
guiScaling = min(1.0,minScrnDim/1080) # scaling factor for the user interface
if(scrnW*scrnH<(1.1*1920*1080)): plt.rcParams.update({'figure.dpi': 50})
else: plt.rcParams.update({'figure.dpi': 100})
guiFontSize_large = int(np.ceil(guiFontSize_large*(guiScaling**1)))
guiFontSize_small = int(np.ceil(guiFontSize_small*(guiScaling**1)))

#-----------------------------------------------------------------------------

class Path(): 
    """
    Data object defining a robot path
    """
    
    def __init__(self):
        
        # Explicit settings
        self.field_x_real = 12*52.4375 # (in) length of the field
        self.field_y_real = 12*26.9375 # (in) width of the field
        self.v_max = 12*15.0 # (in/s) maximum robot velocity
        self.a_max = 12*3.0 # (in/s^2) maximum robot acceleration
        self.step_size = 1.0 # (in) path step size
        self.radius_min = 12.0 # (in) minimum robot turn radius
        self.radius_max = 100.0 # (in) maximum robot turn radius
        
        # Reset the path
        self.reset()
        
    def reset(self):
        
        # Implicit settings
        self.field_x_pixels = 1.0 # (pix) length of the field
        self.field_y_pixels = 1.0 # (pix) width of the field
        self.scale_pi = 1.0 # (pix/in)
        self.loaded_filename = '' # the name of the loaded path
        
        # Way points
        self.ways_x = [] # (in) list of way point x positions
        self.ways_y = [] # (in) list of way point y positions
        self.ways_v = [] # (in/s) list of way point velocities
        self.ways_o = [] # (deg) list of way point orientations
        
        # Smooth path
        self.smooths_x = [] # (in) list of smooth x positions
        self.smooths_y = [] # (in) list of smooth y positions
        self.smooths_v = [] # (in/s) list of sooth velocities
        self.smooths_o = [] # (deg) list of smooth orientations
        self.smooths_d = [] # (in) list of smooth cumulative distance
        self.total_d = 0.0 # (in) total distance along the path
        self.smooths_t = [] # (in) lust of smooth cumulative time
        self.total_t = 0.0 # (s) total path travel time
        
    def fieldScale(self,I):
        
        # Calculate the scaling conversion factors
        self.field_x_pixels = I.shape[1]
        self.field_y_pixels = I.shape[0]
        self.scale_pi = self.field_x_pixels/self.field_x_real
        
    def configureWayPoint(self,x_prior,y_prior):
        
        # Convert the candidate points into inches for search
        x_prior = x_prior/self.scale_pi # (in)
        y_prior = (self.field_y_pixels-y_prior)/self.scale_pi # (in)
        
        # Check to see if this is a new or a pre-exisiting point
        thresh_samePt = 5 # [in] if the selected point is closer than this, you will edit a previous point
        flag_newPt = True
        i = -1 # needed for first call
        for i in range(0,len(self.ways_x),1):
            d = np.sqrt(((self.ways_x[i]-x_prior)**2)+((self.ways_y[i]-y_prior)**2))
            if(d<thresh_samePt):
                flag_newPt = False
                break
        
        if(flag_newPt):
            
            # Index for a new point
            way_index = -1
            
            # Default values for a new point
            x_init = np.round(x_prior,2) # (ft)
            y_init = np.round(y_prior,2) # (ft)
            v_init = 1.0
            o_init = 0.0
            
        else:
            
            # Index for an existing point
            way_index = i
            
            # Default values for a new point
            x_init = np.round(self.ways_x[i],2) # (in)
            y_init = np.round(self.ways_y[i],2) # (in)
            v_init = (1/12)*self.ways_v[i]
            o_init = self.ways_o[i]
        
        return x_init, y_init, v_init, o_init, way_index
    
    def addWayPoint(self,x,y,v,o,way_index):
        
        if(way_index==-1):
            
            # Add a new point to the list of way points
            (self.ways_x).append(x)
            (self.ways_y).append(y)
            (self.ways_v).append(v)
            (self.ways_o).append(o)
            
        else:
            
            # Edit an existing point
            self.ways_x[way_index] = x
            self.ways_y[way_index] = y
            self.ways_v[way_index] = v
            self.ways_o[way_index] = o
        
    def removeWayPoint(self,way_index):
        
        if(way_index==-1): 
            
            # Do not add a new point
            pass
        
        else:
            
            # Remove an existing point
            (self.ways_x).pop(way_index)
            (self.ways_y).pop(way_index)
            (self.ways_v).pop(way_index)
            (self.ways_o).pop(way_index)
            
    def loadWayPoints(self,file_csv):
        
        try: 
            
            # Load the .csv file
            df = pandas.read_csv(file_csv)
            
            # Parse the way point information
            self.ways_x = list(df['Way X (in)'].values)
            self.ways_y = list(df['Way Y (in)'].values)
            self.ways_v = list(df['Way Velocity (in/s)'].values)
            self.ways_o = list(df['Way Orientation (deg)'].values)
            
            # Remove dummy values saved in the .csv file
            while(True):
                if(not math.isnan(self.ways_x[-1])): break
                else:
                    self.ways_x.pop(-1)
                    self.ways_y.pop(-1)
                    self.ways_v.pop(-1)
                    self.ways_o.pop(-1)
                    
            # Record the filename
            filename = file_csv.split('/')[-1]
            filename = filename.split('.')[0]
            self.loaded_filename = filename
            
        except: pass
            
    def numWayPoints(self):
        
        # Calculate the current number of way points
        return len(self.ways_x)
            

    def updateSmoothPath(self,ptxs_smooth,ptys_smooth,vels_smooth,oris_smooth,dsts_smooth,tims_smooth):

        # Update the smooth path
        self.smooths_x = ptxs_smooth
        self.smooths_y = ptys_smooth
        self.smooths_v = vels_smooth
        self.smooths_o = oris_smooth
        self.smooths_d = dsts_smooth
        self.total_d = dsts_smooth[-1]
        self.smooths_t = tims_smooth
        self.total_t = tims_smooth[-1]
        
    def numSmoothPoints(self):
        
        # Calculate the current number of smooth points
        return len(self.smooths_x)
        
# Instantiate the robot path
path = Path()
    
#-----------------------------------------------------------------------------

# Initialize global variables

#-----------------------------------------------------------------------------

def easyPlace(h,x,y):
    """
    Simplifies placing GUI elements in the main window
    """
    
    if((x>=0)&(y>=0)): h.place(x=abs(x)*windW,y=abs(y)*windH,anchor=tk.NW)
    elif((x<0)&(y>=0)): h.place(x=windW-abs(x)*windW,y=abs(y)*windH,anchor=tk.NE)
    elif((x>=0)&(y<0)): h.place(x=abs(x)*windW,y=windH-abs(y)*windH,anchor=tk.SW)
    elif((x<0)&(y<0)): h.place(x=windW-abs(x)*windW,y=windH-abs(y)*windH,anchor=tk.SE)
    
def lockMenus(topMenus,flag_lock): 
    """
    Simplifies the locking and un-locking process for the top menus
    """           
            
    # Set flags for locking an unlocking the menus
    if(flag_lock==True): lock = tk.DISABLED
    else: lock = tk.NORMAL 
    
    # Lock or unlock specific sets of menus
    for menu in topMenus:
        if(menu=='File'):
            menuFile.entryconfig('Load Field Map',state=lock) 

#-----------------------------------------------------------------------------

def actionNope(*args):
    """
    Default error message for un-implimented operations
    """
    
    # Report that the current operation is unavailable
    messagebox.showinfo(softwareName,'Error: This operation is\nnot yet available.')
    
#-----------------------------------------------------------------------------

def actionQuit(*args):
    """
    Closes the main GUI and exits the code
    """
    
    # Close the main GUI window 
    try:
        guiwindow.quit()
        guiwindow.destroy()
    except: pass

#-----------------------------------------------------------------------------

def actionApplySettings(*args):
    """
    Check the settings for errors and save them
    """
    
    # Check entries for errors
    flags = True
    [field_x_real,flags] = gensup.safeTextEntry(flags,textFields[0]['field'],'float',vmin=0.0,vmax=100.0)
    [field_y_real,flags] = gensup.safeTextEntry(flags,textFields[1]['field'],'float',vmin=0.0,vmax=100.0)
    [v_max,flags] = gensup.safeTextEntry(flags,textFields[2]['field'],'float',vmin=1.0,vmax=30.0)
    [step_size,flags] = gensup.safeTextEntry(flags,textFields[3]['field'],'float',vmin=1.0,vmax=100.0)
    [radius_min,flags] = gensup.safeTextEntry(flags,textFields[4]['field'],'float',vmin=1.0,vmax=1000.0)
    [radius_max,flags] = gensup.safeTextEntry(flags,textFields[5]['field'],'float',vmin=1.1*radius_min,vmax=1000.0)
    [a_max,flags] = gensup.safeTextEntry(flags,textFields[6]['field'],'float',vmin=0.5,vmax=100.0)
    
    # Save the error-free entries in the correct units
    if(flags):
        path.field_x_real = 12.0*field_x_real
        path.field_y_real = 12.0*field_y_real
        path.v_max = 12.0*v_max
        path.step_size =step_size
        path.radius_min = radius_min
        path.radius_max = radius_max
        path.a_max = 12.0*a_max
        
#-----------------------------------------------------------------------------

def actionLoadField(*args):
    """
    Loads the field map and allows the user to start planning a path
    """
    
    # Reinitialize the path
    path.reset()
    
    # Ask the user to load a field map
    file_I = filedialog.askopenfilename(initialdir='../field drawings/',title = 'Select a Field Drawing',filetypes=recognizedImageExtensions)
    
    # Ask the user to load a robot model
    file_robot = filedialog.askopenfilename(initialdir='../robot models/',title = 'Select a Robot Model',filetypes=recognizedImageExtensions)
    
    # Ask the user to load a previous path
    file_csv = filedialog.askopenfilename(initialdir='../robot paths/',title = 'Select a Robot Path',filetypes=[('CSV','*.csv ')] )
    path.loadWayPoints(file_csv)
    
    # Start the path planner
    lockMenus(['File'],True)
    plan.definePath(path,file_I,file_robot)
    lockMenus(['File'],False)
    
#-----------------------------------------------------------------------------

# Open the GUI window
guiwindow = tk.Tk()
guiwindow.title(softwareName)
windW = int(0.30*min(1080,minScrnDim)) # window width
windH = int(0.75*min(1080,minScrnDim)) # window height 
guiwindow.geometry(str(windW)+'x'+str(windH))
guiwindow.configure(background=guiColor_offwhite)
guiwindow.resizable(width=False, height=False)

# Set the initial window location
guiwindow.geometry("+{}+{}".format(int(0.5*(guiwindow.winfo_screenwidth()-windW)),int(0.5*(guiwindow.winfo_screenheight()-windH))))

# Configure to handle use of the Windows close button
guiwindow.protocol('WM_DELETE_WINDOW',actionQuit)

# Set up the logos
logo = tk.Canvas(guiwindow,width=int(564*guiScaling),height=int(280*guiScaling),highlightthickness=0,background=guiColor_offwhite)  
I_logo = gensup.convertColorSpace(cv2.imread(dirPvars+'graphic_4265.png')) # load the default image
gensup.easyTkImageDisplay(guiwindow,logo,I_logo,forceSize=[int(564*guiScaling),int(280*guiScaling)])

# Define the settings fields
fieldNames = ['Field Length (ft)',
              'Field Width (ft)',
              'Maximum Robot Velocity (ft/s)',
              'Step Size (in)',
              'Minimum Turn Radius (in)',
              'Maximum Turn Radius (in)',
              'Maximum Robot Acceleration (ft/s²)']
defaults = [path.field_x_real/12.0,
            path.field_y_real/12.0,
            path.v_max/12.0,
            path.step_size,
            path.radius_min,
            path.radius_max,
            path.a_max/12.0]

# Set up the settings elements
textFields = []
for i in range(0,len(fieldNames),1):
    [title,field] = gensup.easyTextField(guiwindow,windW,fieldNames[i],str(defaults[i]))
    textFields.append({'title': title, 'field': field})
buttonApply = tk.Button(guiwindow,text='Apply',fg=guiColor_black,bg=guiColor_hotpink,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.02*windW),command=actionApplySettings)

# Set up the menu bar
menubar = tk.Menu(guiwindow)
menuFile = tk.Menu(menubar, tearoff=0)
menuFile.add_command(label='Load Field Map',command=actionLoadField)
menuFile.add_command(label='Quit',command=actionQuit)
menubar.add_cascade(label='File',menu=menuFile)
guiwindow.config(menu=menubar)

# Place all elements
for i in range(0,len(textFields),1):
    textFields[i]['title'].pack(fill='both')
    textFields[i]['field'].pack()
buttonApply.pack(pady=20)
logo.pack(pady=20)

# Final GUI initializations

# Run the GUI window
if((__name__ == '__main__') and not flag_upgraded):
    plt.ion()
    gensup.flushMatplotlib()
    guiwindow.mainloop()
else: plt.pause(10.0) # allow the users to read any error messages