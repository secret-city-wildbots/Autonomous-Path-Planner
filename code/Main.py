# Date: 2021-02-05
# Description: a path planner for FRC 2020
#-----------------------------------------------------------------------------

# Versioning information
versionNumber = '2.0.2' # breaking.major-feature-add.minor-feature-or-bug-fix
versionType = 'release' # options are "beta" or "release"
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
import sys # system
flag_upgraded = False
if(versionType=='release'):
    
    # Automatic installation
    version.install()

    # Update the readme file
    flag_upgraded = version.upgrade(versionNumber)
    
    # Exit promptly
    if(flag_upgraded): sys.exit()

#-----------------------------------------------------------------------------

# Load remaining external modules
import cv2 # OpenCV
import math # additional math functionality
import matplotlib # Matplotlib module
import matplotlib.pyplot as plt # Matplotlib plotting functionality
import pandas # data handling toolbox
import time # time related functions
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
guiColor_hotgreen = h_npz_settings['guiColor_hotgreen']
guiFontSize_large = h_npz_settings['guiFontSize_large']
guiFontSize_small = h_npz_settings['guiFontSize_small']
guiFontType_normal = h_npz_settings['guiFontType_normal']
guiFontType_uniform = h_npz_settings['guiFontType_uniform']

# Load defaults
try: h_npz_defaults = np.load(dirPvars+'defaults.npz',allow_pickle=True)
except: pass
try: dpiScaling = h_npz_defaults['dpiScaling']
except: dpiScaling = 100.0

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
scrnW *= float(dpiScaling/100.0)
scrnH *= float(dpiScaling/100.0)
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
        
        # Load and set the defaults
        try:
            h_npz_defaults = np.load(dirPvars+'defaults.npz',allow_pickle=True)
            field_x_real = h_npz_defaults['field_x_real']
            field_y_real = h_npz_defaults['field_y_real']
            v_min = 0.0
            v_max = h_npz_defaults['v_max']
            a_max = h_npz_defaults['a_max']
            step_size = h_npz_defaults['step_size']
            try: dpiScaling = h_npz_defaults['dpiScaling']
            except: dpiScaling = 100.0 # backwards compatibility
        except:
            field_x_real = 52.4375
            field_y_real = 26.9375 
            v_min = 1.0
            v_max = 15.0
            a_max = 3.0
            step_size = 1.0
            dpiScaling = 100.0
        self.field_x_real = 12*field_x_real # (in) length of the field
        self.field_y_real = 12*field_y_real # (in) width of the field
        self.v_min = 12*v_min # (in/s) minimum robot velocity
        self.v_max = 12*v_max # (in/s) maximum robot velocity
        self.a_max = 12*a_max # (in/s^2) maximum robot acceleration
        self.step_size = step_size # (in) path step size
        self.dpiScaling = dpiScaling # Windows DPI scaling setting
        
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
        self.ways_R = [] # (in) list of way point turn radii
        self.ways_T = [] # (in) list of way point touch priority
        
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
            R_init = 3*self.step_size
            T_init = False
            
        else:
            
            # Index for an existing point
            way_index = i
            
            # Default values for a new point
            x_init = np.round(self.ways_x[i],2) # (in)
            y_init = np.round(self.ways_y[i],2) # (in)
            v_init = (1/12)*self.ways_v[i]
            o_init = self.ways_o[i]
            R_init = self.ways_R[i]
            T_init = self.ways_T[i]
        
        return x_init, y_init, v_init, o_init, R_init, T_init, way_index
    
    def addWayPoint(self,x,y,v,o,R,T,way_index,way_order):
        
        if(way_index==-1):
            
            # Add a new point to the list of way points
            (self.ways_x).append(x)
            (self.ways_y).append(y)
            (self.ways_v).append(v)
            (self.ways_o).append(o)
            (self.ways_R).append(R)
            (self.ways_T).append(T)
            
        else:
            
            # Edit an existing point
            self.ways_x[way_index] = x
            self.ways_y[way_index] = y
            self.ways_v[way_index] = v
            self.ways_o[way_index] = o
            self.ways_R[way_index] = R
            self.ways_T[way_index] = T
            
            # Move a way point
            if(way_index!=way_order):
                ways_x = []
                ways_y = []
                ways_v = []
                ways_o = []
                ways_R = []
                ways_T = []
                for i in range(0,self.numWayPoints(),1):
                    if(i==way_index): pass
                    elif(i==way_order):
                        ways_x.append(self.ways_x[way_index])
                        ways_y.append(self.ways_y[way_index])
                        ways_v.append(self.ways_v[way_index])
                        ways_o.append(self.ways_o[way_index])
                        ways_R.append(self.ways_R[way_index])
                        ways_T.append(self.ways_T[way_index])
                        ways_x.append(self.ways_x[i])
                        ways_y.append(self.ways_y[i])
                        ways_v.append(self.ways_v[i])
                        ways_o.append(self.ways_o[i])
                        ways_R.append(self.ways_R[i])
                        ways_T.append(self.ways_T[i])
                    else:
                        ways_x.append(self.ways_x[i])
                        ways_y.append(self.ways_y[i])
                        ways_v.append(self.ways_v[i])
                        ways_o.append(self.ways_o[i])
                        ways_R.append(self.ways_R[i])
                        ways_T.append(self.ways_T[i])
                self.ways_x = ways_x
                self.ways_y = ways_y
                self.ways_v = ways_v
                self.ways_o = ways_o
                self.ways_R = ways_R
                self.ways_T = ways_T
        
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
            (self.ways_R).pop(way_index)
            (self.ways_T).pop(way_index)
            
    def loadWayPoints(self,file_csv):
        
        try: 
            
            # Load the .csv file
            df = pandas.read_csv(file_csv)
            
            # Parse the way point information
            self.ways_x = list(df['Way X (in)'].values)
            self.ways_y = list(df['Way Y (in)'].values)
            self.ways_v = list(df['Way Velocity (in/s)'].values)
            self.ways_o = list(df['Way Orientation (deg)'].values)
            self.ways_R = list(df['Way Turn Radius (in)'].values)
            self.ways_T = list(df['Touch this Point'].values)
            
            # Remove dummy values saved in the .csv file
            while(True):
                if(not math.isnan(self.ways_x[-1])): break
                else:
                    self.ways_x.pop(-1)
                    self.ways_y.pop(-1)
                    self.ways_v.pop(-1)
                    self.ways_o.pop(-1)
                    self.ways_R.pop(-1)
                    self.ways_T.pop(-1)
                    
            # Record the filename
            filename = file_csv.split('/')[-1]
            filename = filename.split('.')[0]
            self.loaded_filename = filename
            
        except: pass
            
    def numWayPoints(self):
        
        # Calculate the current number of way points
        return len(self.ways_x)
            
    def updateSmoothPath(self,ptxs_smooth,ptys_smooth,vels_smooth,oris_smooth,dsts_smooth,tims_smooth,tchs_smooth):

        # Update the smooth path
        self.smooths_x = ptxs_smooth
        self.smooths_y = ptys_smooth
        self.smooths_v = vels_smooth
        self.smooths_o = oris_smooth
        self.smooths_d = dsts_smooth
        self.total_d = dsts_smooth[-1]
        self.smooths_t = tims_smooth
        self.total_t = tims_smooth[-1]
        self.smooths_T = tchs_smooth
        
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
    [a_max,flags] = gensup.safeTextEntry(flags,textFields[3]['field'],'float',vmin=0.5,vmax=100.0)
    [step_size,flags] = gensup.safeTextEntry(flags,textFields[4]['field'],'float',vmin=1.0,vmax=100.0)
    [dpiScaling,flags] = gensup.safeTextEntry(flags,textFields[5]['field'],'float',vmin=100.0)
    
    # Save the error-free entries in the correct units
    if(flags):
        
        # Save to the disk
        np.savez(dirPvars+'defaults',
                 field_x_real=field_x_real,
                 field_y_real=field_y_real,
                 v_max=v_max,
                 a_max=a_max,
                 step_size=step_size,
                 dpiScaling=dpiScaling)
        
        # Save in memory
        path.field_x_real = 12.0*field_x_real
        path.field_y_real = 12.0*field_y_real
        path.v_max = 12.0*v_max
        path.a_max = 12.0*a_max
        path.step_size =step_size
        path.dpiScaling = dpiScaling
        
#-----------------------------------------------------------------------------

def actionLoadField(*args):
    """
    Loads the field map and allows the user to start planning a path
    """
    
    # Lock the GUI
    buttonPlan.configure(bg=guiColor_offwhite,state=tk.DISABLED)
    
    # Reinitialize the path
    path.reset()
    
    # Ask the user to load a field map
    file_I = filedialog.askopenfilename(initialdir='../field drawings/',title = 'Select a Field Drawing',filetypes=recognizedImageExtensions)
    
    if(file_I!=''):
    
        # Ask the user to load a robot model
        file_robot = filedialog.askopenfilename(initialdir='../robot models/',title = 'Select a Robot Model',filetypes=recognizedImageExtensions)
        
        # Ask the user to load a previous path
        file_csv = filedialog.askopenfilename(initialdir='../robot paths/',title = 'Select a Robot Path',filetypes=[('CSV','*.csv ')] )
        path.loadWayPoints(file_csv)
        
        # Start the path planner
        plan.definePath(path,file_I,file_robot)
    
    # Reset the GUI
    buttonPlan.configure(bg=guiColor_hotgreen,state=tk.NORMAL)
    
#-----------------------------------------------------------------------------

# Open the GUI window
guiwindow = tk.Tk()
guiwindow.title(softwareName)
windW = int(0.28*min(1080,minScrnDim)) # window width
windH = int(0.65*min(1080,minScrnDim)) # window height 
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
              'Maximum Robot Acceleration (ft/s²)',
              'Step Size (in)',
              'Windows DPI Scaling (%)']
defaults = [path.field_x_real/12.0,
            path.field_y_real/12.0,
            path.v_max/12.0,
            path.a_max/12.0,
            path.step_size,
            path.dpiScaling]

# Set up the settings elements
textFields = []
for i in range(0,len(fieldNames),1):
    [title,field] = gensup.easyTextField(guiwindow,windW,fieldNames[i],str(defaults[i]))
    textFields.append({'title': title, 'field': field})
buttonApply = tk.Button(guiwindow,text='Apply',fg=guiColor_black,bg=guiColor_hotpink,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.02*windW),command=actionApplySettings)
buttonPlan = tk.Button(guiwindow,text='Plan',fg=guiColor_black,bg=guiColor_hotgreen,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.02*windW),command=actionLoadField)

# Place all elements
for i in range(0,len(textFields),1):
    textFields[i]['title'].pack(fill='both')
    textFields[i]['field'].pack()
buttonApply.pack(pady=10)
buttonPlan.pack(pady=10)
logo.pack(pady=10)

# Run the GUI window
if((__name__ == '__main__') and not flag_upgraded):
    plt.ion()
    gensup.flushMatplotlib()
    guiwindow.mainloop()
    sys.exit()
else: time.sleep(10.0) # allow the users to read any error messages