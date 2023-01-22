# Date: 2023-01-22
# Description: a path planner for the FIRST Robotics Competition
#-----------------------------------------------------------------------------

# Versioning information
versionNumber = '2.3.1' # breaking.major-feature-add.minor-feature-or-bug-fix
versionType = 'dev' # options are "dev" or "stable"
print('Loading v%s...' %(versionNumber))

# Ignore future and depreciation warnings when not in development
import warnings 
if(versionType=='stable'):
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
if(versionType=='stable'):
    
    # Automatic installation
    version.install(sys.argv)

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
from Constants import(softwareName,
                      recognizedImageExtensions,
                      guiColor_black,
                      guiColor_offwhite,
                      guiColor_hotpink,
                      guiColor_hotgreen,
                      guiColor_hotyellow,
                      guiFontSize_large,
                      guiFontSize_small,
                      guiFontType_normal)

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
plt.rcParams.update({'font.size': 18}) # change the default font size for plots
plt.rcParams.update({'figure.max_open_warning': False}) # disable warning about too opening too many figures - don't need that kind of negativity 
plt.rcParams['keymap.quit'] = '' # disable matplotlib hotkeys
plt.rcParams['keymap.save'] = '' # disable matplotlib hotkeys
matplotlib.use('QtAgg') # set the rendering backend
matplotlib.rcParams['toolbar'] = 'toolmanager' # set the default window toolbar

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
            try: omega_fraction = h_npz_defaults['omega_fraction']
            except: omega_fraction = 0.3 # backwards compatibility
            try: ref_x = h_npz_defaults['ref_x']
            except: ref_x = 0.0 # backwards compatibility
            try: ref_y = h_npz_defaults['ref_y']
            except: ref_y = 0.0 # backwards compatibility
        except:
            field_x_real = 54.3 # (ft)
            field_y_real = 26.3 # (ft)
            v_min = 0.0 # (ft/s)
            v_max = 15.0 # (ft/s)
            a_max = 8.0 # (ft/s2)
            step_size = 1.0 # (in)
            dpiScaling = 100.0
            omega_fraction = 0.3
            ref_x = 0.0 # (in)
            ref_y = 0.0 # (in)
        self.field_x_real = 12*field_x_real # (in) length of the field
        self.field_y_real = 12*field_y_real # (in) width of the field
        self.v_min = 12*v_min # (in/s) minimum robot velocity
        self.v_max = 12*v_max # (in/s) maximum robot velocity
        self.a_max = 12*a_max # (in/s^2) maximum robot acceleration
        self.step_size = step_size # (in) path step size
        self.dpiScaling = dpiScaling # Windows DPI scaling setting
        self.folder_save = '../robot paths/'
        self.omega_fraction = omega_fraction # fraction of the time of a segment to rotate at cruise velocity
        self.ref_x = ref_x # reference x coordinate
        self.ref_y = ref_y # reference y coordinate
        
        # Reset the path
        self.reset()
        
    def reset(self):
        
        # Implicit settings
        self.field_x_pixels = 1.0 # (pix) length of the field
        self.field_y_pixels = 1.0 # (pix) width of the field
        self.scale_pi = 1.0 # (pix/in)
        self.loaded_filename = 'unamed 1' # the name of the loaded path
        
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
        
    def configureWayPoint(self,x_prior,y_prior,flag_newPt):
        
        # Convert the candidate points into inches for search
        x_prior = x_prior/self.scale_pi # (in)
        y_prior = (self.field_y_pixels-y_prior)/self.scale_pi # (in)
        
        if(flag_newPt):
            
            # Index for a new point
            way_index = -1
            
            # Default values for a new point
            x_init = np.round(x_prior,2) # (ft)
            y_init = np.round(y_prior,2) # (ft)
            v_init = 0.0
            o_init = 0.0
            R_init = 3*self.step_size
            T_init = False
            
        else:
            
            # Find the closest point to edit
            way_index = -1
            d_edit = None
            for i in range(0,len(self.ways_x),1):
                d = np.sqrt(((self.ways_x[i]-x_prior)**2)+((self.ways_y[i]-y_prior)**2))
                if(d_edit is not None):
                    if(d<d_edit): 
                        d_edit = d
                        way_index = i
                else:
                    d_edit = d
                    way_index = i
            
            # Default values for an existing point
            x_init = np.round(self.ways_x[way_index],2) # (in)
            y_init = np.round(self.ways_y[way_index],2) # (in)
            v_init = (1/12)*self.ways_v[way_index]
            o_init = self.ways_o[way_index]
            R_init = self.ways_R[way_index]
            T_init = self.ways_T[way_index]
        
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
            self.folder_save = file_csv[:file_csv.rfind('/')+1]
        
        except: pass
            
    def numWayPoints(self):
        
        # Calculate the current number of way points
        return len(self.ways_x)
            
    def updateSmoothPath(self,ptxs_smooth,ptys_smooth,vels_smooth,oris_smooth,dsts_smooth,tims_smooth,tchs_smooth,omgs_smooth):

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
        self.smooths_w = omgs_smooth
        
    def numSmoothPoints(self):
        
        # Calculate the current number of smooth points
        return len(self.smooths_x)
    
    def probe(self,x_prior,y_prior):
        
        # Convert the click point into inches for search
        x_prior = x_prior/self.scale_pi # (in)
        y_prior = (self.field_y_pixels-y_prior)/self.scale_pi # (in)
        
        # Find the closest point in the path
        smooth_index = -1
        d_closest = None
        for i in range(0,len(self.smooths_x),1):
            d = np.sqrt(((self.smooths_x[i]-x_prior)**2)+((self.smooths_y[i]-y_prior)**2))
            if(d_closest is not None):
                if(d<d_closest): 
                    d_closest = d
                    smooth_index = i
            else:
                d_closest = d
                smooth_index = i
                
        # Extract the relevant info and format a text string
        if(d_closest is not None):
            fmt_str = ''
            fmt_str += 'Smooth point at (%0.2f in, %0.2f in) \n' %(self.smooths_x[smooth_index],self.smooths_y[smooth_index])
            fmt_str += 'Velocity: %0.1f ft/s \n' %(self.smooths_v[smooth_index]/12)
            fmt_str += 'Orientation: %0.1f° \n' %(self.smooths_o[smooth_index])
            fmt_str += 'Distance from Start: %0.1f in \n' %(self.smooths_d[smooth_index])
            fmt_str += 'Time from Start: %0.2f s' %(self.smooths_t[smooth_index])
        else: fmt_str = 'The path could not be probed.'
        
        return fmt_str
    
    def move(self,x_prior,y_prior,x_new,y_new):
        
        # Convert the candidate points into inches for search
        x_prior = x_prior/self.scale_pi # (in)
        y_prior = (self.field_y_pixels-y_prior)/self.scale_pi # (in)
        
        # Find the closest point to edit
        way_index = -1
        d_edit = None
        for i in range(0,len(self.ways_x),1):
            d = np.sqrt(((self.ways_x[i]-x_prior)**2)+((self.ways_y[i]-y_prior)**2))
            if(d_edit is not None):
                if(d<d_edit): 
                    d_edit = d
                    way_index = i
            else:
                d_edit = d
                way_index = i      
           
        # Update the waypoint position
        self.ways_x[way_index] = x_new/self.scale_pi # (in)
        self.ways_y[way_index] = (self.field_y_pixels-y_new)/self.scale_pi # (in)
    
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
    [omega_fraction,flags] = gensup.safeTextEntry(flags,textFields[4]['field'],'float',vmin=0.1,vmax=0.9)
    [step_size,flags] = gensup.safeTextEntry(flags,textFields[5]['field'],'float',vmin=1.0,vmax=100.0)
    [ref_x,flags] = gensup.safeTextEntry(flags,textFields[6]['field'],'float',vmin=0.0,vmax=12.0*field_x_real)
    [ref_y,flags] = gensup.safeTextEntry(flags,textFields[7]['field'],'float',vmin=0.0,vmax=12.0*field_y_real)
    [dpiScaling,flags] = gensup.safeTextEntry(flags,textFields[8]['field'],'float',vmin=50.0)
    
    # Save the error-free entries in the correct units
    if(flags):
        
        # Save to the disk
        np.savez(dirPvars+'defaults',
                  field_x_real=field_x_real,
                  field_y_real=field_y_real,
                  v_max=v_max,
                  a_max=a_max,
                  omega_fraction=omega_fraction,
                  step_size=step_size,
                  ref_x=ref_x,
                  ref_y=ref_y,
                  dpiScaling=dpiScaling)
        
        # Save in memory
        path.field_x_real = 12.0*field_x_real
        path.field_y_real = 12.0*field_y_real
        path.v_max = 12.0*v_max
        path.a_max = 12.0*a_max
        path.step_size =step_size
        path.dpiScaling = dpiScaling
        path.omega_fraction = omega_fraction
        path.ref_x = ref_x
        path.ref_y = ref_y
        
#-----------------------------------------------------------------------------

def actionResetFiles(*args):
    """
    Reset previous file selections
    """
    
    # Confirm intention to reet the previous file selections
    userChoice = tk.messagebox.askyesno(softwareName,'Are you sure you want to reset all previous file selections (e.g., field maps, robot models, and field calibrations)?',icon='warning')
    
    # Delete file selections
    if(userChoice): 
        try: os.remove(dirPvars+'rememberedField.npy')
        except: pass
        try: os.remove(dirPvars+'rememberedRobot.npy')
        except: pass
        try: os.remove(dirPvars+'rememberedBluePoints.npy')
        except: pass
        try: os.remove(dirPvars+'rememberedRedPoints.npy')
        except: pass
        
#-----------------------------------------------------------------------------

def actionLoadField(*args):
    """
    Loads the field map and allows the user to start planning a path
    """
    
    # Lock the GUI
    buttonPlan.configure(bg=guiColor_offwhite,state=tk.DISABLED)
    
    # Reinitialize the path
    path.reset()
    
    # Ask the user which alliance they are planning for
    if(tk.messagebox.askyesno(softwareName,'Are you planning a path for the RED alliance?')): alliance = 'red'
    else: alliance = 'blue'
    
    # Ask the user to load a field map
    try: file_I = str(np.load(dirPvars+'rememberedField.npy'))
    except: file_I = filedialog.askopenfilename(initialdir='../field drawings/',title = 'Select a Field Drawing',filetypes=recognizedImageExtensions)
    np.save(dirPvars+'rememberedField.npy',file_I)
    
    if(file_I!=''):
    
        # Ask the user to load a robot model
        try: file_robot = str(np.load(dirPvars+'rememberedRobot.npy'))
        except: file_robot = filedialog.askopenfilename(initialdir='../robot models/',title = 'Select a Robot Model',filetypes=recognizedImageExtensions)
        if(file_robot!=''): np.save(dirPvars+'rememberedRobot.npy',file_robot)
        
        # Ask the user to load calibration points for the red side of the field
        if(alliance=='red'):
            try: file_red = str(np.load(dirPvars+'rememberedRedPoints.npy'))
            except: file_red = filedialog.askopenfilename(title = 'Select Field Calibration Points for the Red Side',filetypes=[('CSV','*.csv ')])
            if(file_red!=''): np.save(dirPvars+'rememberedRedPoints.npy',file_red)
        else: file_red = ''
        
        # Ask the user to load calibration points for the blue side of the field
        if(alliance=='blue'):
            try: file_blue = str(np.load(dirPvars+'rememberedBluePoints.npy'))
            except: file_blue = filedialog.askopenfilename(title = 'Select Field Calibration Points for the Blue Side',filetypes=[('CSV','*.csv ')])
            if(file_blue!=''): np.save(dirPvars+'rememberedBluePoints.npy',file_blue)
        else: file_blue = ''
        
        # Ask the user to load a previous path
        file_csv = filedialog.askopenfilename(initialdir='',title = 'Select a Robot Path',filetypes=[('CSV','*.csv ')])
        path.loadWayPoints(file_csv)
        
        # Start the path planner
        try: plan.definePath(path,alliance,file_I,file_robot,buttonPlan,file_red,file_blue)
        except: buttonPlan.configure(bg=guiColor_hotgreen,state=tk.NORMAL)
        
    else: buttonPlan.configure(bg=guiColor_hotgreen,state=tk.NORMAL)
    
#-----------------------------------------------------------------------------

# Open the GUI window
guiwindow = tk.Tk()
guiwindow.title(softwareName)
windW = int(300) # window width
windH = int(850) # window height 
guiwindow.geometry(str(windW)+'x'+str(windH))
guiwindow.configure(background=guiColor_offwhite)
guiwindow.resizable(width=False,height=True)

# Set the initial window location
guiwindow.geometry("+{}+{}".format(int(0.5*(guiwindow.winfo_screenwidth()-windW)),int(0.5*(guiwindow.winfo_screenheight()-windH))))

# Configure to handle use of the Windows close button
guiwindow.protocol('WM_DELETE_WINDOW',actionQuit)

# Set up the logos
logo = tk.Canvas(guiwindow,width=int(564*guiScaling),height=int(280*guiScaling),highlightthickness=0,background=guiColor_offwhite)  
I_logo = gensup.convertColorSpace(cv2.imread(dirPvars+'graphic_4265.png')) # load the default image
gensup.easyTkImageDisplay(guiwindow,logo,I_logo,forceSize=[int(564*guiScaling),windW])

# Define the settings fields
fieldNames = ['Field Length (ft)',
              'Field Width (ft)',
              'Maximum Robot Velocity (ft/s)',
              'Maximum Robot Acceleration (ft/s²)',
              'Rotational Cruise Velocity Fraction',
              'Smooth Point Step Size (in)',
              'X Reference Point (in)',
              'Y Reference Point (in)',
              'Windows DPI Scaling (%)']
defaults = [np.round(path.field_x_real/12.0,2),
            np.round(path.field_y_real/12.0,2),
            np.round(path.v_max/12.0,2),
            np.round(path.a_max/12.0,2),
            path.omega_fraction,
            path.step_size,
            path.ref_x,
            path.ref_y,
            path.dpiScaling]

# Set up the settings elements
textFields = []
for i in range(0,len(fieldNames),1):
    [title,field] = gensup.easyTextField(guiwindow,windW,fieldNames[i],str(defaults[i]))
    spacer = tk.Label(guiwindow,text='',bg=guiColor_offwhite,font=(guiFontType_normal,2),anchor='w')
    textFields.append({'title': title, 'field': field, 'spacer': spacer})
buttonApply = tk.Button(guiwindow,text='Save Settings',fg=guiColor_black,bg=guiColor_hotpink,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionApplySettings)
buttonReset = tk.Button(guiwindow,text='Reset File Selections',fg=guiColor_black,bg=guiColor_hotyellow,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionResetFiles)
buttonPlan = tk.Button(guiwindow,text='Plan New Path',fg=guiColor_black,bg=guiColor_hotgreen,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.04*windW),command=actionLoadField)

# Place all elements
for i in range(0,len(textFields),1):
    textFields[i]['title'].pack(fill='x')
    textFields[i]['field'].pack(fill='x',pady=1)
    textFields[i]['spacer'].pack(fill='x')
buttonApply.pack(pady=3,fill='x')
buttonReset.pack(pady=3,fill='x')
buttonPlan.pack(pady=3,fill='x')
logo.pack(pady=10,expand=True)

# Run the GUI window
if((__name__ == '__main__') and not flag_upgraded):
    plt.ion()
    gensup.flushMatplotlib()
    guiwindow.mainloop()
    sys.exit()
else: time.sleep(10.0) # allow the users to read any error messages