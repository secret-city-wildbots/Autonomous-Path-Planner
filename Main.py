# Date: 2020-01-12
# Author: Secret City Wildbots
# Description: allows the user to load an image of the field and generate a 
#              planned path for the robot to follow
#------------------------------------------------------------------------------

# Load external libraries
import cv2 # OpenCV
import datetime
import matplotlib.pyplot as plt # plotting functionality
import numpy as np # math operations
from pylab import ginput # get user mouse inputs
import tkinter as tk # file UI
from tkinter import messagebox
root = tk.Tk() # suppress the default window
root.withdraw() # suppress the default window
from tkinter import filedialog # file UI
import xlrd # Excel read functionality
import xlwt # Excel write functionality

# Change Environment settings (some are only applicable if running in an iPython console)
try:
    from IPython import get_ipython # needed to run magic commands
    ipython = get_ipython() # needed to run magic commands
    ipython.magic('matplotlib qt') # display figures in a separate window
except: pass
plt.rcParams.update({'font.size': 24}) # change the default font size for plots

#------------------------------------------------------------------------------

# Parameters
fieldx = 52 # [feet] field length
fieldy = 27 # [feet] field width
res = 100 # controls the resolution of the displayed image
thresh_samePt = 5 # [inches] if the selected point is closer than this, you will edit a previous point
maxVel_init = 15 # [ft/s] maximum robot velocity
delta_s_init = 1 # [inches] step length along the path
r_min_init = 12 # [inches] turn radius at v = 0
r_max_init = 100 # [inches] turn radius at v = maxVel

#------------------------------------------------------------------------------

def textEntryWidget(title,x_init,y_init,**kwargs):
    """ Creates a blocking text entry widget
    Args:
        title: name of the entry box
    Returns:
        user_input: the user input as a string
    Saves:
        None
    """
    
    # Parse the kwargs
    try: vel_init = kwargs['vel_init']
    except: vel_init = ''
    try: ori_init = kwargs['ori_init']
    except: ori_init = ''
    
    global ptx, pty, vel, ori
    def widgetClose(*args):
        global ptx, pty, vel, ori
        ptx = entryFieldPosX.get()
        pty = entryFieldPosY.get()
        vel = entryFieldVelocity.get()
        ori = entryFieldOrientation.get()
        textwindow.quit()
        textwindow.destroy()
    
    # Open the text entry widget
    textwindow = tk.Tk()
    textwindow.title(title)
    textwindow.geometry('400x300')
    textwindow.configure(background='#%02x%02x%02x' % ((245,245,245)))
    
    # Configure to handle use of the Windows close button
    textwindow.protocol('WM_DELETE_WINDOW',widgetClose)
    
    # Create, pack, and display the widget
    entryFieldPosX = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    entryFieldPosY = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    entryFieldVelocity = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    entryFieldOrientation = tk.Entry(textwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
    labelFieldPosX = tk.Label(textwindow,text='\nX Position (in)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=2,width=30,anchor='w')
    labelFieldPosY = tk.Label(textwindow,text='Y Position (in)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=1,width=30,anchor='w')  
    labelFieldVelocity = tk.Label(textwindow,text='Velocity (ft/s)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=1,width=30,anchor='w')
    labelFieldOrientation = tk.Label(textwindow,text='Orientation (deg)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=1,width=30,anchor='w')         
    labelFieldPosX.pack()                              
    entryFieldPosX.pack()
    labelFieldPosY.pack()
    entryFieldPosY.pack()
    labelFieldVelocity.pack()
    entryFieldVelocity.pack()
    labelFieldOrientation.pack()
    entryFieldOrientation.pack()
    entryFieldPosX.focus_set()
    entryFieldPosX.insert(0,'%0.2f' %(x_init))
    entryFieldPosY.insert(0,'%0.2f' %(y_init))
    entryFieldVelocity.insert(0,'%s' %(str(vel_init)))
    entryFieldOrientation.insert(0,'%s' %(str(ori_init)))
    textwindow.mainloop()
    
    return ptx, pty, vel, ori

#------------------------------------------------------------------------------

def openWorkbook(sheetNames,sheetHeaders):
    """ Opens a workbook on the hardisk 
    Args:
        sheetNames: 1D list of names for each sheet in the workbook
        sheetHeaders: 2D list of horizontal headers in each sheet
    Returns:
        book: workbook object
        sheets: list of sheet objects
    Saves:
        None
    """
    
    # Initialize the workbook
    numSheets = len(sheetNames)
    book = xlwt.Workbook()
    sheets = []
    for i in range(0,numSheets,1):
        sheets.append(book.add_sheet(sheetNames[i]))
    
    # Place the headers
    for i in range(0,numSheets,1):
        for j in range(0,len(sheetHeaders[i]),1):
            sheets[i].write(0,j,sheetHeaders[i][j].replace('_',' ').replace('@',''))
    
    return book, sheets

#------------------------------------------------------------------------------

def saveToWorkbook(book,sheets,data):
    """ Saves data into an already-opened workbook
    Args:
        book: workbook object
        sheets: list of sheet objects
        data: list of numpy data arrays [sheet][column][rows (vector of data)]
    Returns:
        flag_writeok: is returned as True if the data write was successfull
    Saves:
        None
    """
    
    try: 
        for i in range(0,len(sheets),1):
            # Each sheet
            for j in range(0,len(data[i]),1):
                # Each column
                for k in range(0,len(data[i][j]),1):
                    # Each row (writable datum)
                    try: d = data[i][j][k].astype(float)
                    except: d = data[i][j][k]
                    sheets[i].write(k+1,j,d)
        flag_writeok = True
    except:
        flag_writeok = False

    return flag_writeok

#------------------------------------------------------------------------------
    
def saveWorkbook(book,filepath):
    """ Saves a workbook to the hardisk
    Args:
        book: workbook object
        filepath: full file path (including file name) to save the workbook
    Returns:
        flag_saveok: is returned as True if the save operation was successfull
    Saves:
        None
    """
    
    try:
        book.save(filepath+'.xls')
        flag_saveok = True
    except:
        flag_saveok = False
        
    return flag_saveok

#------------------------------------------------------------------------------
    
def readWorkbookColumn(book,sheet,headerHeight,column,fmt):
    """ Reads data from a specified column in a workbook 
    Args:
        book: workbook object
        sheet: workbook sheet number
        headerHeight: number of rows that should be ignored as part of the header
        column: target column index as a string, e.g. 'AM'
        fmt: expected data format, options are 'date', 'int', 'float', and 'string'
    Returns:
        values: a list containing all of the formatted values from the targeted column
    Saves:
        None
    """

    # Initialize
    table = book.sheet_by_index(sheet)
    nRows = table.nrows # not counting the header row
    values = []
    
    # Parse column information
    alphabet = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
    numDigits = len(column)
    digits = []
    for i in range(0,numDigits,1):
        for j in range(0,len(alphabet),1):
            if(column[i]==alphabet[j]): break
        digits.append(j)
    column_idx = 0
    for i in range(0,numDigits,1):
        column_idx += (digits[i]+1)*(len(alphabet)**(numDigits-i-1))
    column_idx -= 1

    for i in range(headerHeight,nRows,1):
        
        # Pull the value from the current cell
        value = table.cell_value(rowx=i,colx=column_idx)
               
        # Format the value
        if(fmt=='date'): value = datetime.datetime(*xlrd.xldate_as_tuple(value,book.datemode))
        elif(fmt=='int'): 
            try: value = int(value)
            except: value = 0
        elif(fmt=='float'): 
            try: value = float(value)
            except: value = 0.0
        elif(fmt=='string'):  
            pass
        
        # Add the value to the list of values
        values.append(value)
        
    return values

    
#------------------------------------------------------------------------------
    
def generatePath(ptxs_way,ptys_way,vels_way,oris_way):
    
    # Pull the current parameters
    try: maxVel = float(entryMaxVel.get())
    except: maxVel = maxVel_init
    try: delta_s = float(entryStepSize.get())
    except: delta_s = delta_s_init
    try: r_min = float(entryMinTurnRadius.get())
    except: r_min = r_min_init
    try: r_max = float(entryMaxTurnRadius.get())
    except: r_max = r_max_init
    
    # Insert dense points along a piece-wise linear interpolation
    vels_segs = [] # [in/s] the dense v points listed per segment
    oris_segs = [] # [degrees] the dense orientation points lister per segment
    ptxs_segs = [] # [inches] the dense x points listed per segment
    ptys_segs = [] # [inches] the dense y points listed per segment
    for i in range(0,len(ptxs_way)-1,1):
        
        # Initialize
        ptxs_seg = []
        ptys_seg = []
        vels_seg = []
        oris_seg = []
        
        # Calculate the length of the path segment
        max_s = np.sqrt(((ptys_way[i+1]-ptys_way[i])**2)+((ptxs_way[i+1]-ptxs_way[i])**2)) 
        
        # Calculate the steps for the current path segment 
        gamma = np.arctan2((ptys_way[i+1]-ptys_way[i]),(ptxs_way[i+1]-ptxs_way[i]))
        delta_x = delta_s*np.cos(gamma)
        delta_y = delta_s*np.sin(gamma)
        delta_v = (vels_way[i+1]-vels_way[i])/max_s
        if((oris_way[i]!='ahead')&(oris_way[i+1]!='ahead')):
            delta_o = (oris_way[i+1]-oris_way[i])/max_s
        
        # Include the current waypoint
        ptxs_seg.append(ptxs_way[i]) 
        ptys_seg.append(ptys_way[i]) 
        vels_seg.append(vels_way[i])
        oris_seg.append(oris_way[i])
        
        # Step the inserted points along the path
        total_s = 0 # tracks total travel along the current path segment
        
        while((total_s+delta_s)<max_s):
            
            ptxs_seg.append(ptxs_seg[-1]+delta_x)
            ptys_seg.append(ptys_seg[-1]+delta_y)
            vels_seg.append(vels_seg[-1]+delta_v)
            if((oris_way[i]!='ahead')&(oris_way[i+1]!='ahead')):
                oris_seg.append(oris_seg[-1]+delta_o)
            else: 
                oris_seg.append('ahead')
            total_s += delta_s 
            
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
        numer = (ptys_way[segNum+1]-ptys_way[segNum])
        denom = (ptxs_way[segNum+1]-ptxs_way[segNum])
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
    ptxs_smooth.append(ptxs_way[-1])
    ptys_smooth.append(ptys_way[-1])
    vels_smooth.append(vels_way[-1])
    oris_smooth.append(oris_way[-1])
    
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
        
        # Calculate the along-path orientations [180,-180]
        theta = 180.0*np.arctan2((yb-ya),(xb-xa))/np.pi
        
        # Fill in all unspecified orientations with the natural angle
        if(oris_smooth[i]=='ahead'):
            oris_smooth[i] = theta
            
    # Fix the final orientation if it wasn't specified
    if(oris_smooth[-1]=='ahead'): oris_smooth[-1] = oris_smooth[-2]
        
    # Report the total path travel time
    print('\nTime to travel the path: %0.2fs' %(tims_smooth[-1]))

    return ptxs_smooth,ptys_smooth,vels_smooth,oris_smooth,dsts_smooth,tims_smooth

#------------------------------------------------------------------------------

def actionPlan():
    
    # Load the field image
    path_I = filedialog.askopenfilename(initialdir='',title="Select a Cropped Field Diagram",filetypes=[('Images','*.jpg *.jpeg *.png *.tif *.tiff')]);
    I = cv2.imread(path_I,cv2.IMREAD_GRAYSCALE) # load the selected image 
    I = cv2.resize(I,(int(res*fieldx),int(res*fieldy))) # resize the image
    
    # Ask the user to pre-load any way points
    userInput = messagebox.askyesno("Path Planned", "Load Saved Waypoints?")
    if(userInput):
        
        # Load the selected waypoints
        path_way = filedialog.askopenfilename(initialdir='',title="Load Saved Waypoints",filetypes=[('Paths','*xls')]);
        book = xlrd.open_workbook(path_way)
        ptxs_way_load = readWorkbookColumn(book,0,1,'G','float')
        ptys_way_load = readWorkbookColumn(book,0,1,'H','float')
        vels_way_load = readWorkbookColumn(book,0,1,'I','float')
        oris_way_load = readWorkbookColumn(book,0,1,'J','string')
        
        # Remove extra points (assumes you don't ever go to y = 0.0)
        for i in range(0,len(ptys_way_load),1):
            if(ptys_way_load[i]==0): break;
        ptxs_way_load = ptxs_way_load[0:i]
        ptys_way_load = ptys_way_load[0:i]
        vels_way_load = vels_way_load[0:i]
        oris_way_load = oris_way_load[0:i]
        
    else:
            
        ptxs_way_load = [] # initialize
    
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
    plt.imshow(I_color,extent=[0,I.shape[1],I.shape[0],0])
    plt.title('Right-Click to Select Waypoints',fontsize=24)
    plt.draw()
    
    # Watch for window close event
    global flag_abort
    flag_abort = False
    def actionWindowClose(evt):
        global flag_abort
        flag_abort = True
    h_fig.canvas.mpl_connect('close_event', actionWindowClose) 
    
    # Let the user select the requested number of points
    ptxs_way = [] # initialize
    ptys_way = [] # initialize
    vels_way = [] # initialize
    oris_way = [] # initialize
    hs_way = [] # initialize
    hs_smooth = [] # initialize
    hs_ori = [] # initialize
    numLoaded = len(ptxs_way_load)
    loadedNum = 0
    while(flag_abort==False):
        
        # Pull the current parameters
        try: maxVel = float(entryMaxVel.get())
        except: maxVel = maxVel_init
        try: delta_s = float(entryStepSize.get())
        except: delta_s = delta_s_init
        
        # Check for click
        if(loadedNum<numLoaded): pt = [0]
        else: pt = ginput(1,show_clicks=True,timeout=0.25,mouse_add=3,mouse_pop=None,mouse_stop=1) 
            
        
        if(len(pt)!=0): 
            
            # Convert the point into real units
            if(loadedNum<numLoaded):
                x_new = ptxs_way_load[loadedNum]
                y_new = ptys_way_load[loadedNum]
            else:
                x_new = pt[0][0]/scale_pi # [in]
                y_new = (fieldy_pixels-pt[0][1])/scale_pi # [in]
            
            # Check to see if this is a new or a pre-exisiting point
            flag_newPt = True
            i = -1 # needed for first call
            for i in range(0,len(ptxs_way),1):
                d = np.sqrt(((ptxs_way[i]-x_new)**2)+((ptys_way[i]-y_new)**2))
                if(d<thresh_samePt):
                    flag_newPt = False
                    break
            
            # Edit or add a new point
            if(flag_newPt):
                # Add a new point
                if(loadedNum<numLoaded):
                    ptx = ptxs_way_load[loadedNum]
                    pty = ptys_way_load[loadedNum]
                    vel = vels_way_load[loadedNum]/12
                    ori = oris_way_load[loadedNum]
                else:
                    [ptx,pty,vel,ori] = textEntryWidget('New Way Point #%i' %(i+1),x_new,y_new)
            else:
                # Edit an existing point
                print('editing')
                [ptx,pty,vel,ori] = textEntryWidget('Edit Way Point #%i' %(i),ptxs_way[i],ptys_way[i],vel_init=vels_way[i]/12,ori_init=oris_way[i])
    
            # Check for point deletion
            if((ptx=='')|(pty=='')):
                if(flag_newPt==False):
                    print(ptxs_way)
                    print('deleting %i' %i)
                    del ptxs_way[i]
                    del ptys_way[i]
                    del vels_way[i]
                    del oris_way[i]
                    print(ptxs_way)
            else:
            
                # Coerce to valid inputs
                try: ptx = np.float(ptx)
                except: ptx = x_new
                try: pty = np.float(pty)
                except: pty = y_new
                try: vel = 12*np.float(vel)
                except: vel = 0.0
                try: ori = np.float(ori)
                except: ori = 'ahead'
                
                # Append or insert the user inputs
                if(flag_newPt):
                    ptxs_way.append(ptx)
                    ptys_way.append(pty)
                    vels_way.append(vel)
                    oris_way.append(ori)
                else:
                    ptxs_way[i] = ptx
                    ptys_way[i] = pty
                    vels_way[i] = vel
                    oris_way[i] = ori
                    
            # Generate the path
            if(len(ptxs_way)>=2):
                [ptxs_smooth,ptys_smooth,vels_smooth,oris_smooth,dsts_smooth,tims_smooth] = generatePath(ptxs_way,ptys_way,vels_way,oris_way)
            
            # Display the path
            for j in range(len(hs_way)): hs_way[j].remove()
            hs_way = []
            for j in range(0,len(ptxs_way),1):
                h = ax.scatter(ptxs_way[j]*scale_pi,fieldy_pixels-(ptys_way[j]*scale_pi),facecolors='none', edgecolors='r',marker='o',s=400)
                hs_way.append(h)
            for j in range(len(hs_smooth)): hs_smooth[j].remove()
            hs_smooth = []
            if(len(ptxs_way)>=2):
                for j in range(0,len(ptxs_smooth),1):
                    ptColor = plt.cm.plasma(vels_smooth[j]/(12*maxVel))
                    ptColor = np.array([ptColor[0],ptColor[1],ptColor[2]])
                    h = ax.scatter(ptxs_smooth[j]*scale_pi,fieldy_pixels-(ptys_smooth[j]*scale_pi),color=ptColor,marker='.',s=200)
                    hs_smooth.append(h)
            for j in range(len(hs_ori)): hs_ori[j][0].remove()
            hs_ori = []
            if(len(ptxs_way)>=2):
                for i in range(0,len(ptxs_smooth),1):
                    xa = ptxs_smooth[i]*scale_pi
                    ya = fieldy_pixels-(ptys_smooth[i]*scale_pi)
                    oa = np.pi*oris_smooth[i]/180.0
                    xb = xa + (delta_s*scale_pi)*np.cos(oa)
                    yb = ya - (delta_s*scale_pi)*np.sin(oa)
                    h = plt.plot(np.array([xa,xb]),np.array([ya,yb]),color='k')
                    hs_ori.append(h)
                    
            # Increment through the loaded path
            if(loadedNum<numLoaded): loadedNum += 1
                    
    # Save the generated path to a spreadsheet
    plt.pause(1.0)
    path_save = filedialog.asksaveasfilename(title='Save Path File',filetypes=[('Spreadsheet','*.xls')])
    if(len(path_save)!=0):
        [book,sheets] = openWorkbook(['Smooth Path and Waypoints'],[['Distance (in)','Time (s)','X (in)','Y (in)','Velocity (in/s)','Orientation (deg)','Way X (in)','Way Y (in)','Way Velocity (in/s)','Way Orientation (deg)']])
        data = [[np.array(dsts_smooth),np.array(tims_smooth),np.array(ptxs_smooth),np.array(ptys_smooth),np.array(vels_smooth),np.array(oris_smooth),np.array(ptxs_way),np.array(ptys_way),np.array(vels_way),np.array(oris_way)]]
        saveToWorkbook(book,sheets,data)
        saveWorkbook(book,path_save)

#------------------------------------------------------------------------------
    
def guiClose(*args):
    guiwindow.quit()
    guiwindow.destroy()

# Open the GUI
guiwindow = tk.Tk()
guiwindow.title('Secret City Wildbots')
guiwindow.geometry('400x300')
guiwindow.configure(background='#%02x%02x%02x' % ((245,245,245)))

# Configure to handle use of the Windows close button
guiwindow.protocol('WM_DELETE_WINDOW',guiClose)

# Create, pack, and display the widget
menubar = tk.Menu(guiwindow)
menuFile = tk.Menu(menubar, tearoff=0)
menuFile.add_command(label='Open Field Map',command=actionPlan)
menubar.add_cascade(label='File',menu=menuFile)
guiwindow.config(menu=menubar)

# Set up entry fields and pack
entryMaxVel = tk.Entry(guiwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
entryStepSize = tk.Entry(guiwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
entryMinTurnRadius = tk.Entry(guiwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
entryMaxTurnRadius = tk.Entry(guiwindow,fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((255,255,255)),font=('Arial',16),state=tk.NORMAL,width=30)
labelMaxVel = tk.Label(guiwindow,text='\nMaximum Velocity (ft/s)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=2,width=30,anchor='w')
labelStepSize = tk.Label(guiwindow,text='Step Size (in)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=1,width=30,anchor='w')  
labelMinTurnRadius = tk.Label(guiwindow,text='Minimum Turn Radius (in)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=1,width=30,anchor='w')
labelMaxTurnRadius = tk.Label(guiwindow,text='Maximum Turn Radius (in)',fg='#%02x%02x%02x' % ((0,0,0)),bg='#%02x%02x%02x' % ((245,245,245)),font=('Arial',16),height=1,width=30,anchor='w')         
labelMaxVel.pack()                              
entryMaxVel.pack()
labelStepSize.pack()
entryStepSize.pack()
labelMinTurnRadius.pack()
entryMinTurnRadius.pack()
labelMaxTurnRadius.pack()
entryMaxTurnRadius.pack()
entryMaxVel.focus_set()
entryMaxVel.insert(0,'%0.2f' %(maxVel_init))
entryStepSize.insert(0,'%0.1f' %(delta_s_init))
entryMinTurnRadius.insert(0,'%0.0f' %(r_min_init))
entryMaxTurnRadius.insert(0,'%0.0f' %(r_max_init))
guiwindow.mainloop()   

#------------------------------------------------------------------------------