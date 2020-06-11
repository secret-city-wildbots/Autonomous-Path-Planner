# Date: 2020-06-11
# Description: general support functions
#-----------------------------------------------------------------------------

# Load external modules
import cv2 # OpenCV
import matplotlib.pyplot as plt # Matplotlib plotting functionality
from matplotlib.patches import Patch # graphics object
import numpy as np # Numpy toolbox
import os # access to Windows OS
from PIL import ImageTk,Image # TkInter-integrated image display functionality 
import tkinter as tk # TkInter UI backbone
import xlwt # Excel write functionality

# Hardcoded directory paths
dirPvars = '../vars/' # persistent variables directory

# Load persistent variables
try:
    h_npz_settings = np.load(dirPvars+'settings.npz',allow_pickle=True)
    softwareName = str(h_npz_settings['softwareName'])
    figSize = list(h_npz_settings['figSize'])
    guiColor_black = h_npz_settings['guiColor_black']
    guiColor_white = h_npz_settings['guiColor_white']
    guiColor_offwhite = h_npz_settings['guiColor_offwhite']
    guiColor_darkgreen = h_npz_settings['guiColor_darkgreen']
    guiColor_cherryred = h_npz_settings['guiColor_cherryred']
    guiColor_hotpink = h_npz_settings['guiColor_hotpink']
    guiFontSize_large = h_npz_settings['guiFontSize_large']
    guiFontSize_small = h_npz_settings['guiFontSize_small']
    guiFontType_normal = h_npz_settings['guiFontType_normal']
    guiFontType_uniform = h_npz_settings['guiFontType_uniform']
except: pass # safe for initial installation loading

# Check the specified operating system
try: ostype = str(np.load(dirPvars+'ostype.npy'))
except: ostype = None

#-----------------------------------------------------------------------------

def clip(x,floor,ceil):
    """
    Custom clipping (coercion) function 
    Args:
        x: input value to be clipped
        floor: floor for the clipping operation
        ceil: ceiling for the clipping operation
    Returns:
        clipped: the clipped (coerced) version of the input function
    Saves:
        None
    """
    
    if(x<floor):
        clipped = floor
    elif(x>ceil):
        clipped = ceil
    else:
        clipped = x
        
    return clipped

#-----------------------------------------------------------------------------

def copySelectedFile(path,moveto):
    """ 
    Copies a file from one location (and name) to another
    Args:
        path: source file path
        moveto: path to the target directory
    Returns:
        None
    Saves:
        None
    """
    
    # Construct the required Windows command string 
    os.system('copy "'+path.replace('\\','\\').replace('/','\\')+'" "'+moveto.replace('\\','\\').replace('/','\\')+'"')
    
    # Report the copied file
    print(path)
    
#-----------------------------------------------------------------------------

def ospath(path,**kwargs):
    """
    Replaces characters in file paths and folder names based on the operating 
    system
    Args:
        path: the input path
        **mode: ['folder','path'] operating mode
    Returns:
        path: the edited path
    Saves:
        None
    """
    
    # Parse the kwargs
    try: mode = kwargs['mode']
    except: mode = 'folder'
    
    # Check Operating System
    if(ostype is None): ostype_confirmed = str(np.load(dirPvars+'ostype.npy'))
    else: ostype_confirmed = np.copy(ostype)
    
    if(mode=='folder'):
        
        # Modify a folder name as needed
        if(ostype_confirmed=='Windows-Mac'): path = path
        elif(ostype_confirmed=='Linux'): path = path.replace(' ','_')
        
    elif(mode=='path'):
        
        # Modify a path to be OS agnositic
        path = path.replace('\\','/')
    
    return path

#-----------------------------------------------------------------------------

def flushMatplotlib():
    """
    Flushes the Matplotlib figure buffer
    Args:
        None
    Returns:
        None
    Saves:
        None
    """

    # Flush the matplotlib buffer
    h_reset = plt.figure('matplotlib reset')
    plt.show()
    plt.close(h_reset)

#-----------------------------------------------------------------------------

def printmb(messageBoard,text,**kwargs):
    """
    Prints a message on the messageboard
    Args:
        messageBoard: handle for the message board
        text: text to display on the message board
        **guiwindow: handle for the entire guiwindow, only needed to force a refresh
    Returns:
        None
    Saves:
        None
    """
    
    # Update the messageboard
    messageBoard.configure(state=tk.NORMAL)
    messageBoard.insert(tk.END,text)
    messageBoard.yview(tk.END)
    messageBoard.focus_set()
    messageBoard.configure(state=tk.DISABLED)
    
    # Force refresh of the GUI
    try:
        guiwindow = kwargs['guiwindow']
        guiwindow.update_idletasks()
    except: pass

#-----------------------------------------------------------------------------

def ljustifyOffset(itemList):
    """
    Calculates the character offset needed to left-justify a table of values
    Args:
        itemList: string list of table items
    Returns:
        ljoff: character offset
    Saves:
        None
    """

    maxcharlength = 0 # initialize
    for i in range(0,len(itemList),1): 
        charlength = len(itemList[i])
        if(charlength>maxcharlength):
            maxcharlength = charlength
    ljoff = maxcharlength + 1
    
    return ljoff

#-----------------------------------------------------------------------------

def progressGUI(messageBoard,guiwindow,printtext,first,current,last):
    """ 
    Renders in-GUI progress updates 
    Args:
        messageBoard: handle for the GUI message board
        guiwindow: handle for the GUI window
        printtext: prefix text for the progress update, e.g. "Progress: "42%
        first: starting value of the incrementer
        current: current value of the incrementer
        last: final value of the incrementer
    Returns:
        None
    Saves:
        None
    """
    
    # Determine the progress percentage
    try:
        perc_old = int(round(100.0*float(current-1)/float(last),0)) # previous percentage
        perc = int(round(100.0*float(current)/float(last),0)) # current percentage
    except:
        perc_old = 0.0
        perc = 100.0
    
    if(current<=first):
        # If this is the first function call
        messageBoard.configure(state=tk.NORMAL)
        messageBoard.insert(tk.END,printtext)
        messageBoard.insert(tk.END,'%d%%' %perc)
        messageBoard.configure(state=tk.DISABLED)
        
    elif(perc!=perc_old):
        # If there is an update needed
        messageBoard.configure(state=tk.NORMAL)
        messageBoard.yview(tk.END)
        e = messageBoard.index(tk.END)
        r = e[0:str.find(e,'.')]
        r = int(r)-1 
        c = len(printtext)
        e_new = str(r)+'.'+str(c)
        messageBoard.delete(e_new,tk.END)
        messageBoard.insert(tk.END,'%d%%' %perc)
        messageBoard.yview(tk.END)
        guiwindow.update_idletasks() # force GUI update
        messageBoard.configure(state=tk.DISABLED)
        
    if(current==last):
        # If this is the last function call
        messageBoard.configure(state=tk.NORMAL)
        messageBoard.insert(tk.END,'\n')
        messageBoard.yview(tk.END)
        messageBoard.configure(state=tk.DISABLED)
        guiwindow.update_idletasks() # force GUI update

#-----------------------------------------------------------------------------

def easyTkImageDisplay(guiwindow,graphic,I,**kwargs):
    """
    Simplifies the process for displaying an image in Tk Canvas widget
    Args:
        guiwindow: handle for the entire guiwindow
        graphic: handle for the Tk Canvase widget
        I: image to display
        **flag_border: set to True to display a dark green border around the image
        **forceSize: [height,width] force a specific resizing
    Returns:
        None
    Saves:
        None
    """
    
    # Parse kwargs
    try: flag_border = kwargs['flag_border']
    except: flag_border = False
    try: forceSize = kwargs['forceSize']
    except: 
        # Attempt to automatically determine the resizing
        graphic.update()
        forceSize = [graphic.winfo_height(),graphic.winfo_width()]
    
    # Format the image for the graphics window
    resizeFactor = min((forceSize[1]/I.shape[1]),(forceSize[0]/I.shape[0]))
    I_small = cv2.resize(I,(int(I.shape[1]*resizeFactor),int(I.shape[0]*resizeFactor)))
    if(flag_border):
        borderWidth = max(int(np.ceil(0.001*I_small.shape[0])),int(np.ceil(0.001*I_small.shape[1])))
        I_small[0:1*borderWidth,:] = np.array([0,121,52])
        I_small[:,0:1*borderWidth] = np.array([0,121,52])
        I_small[(I_small.shape[0]-borderWidth):I_small.shape[0],:] = np.array([0,121,52])
        I_small[:,(I_small.shape[1]-borderWidth):I_small.shape[1]] = np.array([0,121,52])

    # Update the GUI graphic
    I_out = Image.fromarray(I_small)
    I_out = ImageTk.PhotoImage(I_out,master=guiwindow)
    image_on_canvas_analysis = graphic.create_image(0,0,anchor=tk.NW,image=I_out)
    graphic.itemconfig(image_on_canvas_analysis,image=I_out)
    graphic.image = I_out
    graphic.update_idletasks()
    
#-----------------------------------------------------------------------------

def easyTextField(guiwindow,windW,titletext,default):
    """
    Simplifies the process of creating a text field widget
    Args:
        guiwindow: handle for the parent GUI window
        windW: width of the GUI window
        titletext: text for the text above the field
        default: [string] default value for the field
    Returns:
        title: handle for the text above the field
        field: handle for the field
        input_var: the variable associated with the widget
    Saves:
        None
    """
    
    title = tk.Label(guiwindow,text='\n'+titletext,fg=guiColor_white,bg=guiColor_offwhite,font=(guiFontType_normal,guiFontSize_large),height=2,width=len(titletext)+1,anchor='w')
    field = tk.Entry(guiwindow,textvariable=tk.StringVar(guiwindow),fg=guiColor_black,bg=guiColor_white,font=(guiFontType_normal,guiFontSize_large),width=int(0.5*windW))
    field.insert(0,default)
    
    return title, field

#-----------------------------------------------------------------------------

def safeTextEntry(flag_okay_in,field,dtype,**kwargs):
    """
    Checks for invalid data entries, coercion can be turned on or off
    Args:
        flag_okay_in: True if the output value is now valid
        field: handle for the text entry field
        dtype: ['string','stringorempty','path','int','float','bool'] data type
        **vmin: minimum valid value
        **vmax: maximum valid value
        **illegal: list of illegal characters
        **coerce: set to True to enforce coercion using vmin, vmax, or illegal
    Returns:
        value: the typecast and coerced value
        flag_okay_out: True if the output value is now valid
    Saves:
        None
    """
    
    # Parse the kwargs
    try: vmin = kwargs['vmin']
    except: vmin = None
    try: vmax = kwargs['vmax']
    except: vmax = None
    try: illegal = kwargs['illegal']
    except: illegal = None
    try: coerce = kwargs['coerce']
    except: coerce = False
    
    # Initialize
    flag_okay = True
    value = field.get()
    
    # Check for problems
    if(dtype=='string'):
        if(illegal is not None):
            for ichar in illegal:
                if(coerce):
                    value = value.replace(ichar,'')
                else:
                    if(value.find(ichar)!=-1): flag_okay = False
        if(len(value)==0): 
            value = '*'
            flag_okay = False
    if(dtype=='stringorempty'):
        if(illegal is not None):
            for ichar in illegal:
                if(coerce):
                    value = value.replace(ichar,'')
                else:
                    if(value.find(ichar)!=-1): flag_okay = False
    elif(dtype=='path'):
        value = value.replace('\\','/')
        for ichar in ['*','<','>',':','"','|','?']: value = value.replace(ichar,'')
        if(value[-1]!='/'): value += '/'
        if(len(value)==0): 
            value = '*'
            flag_okay = False
    elif(dtype=='int'):
        try: 
            value = int(np.round(float(value),0))
            if(vmin is not None):
                if(value<vmin): 
                    if(coerce): value = vmin
                    else: flag_okay = False
            if(vmax is not None):
                if(value>vmax): 
                    if(coerce): value = vmax
                    else: flag_okay = False
        except: flag_okay = False
    elif(dtype=='float'):
        try: 
            value = float(value)
            if(vmin is not None):
                if(value<vmin): 
                    if(coerce): value = vmin
                    else: flag_okay = False
            if(vmax is not None):
                if(value>vmax): 
                    if(coerce): value = vmax
                    else: flag_okay = False
        except: flag_okay = False
    elif(dtype=='bool'):
        if((value=='True')|(value=='true')|(value=='T')|(value=='t')|(value=='Yes')|(value=='yes')|(value=='Y')|(value=='y')):
            value = True
        elif((value=='False')|(value=='false')|(value=='F')|(value=='f')|(value=='No')|(value=='no')|(value=='N')|(value=='n')):
            value = False
        else:
            flag_okay = False
        
    # Set the text color
    if(flag_okay): field.configure(fg=guiColor_black)
    else: field.configure(fg=guiColor_cherryred)
    
    # Combine the okay flags
    flag_okay_out = flag_okay&flag_okay_in
        
    return value, flag_okay_out
    
#-----------------------------------------------------------------------------

def easyDropdown(guiwindow,titletext,options):
    """
    Simplifies the process of creating a dropdown menu widget
    Args:
        guiwindow: handle for the parent GUI window
        titletext: text for the text above the field
        options: list of options to include in the dropdown
    Returns:
        dropdown: handle for the widget
        dropdown_var: the variable associated with the widget
    Saves:
        None
    """
    
    title = tk.Label(guiwindow,text=titletext,fg=guiColor_black,bg=guiColor_offwhite,font=(guiFontType_normal,guiFontSize_large),height=1,width=len(titletext)+1)
    dropdown_var = tk.StringVar(guiwindow)
    dropdown = tk.OptionMenu(guiwindow,dropdown_var,*[''])
    dropdown.config(fg=guiColor_black,bg=guiColor_white,font=(guiFontType_normal,guiFontSize_large),state=tk.NORMAL)
    menu = dropdown["menu"]
    menu.delete(0, "end")
    for string in options:
        menu.add_command(label=string,command=lambda value=string: dropdown_var.set(value))    
    dropdown_var.set(options[0])
    
    return title, dropdown, dropdown_var

#-----------------------------------------------------------------------------

def popupTextEntry(titletext):
    """
    Opens a single text entry box
    Args:
        titletext: short string to provide the user with instructions
    Returns:
        userChoice: the user's entry as a string
    Saves:
        None
    """
    
    # Define button callbacks
    global userEntry
    def actionClose(*args):
        global userEntry
        userEntry = entry0.get()
        entrywindow.unbind('<Return>')
        entrywindow.quit()
        entrywindow.destroy()
    
    # Configure the widget's window
    entrywindow = tk.Toplevel()
    entrywindow.title(str(softwareName))
    windW = 400
    windH = 125
    entrywindow.geometry(str(windW)+'x'+str(windH))
    entrywindow.configure(background=guiColor_offwhite)
    entrywindow.resizable(width=False, height=False)
    
    # Set the initial window location
    entrywindow.geometry("+{}+{}".format(int(0.5*(entrywindow.winfo_screenwidth()-windW)),int(0.5*(entrywindow.winfo_screenheight()-windH))))
    
    # Configure to handle use of the Windows close button
    entrywindow.protocol('WM_DELETE_WINDOW',actionClose)
    
    # Set up the title
    title = tk.Label(entrywindow,text=titletext,fg=guiColor_white,bg=guiColor_offwhite,font=(guiFontType_normal,guiFontSize_large),height=1,width=len(titletext)+1)
    
    # Set up GUI elements
    input0_var = tk.StringVar(entrywindow)
    input0_var.set('')
    entry0 = tk.Entry(entrywindow,textvariable=input0_var,fg=guiColor_offwhite,bg=guiColor_white,font=(guiFontType_normal,guiFontSize_large),width=int(0.07*windW))
    buttonApply = tk.Button(entrywindow,text='Apply',fg=guiColor_black,bg=guiColor_hotpink,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.02*windW),command=actionClose)
    
    # Configure and place the gui elements
    title.pack(fill=tk.X)
    entry0.pack(pady=5)
    buttonApply.pack(pady=5)
    entrywindow.bind('<Return>',actionClose)
    entry0.lift() 
    entrywindow.focus_set()
    entry0.focus_set()
    
    # Run the GUI
    entrywindow.mainloop()

    return userEntry

#-----------------------------------------------------------------------------

def removeIllegalCharacters(string,illegalchars):
    """
    Removes illegal characters from a string
    Args:
        string: raw input string
        illegalchars = list of illegal characters
    Returns:
        string: cleaned output string
    Saves:
        None
    """
    
    for ichar in illegalchars:
        string = string.replace(ichar,'')
        
    return string

#-----------------------------------------------------------------------------

def openWorkbook(sheetNames,sheetHeaders):
    """ 
    Opens a workbook on the hardisk 
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
            sheets[i].write(0,j,sheetHeaders[i][j])
    
    return book, sheets

#------------------------------------------------------------------------------

def saveToWorkbook(book,sheets,data,**kwargs):
    """ 
    Saves data into an already-opened workbook
    Args:
        book: workbook object
        sheets: list of sheet objects
        data: list of numpy data arrays [sheet][column][rows (vector of data)]
        **cell_colors: list of cell colors (by column) accepted by xlwt
    Returns:
        flag_writeok: is returned as True if the data write was successfull
    Saves:
        None
    """
    
    # Parse kwargs
    try: cell_colors = kwargs['cell_colors']
    except: cell_colors = []
    
    try: 
        for i in range(0,len(sheets),1):
            # Each sheet
            for j in range(0,len(data[i]),1):
                # Each column
                for k in range(0,len(data[i][j]),1):
                    # Each row (writable datum)
                    try: d = data[i][j][k].astype(float)
                    except: d = data[i][j][k]
                    
                    # Optional cell coloring
                    if(len(cell_colors)==0):
                        sheets[i].write(k+1,j,d)
                    else:
                        cell_color = cell_colors[i][j]
                        if(cell_color!='none'):
                            styleString = 'pattern: pattern solid, fore_colour %s;' %(cell_color)
                            style = xlwt.easyxf(styleString)
                            sheets[i].write(k+1,j,d,style)
                        else:
                            sheets[i].write(k+1,j,d)
                    
                    
                    
        flag_writeok = True
    except:
        flag_writeok = False

    return flag_writeok

#------------------------------------------------------------------------------
    
def saveWorkbook(book,filepath):
    """ 
    Saves a workbook to the hardisk
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


#-----------------------------------------------------------------------------

def safeImageLoad(path,mode):
    """ 
    Safely loads the image with the requested filename by cycling through all 
    recognized file extensions
    Args:
        path: file path of the target image, including the filename but sans the extension
        mode: [cv2.IMREAD_COLOR,cv2.IMREAD_GRAYSCALE,cv2.IMREAD_UNCHANGED] set the cv2.imread color import options
    Returns:
        I: the loaded image
    Saves:
        None
    """
    
    # Try loading the image with each possible file extension until a working one is found
    for ext in ['.jpg','.jpeg','.png','.tif','.tiff']:
        I = cv2.imread(path+ext,mode)
        try: 
            I.any() 
            break
        except: pass
    
    # Raise an exception if none of the accepted file extensions worked
    try: I.any()
    except: raise Exception('Image could not be loaded')
    
    return I

#-----------------------------------------------------------------------------

def convertColorSpace(I_in):
    """ 
    Converts a color image in BGR space into a color image in RGB space or 
    vice versa
    Args:
        I_in: image in original color space
    Returns:
        I_out: image in new space
    Saves:
        None
    """
    
    if(I_in.shape[2]==3):
        I_out = np.zeros_like(I_in)
        I_out[:,:,0] = I_in[:,:,2]
        I_out[:,:,1] = I_in[:,:,1]
        I_out[:,:,2] = I_in[:,:,0]
    elif(I_in.shape[2]==4):
        I_out = np.zeros_like(I_in)
        I_out[:,:,0] = I_in[:,:,2]
        I_out[:,:,1] = I_in[:,:,1]
        I_out[:,:,2] = I_in[:,:,0]
        I_out[:,:,3] = I_in[:,:,3]
    else:
        I_out = I_in # unrecognized input
        
    return I_out

#-----------------------------------------------------------------------------

def improveVisualization(I,alpha):
    """ 
    Performs a contrast adjustment to enhance the visibility of a powder bed image
    Note: this is NOT intended for use by the DSCNN
    Args:
        I: original image
        alpha: [0 - 127] contrast adjustment
    Returns:
        I: enhanced image
    Saves: 
        None
    """
    
    # Parameters
    thresh_pass = 0.10 # [0 - 1] avoids image enhancement if the image intensities are naturally flat
    
    # Histogram normalization
    I_pass = np.copy(I) # save a copy with no modifications
    I = I.astype(float) # cast for calculations
    I += - np.min(I) # shift to uint8 floor
    [_,binEdges] = np.histogram(I,10)
    I += -binEdges[1] # shift down
    I *= (255.0/binEdges[8]) # pull up
    I = np.maximum(I,np.zeros_like(I)) # clip lower
    I = np.minimum(I,255*np.ones_like(I)) # clip upper
    I = I.astype(np.uint8) # recast
    if(abs(binEdges[1]-binEdges[8])<(thresh_pass*255.0)):
        # If the image is really flat, don't enhance it
        I = I_pass
    
    return I

#-----------------------------------------------------------------------------

def imageFuse(I_raw,mask_labels,classColors,**kwargs):
    """ 
    Fuses a label mask onto an overlay image 
    Args:
        I_raw: the underlying image 
        mask_labels: non-binary mask of anomaly classifications
        classColors: list of RGB class colors 
        **flag_enhance: set to True to enhnace the background image for display
        **window: boundaries of a zoom window
    Return:
        I_fused: the overlay image, in the format of a three-channel color image
    Saves:
        None
    """
    
    # Check color formatting of the input image
    numDims = len(I_raw.shape)
    if(numDims==3):
        if(I_raw.shape[2]==3):
            # Three channel image
            I = np.copy(I_raw)
        else:
            # Four channel image
            I = I_raw[0:3]
    else:
        # Single channel image
        I = np.zeros((I_raw.shape[0],I_raw.shape[1],3),np.uint8)
        I[:,:,0] = I_raw
        I[:,:,1] = I_raw
        I[:,:,2] = I_raw
    
    # Parse kwargs with defaults
    try: flag_enhance = kwargs['flag_enhance']
    except: flag_enhance = False
    try: window = kwargs['window']
    except: window = []
    
    # Apply a gamma correction to the background powder bed image
    if(flag_enhance): I = improveVisualization(I,127)
    
    # Crop down to a window if zoomed in
    if(len(window)!=0):
        I = I[window[0]:window[1],window[2]:window[3],:]
        mask_labels = mask_labels[window[0]:window[1],window[2]:window[3]]
    
    # Initialize the color channels
    mask_red = np.zeros((I.shape[0],I.shape[1]),float)
    mask_green = np.zeros((I.shape[0],I.shape[1]),float)
    mask_blue = np.zeros((I.shape[0],I.shape[1]),float)
    
    # Classes
    for i in range(0,len(classColors),1):
        locs = (mask_labels==i)
        mask_red[locs] = classColors[i][0]
        mask_green[locs] = classColors[i][1]
        mask_blue[locs] = classColors[i][2]
        
    # Alpha settings
    if(flag_enhance): alpha_overlay = 0.4
    else: alpha_overlay = 0.4
    alpha_background = 1.0 - alpha_overlay
    
    # Set the color channels
    I_fused = np.zeros((I.shape[0],I.shape[1],3),np.uint8)
    I_fused[:,:,0] = alpha_overlay*mask_red + alpha_background*I[:,:,0] # update the red channel
    I_fused[:,:,1] = alpha_overlay*mask_green + alpha_background*I[:,:,1] # update the green channel
    I_fused[:,:,2] = alpha_overlay*mask_blue + alpha_background*I[:,:,2] # update the blue channel
    
    # Brighten the transparent overlays
    locs = (mask_labels==-1).astype(float)
    I_fused[:,:,0] = I_fused[:,:,0] + (alpha_overlay*I[:,:,0]*locs).astype(np.uint8)
    I_fused[:,:,1] = I_fused[:,:,1] + (alpha_overlay*I[:,:,1]*locs).astype(np.uint8)
    I_fused[:,:,2] = I_fused[:,:,2] + (alpha_overlay*I[:,:,2]*locs).astype(np.uint8)
    
    return I_fused

#-----------------------------------------------------------------------------
    
def smartRealImageDisplay(I,actualSize,titletext,**kwargs):
    """ 
    Very flexible API for automatic formatting an image with real units
    Args:
        I: image to be displayed
        actualSize: the size of the image in real units
        titletext: the text to display in the matplotlib window header
        **h_fig: allows a figure handle to be created on the outside (as for a timelapse)
        **h_im: allows a plt.imshow handle to be passed, allowing for reduced figure generation time (as for a timelapse)
        **origin: ['bottomleft', 'center'] moves the origin of the coordinate system
        **x_real: label for the x-axis
        **y_real: label for the y-axis
        **units: units for the axes (same for both)
        **bannertext: the text to display across the top, within the figure window itself
        **legendNames: list of names in the legend
        **legendColors: list of RGB arrays for the legend
        **flag_grid: set to True to display the grid lines
    Returns:
        h_fig: handle for the figure
        h_im: handle for the image
        ax: handle for the axes
        h_lgd: handle for the legend
    Saves:
        None
    """
    
    # Parse kwargs with defaults
    try: h_fig = kwargs['h_fig']
    except: h_fig = None
    try: h_im = kwargs['h_im']
    except: h_im = None
    try: origin = kwargs['origin']
    except: origin = 'bottomleft'
    try: x_real = kwargs['x_real']
    except: x_real = 'x'
    try: y_real = kwargs['y_real']
    except: y_real = 'y'
    try: units = kwargs['units']
    except: units = 'pixels'
    try: bannertext = kwargs['bannertext']
    except: bannertext = None
    try: legendNames = kwargs['legendNames']
    except: legendNames = None
    try: legendColors = kwargs['legendColors']
    except: legendColors = None
    try: flag_grid = kwargs['flag_grid']
    except: flag_grid = False
    
    # Create a new figure window
    imageSize = [I.shape[1],I.shape[0]] # [width, height] size of the input image
    if(h_fig is None):
        if(imageSize[0]>=imageSize[1]): figSize_adj = [figSize[0],(imageSize[1]/imageSize[0])*figSize[1]]
        else: figSize_adj = [(imageSize[0]/imageSize[1])*figSize[0],figSize[1]]
        h_fig = plt.figure(titletext,figsize=figSize_adj,facecolor='w')
    
    # Set the plot axes
    ax = plt.gca() # get current axis
    ax.axis('image') # set axis type
    ax.set_xlim((0,imageSize[0]))        
    ax.set_ylim((imageSize[1],0))
    
    # Tick mark settings
    if(actualSize[0]>actualSize[1]):
        numxticks = 10 # number of ticks to show along the x axis
        numyticks = max(1,int(np.round((actualSize[1]/actualSize[0])*numxticks,0))) # auto-scaled number of ticks to show along the y axis  
    else:
        numyticks = 10 # number of ticks to show along the y axis
        numxticks = max(1,int(np.round((actualSize[0]/actualSize[1])*numyticks,0))) # auto-scaled number of ticks to show along the y axis  
    scale = [float(imageSize[0])/float(actualSize[0]),float(imageSize[1])/float(actualSize[1])] # [width, height]
    
    # Determine the correct number of digits to display
    try: xdigits = len(str(actualSize[0]).split('.')[1]) 
    except: xdigits = 0
    try: ydigits = len(str(actualSize[1]).split('.')[1]) 
    except: ydigits = 0
    digits = min(2,max([xdigits,ydigits]))
    
    # Configure the tick marks and grid lines
    xticks_num = np.linspace(0,imageSize[0],numxticks+1)/scale[0]
    yticks_num = np.linspace(0,imageSize[1],numyticks+1)/scale[1]
    if(origin=='bottomleft'): pass
    elif(origin=='center'): 
        xticks_num = xticks_num - 0.5*max(xticks_num)
        yticks_num = yticks_num - 0.5*max(yticks_num)
    xticks_num = np.round(xticks_num,digits)
    yticks_num = np.round(yticks_num,digits) 
    xticks_str = []
    yticks_str = []
    for i in xticks_num: xticks_str.append(str(i))
    for i in yticks_num: yticks_str.append(str(i))
    yticks_str = np.flip(yticks_str,0)
    if(flag_grid):
        ax.set_xticklabels(xticks_str)
        ax.set_yticklabels(yticks_str)
    else:
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.tick_params(axis='x',which='both',bottom=False,top=False)
        ax.tick_params(axis='y',which='both',right=False,left=False)
    xstep = scale[0]*actualSize[0]/(numxticks)
    ystep = scale[1]*actualSize[1]/(numyticks)
    xticklocs = np.zeros((numxticks+1),float)
    yticklocs = np.zeros((numyticks+1),float)
    xticklocs[0:numxticks] = np.arange(0,imageSize[0]-0.1,xstep)
    yticklocs[0:numyticks] = np.arange(0,imageSize[1]-0.1,ystep)
    xticklocs[numxticks] = imageSize[0]
    yticklocs[numyticks] = imageSize[1]
    if(flag_grid):
        ax.set_xticks(xticklocs,minor=False)
        ax.set_yticks(yticklocs,minor=False)
        plt.grid(b=None,which='major',axis='both',color=np.array([0.8,0.8,0.8]))    
    
    # Label the axes
    if(flag_grid):
        plt.xlabel('%s (%s)' %(x_real,units))
        plt.ylabel('%s (%s)' %(y_real,units))
          
    # Display the image
    if(h_im is None):
        if(I.ndim==3): h_im = plt.imshow(I,vmin=0,vmax=255,interpolation='none',extent=[0,imageSize[0],imageSize[1],0]) # RGB color image
        else: h_im = plt.imshow(I,cmap='gray',vmin=0,vmax=255,interpolation='none',extent=[0,imageSize[0],imageSize[1],0]) # grayscale image
    else: h_im.set_data(I) # pre-existing image
        
    # Generate and display a legend  
    legend_elements = []
    if((legendNames is not None) and (legendColors is not None)):  
        numLegendItems = min(len(legendNames),len(legendColors)) # number of legend items
        for i in range(0,numLegendItems,1):
            legend_elements.append(Patch(facecolor=(legendColors[i]).astype(float)/255,edgecolor='k',label=legendNames[i]))
        anchorScale = 0.25*(imageSize[0]/imageSize[1])
        h_lgd = ax.legend(handles=legend_elements,bbox_to_anchor=(1.00+anchorScale,1.00))
    else: h_lgd = None
    
    # Place the banner text
    if(bannertext!=None): plt.title(bannertext)
        
    # Force the plot to update
    h_fig.tight_layout(rect=[0, 0.05, 1, 1]) 
    plt.draw()
    
    return h_fig, h_im, ax, h_lgd

#-----------------------------------------------------------------------------

def popupMultichoice(titletext,options):
    """ Opens a multiple choice menu with no restriction on the number of choices
    Args:
        titletext: short string to provide the user with instructions
        options: list of options
    Returns:
        userChoice: the user's selection as the option string
    Saves:
        None
    """
    
    # Define button callbacks
    global userEntry
    def actionClose(*args):
        global userEntry
        userEntry = dropdown0_var.get()
        entrywindow.unbind('<Return>')
        entrywindow.quit()
        entrywindow.destroy()
    
    # Configure the widget's window
    entrywindow = tk.Tk()
    entrywindow.title(str(softwareName))
    windW = 400
    windH = 125
    entrywindow.geometry(str(windW)+'x'+str(windH))
    entrywindow.configure(background=guiColor_offwhite)
    entrywindow.resizable(width=False, height=False)
    
    # Set the initial window location
    entrywindow.geometry("+{}+{}".format(int(0.5*(entrywindow.winfo_screenwidth()-windW)),int(0.5*(entrywindow.winfo_screenheight()-windH))))
    
    # Configure to handle use of the Windows close button
    entrywindow.protocol('WM_DELETE_WINDOW',actionClose)
    
    # Set up GUI elements
    buttonApply = tk.Button(entrywindow,text='Apply',fg=guiColor_white,bg=guiColor_darkgreen,font=(guiFontType_normal,guiFontSize_large),height=1,width=int(0.02*windW),command=actionClose)
    [title0,dropdown0,dropdown0_var] = easyDropdown(entrywindow,titletext,options)
    
    # Configure and place the gui elements
    title0.pack(fill=tk.X)
    dropdown0.pack(pady=5)
    buttonApply.pack(pady=5)
    entrywindow.bind('<Return>',actionClose)
    entrywindow.focus_set()
    dropdown0.focus_set()
    
    # Run the GUI
    entrywindow.mainloop()

    return userEntry