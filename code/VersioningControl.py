# Date: 2020-06-09
# Description: auto-generates the readme.txt file and handle software upgrades
# and installation
#-----------------------------------------------------------------------------

# Load external libraries
import numpy as np # math operations
import os # hooks into Windows operating system
from tkinter import messagebox # TkInter popup windows

# Hardcoded directory paths
dirPvars = '../vars/' # persistent variables directory

#-----------------------------------------------------------------------------

def install():
    """
    Performs intial software installation
    Args:
        None
    Returns:
        None
    Saves:
        A bunch
    """
    
    try: 
        
        # Test to determine if installation is necessary
        np.load(dirPvars+'ostype.npy')
        
    except: 
        
        # Indicate that the installation is begining 
        print('Installing the 4265 Path Planner...')
        
        # Ask the user which Operating System they are using 
        userInput = messagebox.askyesno(' 4265 Path Planner','Are you installing on a Linux system?')
        if(userInput): ostype = 'Linux'
        else: ostype = 'Windows-Mac'
        
        # Save the Operating System information
        np.save(dirPvars+'ostype.npy',ostype)
        
        # Create always-local directories
        try: os.mkdir(dirPvars)
        except: pass
        try: os.mkdir('../workspaces')
        except: pass
            
        # Move supporting files to the correct folders
        os.system('move '+'graphic_default.png'+' "'+'../vars'+'"')
        os.system('move '+'graphic_4265.png'+' "'+'../vars'+'"')
        
        # Notify the user of successful installation
        instructions = 'Next Steps:\n'
        instructions += '1. The software upgrade process will begin automatically.\n'
        instructions += '2. The software will close.\n'
        instructions += '3. Double-click the "4265 Path Planned.exe" file or create a shortcut.\n'
        instructions += '4. Review the readme file.\n'
        messagebox.showinfo('Installation Instructions',instructions)
        print('Initial installation complete.')

#-----------------------------------------------------------------------------

def upgrade(versionNumber_current):
    """ Creates the readme.txt file if the software version number has changed 
        and updates the settings files
    Args:
        versionNumber_current: the current version number as a string "x.x.x"
    Returns:
        flag_upgraded: True if the software version just changed
    Saves:
        readme.txt: readme file including documentation a change log
    """
    
    # Load the previous version number
    try: versionNumber_old = np.load(dirPvars+'versionNumber.npy')
    except: versionNumber_old = '0.0.0'
    
    if(versionNumber_current!=versionNumber_old):
        
        # Notify the user
        print('Finalizing upgrade from v%s...' %(versionNumber_old))
        
        # Update the current version number 
        try: os.remove(dirPvars+'versionNumber.npy')
        except: pass
        np.save(dirPvars+'versionNumber',versionNumber_current)
        
        # Delete the old readme file
        try: os.remove(dirPvars+'readme.txt')
        except: pass
                            
        # Save the new readme text to the file
        h_readme = open(dirPvars+'readme.txt',"w") 
        h_readme.write(textReadme())
        h_readme.close() 
        
        # Report out that the software was upgraded
        print('\nUpgrade to v%s complete. Your settings files have been automatically upgraded.' %(versionNumber_current))
        print('The software will close automatically in 10 seconds...')
        print('Refer to the changelog in the readme.txt file for any additional instructions.')
        print('\n\n\n\n')
        flag_upgraded = True
        
    else: flag_upgraded = False
        
    return flag_upgraded

#-----------------------------------------------------------------------------
    
def textReadme():
    """ Contains the readme text itself
    Args:
        None
    Returns:
        text: actual text of the readme file including documentation a change log
    Saves:
        None
    """
    
    text = """
-------------------------------------------------------------------------------------
About
-------------------------------------------------------------------------------------

"4265 Path Planner" An Autonomous Path Planner for FRC 2020
FIRST Robotics Team 4265

-------------------------------------------------------------------------------------
Quick Help
-------------------------------------------------------------------------------------

1. File
1a. Load Field Map
***

1b. Quit
Cleanly exits the software. This is functionally identical to selecting the close
button in the upper-right corner.

-------------------------------------------------------------------------------------
Release Notes
-------------------------------------------------------------------------------------

v1.0.0
-This is the first release version, all previous versions existed only in beta.

-------------------------------------------------------------------------------------
"""
    
    return text