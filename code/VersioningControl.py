# Date: 2021-02-26
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
        
        # Create always-local directories
        try: os.mkdir(dirPvars)
        except: pass
        try: os.mkdir('../field drawings/')
        except: pass
        try: os.mkdir('../robot paths/')
        except: pass
        try: os.mkdir('../robot models/')
        except: pass
    
        # Save the Operating System information
        np.save(dirPvars+'ostype.npy','Windows-Mac')
            
        # Move supporting files to the correct folders
        os.system('move '+'settings.npz'+' "'+'../vars'+'"')
        os.system('move '+'graphic_4265.png'+' "'+'../vars'+'"')
        os.system('move '+'waypoint.png'+' "'+'../vars'+'"')
        
        # Notify the user of successful installation
        instructions = 'Next Steps:\n'
        instructions += '1. The software will automatically complete the rest of the installation process and then close.\n'
        instructions += '2. Double-click the "4265 Path Planner.exe" file or create a shortcut.\n'
        instructions += '3. Review the readme file.\n'
        print(instructions)
        messagebox.showinfo('Installation Instructions',instructions)

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
        notification = 'Upgrade to v%s complete. Refer to the readme.txt file for any additional instructions. The software will close automatically in 10 seconds...' %(versionNumber_current)
        print('\n'+notification+'\n')
        messagebox.showwarning('4265 Path Planner',notification)
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

"4265 Path Planner" An Autonomous Path Planner for the 2021 FIRST Robotics Competition
FIRST Robotics Team 4265

-------------------------------------------------------------------------------------
Release Notes
-------------------------------------------------------------------------------------

v***
> Overhauls the path saving code.
> Adds a seperate button for editing a waypoint.
> Fixes a bug in the x-axis label display.
> Radio group is fixed custom toolbar buttons.

v2.1.0
> Fixes a minor installation bug.
> Placing waypoints is now a left-click.
> Overhauls the GUI to improve stability.
> The main GUI window can now be resized in the vertical direction.

v2.0.4
> Corrects for more pathing issues when the waypoint velocities are set to 0 ft/s.

v2.0.3
> Cleans up the buttons on the field display GUI.
> Fixes critical pathing bugs which occurred when waypoint velocities of 0 ft/s were entered.
> The robot will now accelerate as hard as possible if it missed its previous velocity waypoint and is still requested to accelerate.

v2.0.2
> Allows users to enter a minimum robot velocity of 0 ft/s.
> The planner now times out if no pathing solution can be found.
> Adds a manual DPI scaling override to improve rendering on newer laptop screens.
> Improves the installation process.
> Fixes a bug which can occur during waypoint insertion.

v2.0.0
> Users can now change the order of a way point.
> Users can now mark some points as "must touch" points for later interpretation by the path follower.
> Simplifies the GUI and fixes several bugs that could cause the interface to crash.
> User default settings will now persist between software restarts.
> Overhauls the pathing algorithm. Multiple edge cases fixed, point spacing is now more consitent, turn radii can be entered manually, and accelleration limiting is more accuracte.

v1.2.0
> Fixes a critical bug that prevented any user-configurable settings from actually being updated via the GUI.
> Adds acceleration limiting.
> Fixes a bug with the <Enter> key when saving a path.

v1.0.0
> This is the first release version, all previous versions existed only in beta.

-------------------------------------------------------------------------------------
"""
    
    return text