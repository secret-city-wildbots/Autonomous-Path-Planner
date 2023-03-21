# Date: 2023-01-15
# Description: auto-generates the readme.txt file and handle software upgrades
# and installation
#-----------------------------------------------------------------------------

# Load external libraries
import numpy as np # math operations
import os # hooks into Windows operating system
import shutil # used for copying files
import sys # access to the OS
from tkinter import messagebox # TkInter popup windows
from win32com.client import Dispatch # used for creating the desktop shortcut

# Hardcoded directory paths
dirPvars = '../vars/' # persistent variables directory

#-----------------------------------------------------------------------------

def resourcePath(relative_path):
    """
    Handles file paths transparently for development and pyinstaller
    """
    
    # Choose between the development path and the pyinstaller temp folder path
    try: base_path = sys._MEIPASS
    except: base_path = os.path.abspath(".")
    
    return os.path.join(base_path,relative_path)

#-----------------------------------------------------------------------------

def unpackResource(filename):
    """
    Unpacks the specified resource file to the vars/ directory
    """
    
    try: shutil.copyfile(resourcePath(filename),dirPvars+filename)
    except: pass

#-----------------------------------------------------------------------------

def install(argv): 
    """
    Performs intial software installation
    Args:
        argv: command line arguments
    Returns:
        None
    Saves:
        A bunch
    """
    
    # Check if unpacking has to happen first
    flag_unpack = os.getcwd().split('\\')[-1]!='code'
    
    if(flag_unpack):
        
        # Set the installation directory
        path_install = 'C:/Program Files (x86)/'
        
        # Create the folder structure
        print('Creating the folder structure...')
        try: os.mkdir(path_install+'/FRC 4265 Path Planner')
        except: pass
        try: os.mkdir(path_install+'/FRC 4265 Path Planner/code')
        except: pass
    
        # Copy the executable files
        print('Unpacking the executable files...')
        path_src = argv[0].replace('\\','/')
        shutil.copyfile(path_src,path_install+'/FRC 4265 Path Planner/code/FRC 4265 Path Planner.exe')
        print('Executable copied successfully.')
        
        # Change working directories
        print('Changing the working directory...')
        os.chdir(path_install+'/FRC 4265 Path Planner/code')
    
    try: 
        
        # Test to determine if installation is necessary
        np.load(dirPvars+'ostype.npy')
        
    except: 
        
        # Indicate that the installation is begining 
        print('Installing the 4265 Path Planner...')
        messagebox.showinfo('4265 Path Planner','Select OK to continue installing the FRC 4265 Path Planner.')
        
        # Create always-local directories
        try: os.mkdir(dirPvars)
        except: pass
        try: os.mkdir('../field drawings/')
        except: pass
        try: os.mkdir('../robot paths/')
        except: pass
        try: os.mkdir('../robot models/')
        except: pass
    
        # Ask the user if they would like to create a desktop shortcut
        create_shortcut = messagebox.askyesno('4265 Path Planner','Would you like to create a desktop shortcut?')
        if(create_shortcut):
            try:
                print('Creating a desktop shortcut...')
                desktop = os.path.expanduser('~/Desktop')
                path = os.path.join(desktop,'FRC 4265 Path Planner.lnk')
                target = os.path.join(os.getcwd(),'FRC 4265 Path Planner.exe')
                wDir = os.getcwd()
                icon = os.path.join(os.getcwd(),'FRC 4265 Path Planner.exe')
                shell = Dispatch('WScript.Shell')
                shortcut = shell.CreateShortCut(path)
                shortcut.Targetpath = target
                shortcut.WorkingDirectory = wDir
                shortcut.IconLocation = icon
                shortcut.save()
            except: print('Error: Could not create the desktop shortcut.')
            
        # Save the Operating System information
        np.save(dirPvars+'ostype.npy','Windows')
        
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
        print('Finalizing upgrade from v%s to v%s...' %(versionNumber_old,versionNumber_current))
        
        # Move supporting files to the correct folders
        unpackResource('graphic_4265.png')
        unpackResource('addwaypoint.png')
        unpackResource('editwaypoint.png')
        unpackResource('savepath.png')
        unpackResource('probe.png')
        unpackResource('movewaypoint.png')
        
        # Delete the old readme file
        try: os.remove(dirPvars+'readme.txt')
        except: pass
                            
        # Save the new readme text to the file
        h_readme = open(dirPvars+'readme.txt',"w") 
        h_readme.write(textReadme())
        h_readme.close() 
        
        # Update the current version number 
        try: os.remove(dirPvars+'versionNumber.npy')
        except: pass
        np.save(dirPvars+'versionNumber',versionNumber_current)
        
        # Report out that the software was upgraded
        notification = 'Installation of v%s of the FRC 4265 Path Planner is complete, please review the release notes for any important changes. Select OK to close the installer and then you may restart the software using the shortcut.' %(versionNumber_current)
        print('\n'+notification+'\n')
        messagebox.showinfo('4265 Path Planner',notification)
        flag_upgraded = True
        
        # Open the release notes
        os.startfile('%s' %(os.path.abspath('.').replace('\\','/')+'/../vars/readme.txt'))
        
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

The "4265 Path Planner" an autonomous path planner for the FIRST Robotics Competition, created by FIRST Robotics Team 4265.

-------------------------------------------------------------------------------------
Release Notes
-------------------------------------------------------------------------------------

v2.3.3
> Fixing a critical bug regarding the blue-side calibration points.

v2.3.2
> Fixes a bug in how the starting acceleration is handled.
> Calibration points can now be measured relative to an arbitrary number of different reference points.
> Fixes a bug introduced in v2.3.1 with the calibration points.

v2.3.1
> Waypoints can now be flipped about the x or y axes by enditing a waypoint and placing a negative sign in front of the desired coordinate.
> Changes the path velocity color map.
> Robot and waypoint colors will adjust automatically based on the alliance selection.
> The user can now specify which alliance they are planning a path for so that the field is flipped the correct direction.
> The installer will no longer fail if the desktop shortcut can't be created.
> Updates the file copying backend from os to shutil to fix issues with different user accounts unpacked resource files.

v2.3.0
> Fixes a bug with the custom toolbar icons.
> Allows the path planner version to be updated by just double-clicking a new executable.
> Fixes a minor issue with how the default settings are delayed.
> Hard-codes the installation directory to reduce chances for user error.
> Prevents the main GUI from locking if a bad field image is selected by the user.
> Adding field drawings and initial robot models for the 2023 Charged Up game.
> Updating the default field dimensions for the 2023 Charged Up game.
> Updating the packages for 2023.

v2.2.8
> Field calibration points can now be used to aid the user in adjusting paths for competition fields.
> Users can now click a waypoint to move it with the mouse.
> Users can now reset file selections from the GUI.
> The rotational cruise velocity fraction can now be configured as a setting.
> Tweaks the GUI layout.
> Hardcoded constants are now handled more robustly.

v2.2.7
> Adds an acceleration phase to the rotational velocity feed-forward.
> Fixes orientation some overshooting issues.

v2.2.6
> Adds a feed-forward rotational velocity.
> Reduces the font size on the generated path figures.
> Allows for DPI scaling values less than 100%.
> More user selections will be remembered between sessions.

v2.2.5
> Maybe this one work-o with opening old paths.

v2.2.2
> Updates the default values for the 2022 Rapid React FRC Game.

v2.2.1
> Significantly improves the initial software installation experience.

v2.1.3
> More intelligent path saving options (you can now choose the save location).
> Adds the ability to "probe" the generated path.
> Updates the button icons.

v2.1.2
> Changes "release" tag to "stable" to align with the nomenclature used for the robot projects.
> The user will now only be asked to save the path on-close if there are remaining un-saved changes.
> Tweaks to some of the GUI wording for clarity.
> Fixes a bug which occurred if the user canceled selecting a field map.

v2.1.1
> Overhauls the path saving code.
> Adds a seperate button for editing a waypoint.
> Fixes a bug in the x-axis label display.
> Radio group is fixed for custom toolbar buttons.

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