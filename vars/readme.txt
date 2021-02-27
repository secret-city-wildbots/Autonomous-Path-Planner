
-------------------------------------------------------------------------------------
About
-------------------------------------------------------------------------------------

"4265 Path Planner" An Autonomous Path Planner for the 2021 FIRST Robotics Competition
FIRST Robotics Team 4265

-------------------------------------------------------------------------------------
Release Notes
-------------------------------------------------------------------------------------

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
