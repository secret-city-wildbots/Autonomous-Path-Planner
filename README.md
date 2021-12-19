# About
Autonomous path planner designed to integrate with a modified pure-pursuit path following algorithm. Please contact the Secret City Wildbots if you would like the Windows installation files.

Programming Lead: Amelie

Programming Lead-in-Training: Anya

Controls Mentor: Dr. Luke Scime (lrscime.alpha@gmail.com)

# Programmer Instructions
1. Clone the "main" branch (GitClone)
2. Create a new branch and switch your local repository (TortoiseGit > Switch/Checkout)
3. Develop your awesome code
4. Commit and push your changes regularly with descriptive messages (Git Commit -> "branch-name")
5. Once you are ready to integrate your changes, create a merge request (using GitHub) from your branch into the "main-arch" branch

# Programming Lead Instructions
1. Manually approve and merge any subsystem branches with open merge requests
2. Test the merged branches in combination with any architecture changes made in "main-arch"
3. Merge "remotes/origin/main" into your local copy of "main-arch"
4. Resolve any onflicts either individually or by right-clicking on files and selecting "Resolve with HEAD"
5. Confirm resolution of all conflicts
6. Update the version number inside of the code
7. Perform final testing, if a robot project, hard-deploy and confirm all functionality
8. Update the list of un-tested features and add any other important notes
9. Commit and push to "main-arch"
10. Create a merge request (using GitHub) from "main-arch" to "main" and assign a mentor to the approval role
11. Perform the automatic merge via GitHub (the request will close automatically)
12. Create a new tag (with the correct version number) and release -- a new release should be made at least once per sprint