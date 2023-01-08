# -*- mode: python -*-

# Set recursion limit
import sys
sys.setrecursionlimit(5000)

# Import modules
import glob
import os

# Search for any DLLs that aren't added automatically
print('\nPackaging DLLs:')
other_dlls = []
for dll in glob.glob('platforms/*.dll'):
	other_dlls.append((str(dll), '.'))
missing_dlls = other_dlls
print('\nDLLs:')
for dll in missing_dlls: print(dll)

# Search for any data files to include
print('\nPackaging data files:')
data_files = [(str(data), '.') for data in glob.glob('datas/*.*')]
data_files += [('platforms/zlib.dll','platforms')]
print('\nData Files:')
for data in data_files: print(data)

# Specify hidden imports
print('\nAdding hidden imports:')
hidden_imports = ['cython',
				  'pkg_resources.py2_warn',
				  'matplotlib',
				  'matplotlib.pyplot']
for hidden in hidden_imports: print(hidden)

# Update the cached code
print('\nUpdating the code cache:')
paths_cache = glob.glob('cache/*.*')
for path_cache in paths_cache: os.remove(path_cache)
paths_new = glob.glob('../code/*.*')
for path_new in paths_new:
    path_movefrom = path_new.replace('/','\\')
    filename = path_movefrom.split('\\')[-1]
    print('  ',filename)
    path_moveto = 'cache\\'+filename
    os.system('copy'+' "'+path_movefrom+'" "'+path_moveto+'"')

# Automatically find the version number
h_log = open('cache/Main.py','r')
for line in h_log:
    if(line.find('versionNumber')!=-1):
        versionNumber = line.split("'")[1]
    if(line.find('versionType')!=-1):
        versionType = line.split("'")[1]
        break
versionNumber = versionNumber.replace('.','-')
if(versionType=='stable'): versionType = ''
filename = 'FRC 4265 Path Planner_v%s%s' %(versionNumber,versionType)
print('\nFreezing:',filename+'.exe\n')

sys.path.append('cache/')
a = Analysis(['cache/Main.py'],
             binaries=missing_dlls,
             datas=data_files,
             hiddenimports=hidden_imports,
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=None,
             noarchive=False)
pyz = PYZ(a.pure,
		  a.zipped_data,
          cipher=None)
exe = EXE(pyz,
          a.scripts,
		  [('W ignore', None, 'OPTION')],
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name=filename,
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=True,
		  icon='logo.ico')
