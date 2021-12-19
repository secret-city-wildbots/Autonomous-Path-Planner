# -*- mode: python -*-

# Set recursion limit
import sys
sys.setrecursionlimit(5000)

# Import modules
import glob

# Search for any DLLs that aren't added automatically
other_dlls = []
for dll in glob.glob('platforms/*.dll'):
	other_dlls.append((str(dll), '.'))
missing_dlls = other_dlls
print('\nDLLs:')
for dll in missing_dlls: print(dll)

# Search for any data files to include
data_files = [(str(data), '.') for data in glob.glob('datas/*.*')]
data_files += [('platforms/zlib.dll','platforms')]
print('\nData Files:')
for data in data_files: print(data)

# Specify hidden imports
hidden_imports = ['cython',
				  'pkg_resources.py2_warn',
				  'matplotlib',
				  'matplotlib.pyplot']
print('\nHidden Imports:')
for hidden in hidden_imports: print(hidden)

# Automatically find the version number
h_log = open('../code/Main.py','r')
for line in h_log:
    if(line.find('versionNumber')!=-1):
        versionNumber = line.split("'")[1]
    if(line.find('versionType')!=-1):
        versionType = line.split("'")[1]
        break
versionNumber = versionNumber.replace('.','-')
if(versionType=='stable'): versionType = ''
filename = 'FRC 4265 Path Planner_v%s%s' %(versionNumber,versionType)
print('\nFilename:',filename+'.exe')

print('\nFreezing...\n')
sys.path.append('../code/')
a = Analysis(['../code/Main.py'],
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
