# -*- mode: python -*-

import sys
sys.setrecursionlimit(5000)

hiddenimports=['cython']

block_cipher = None


a = Analysis(['Main.py'],
             pathex=['D:\\Temp\\code'],
             binaries=[],
             datas=[('platforms/zlib.dll','platforms')],
			 hiddenimports=['cython', 'pkg_resources.py2_warn', 'matplotlib', 'matplotlib.pyplot'],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='4265 Path Planner',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          console=False , icon='logo.ico')
