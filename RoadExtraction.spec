# -*- mode: python ; coding: utf-8 -*-
import sys
import os.path as path
sys.setrecursionlimit(5000)

block_cipher = None
PROJECT_MAIN_DIR = "C:/Users/h2958/Desktop/RoadExtraction"
project_py_files = ['RoadExtraction.py',
                    'MainWindowUI.py',
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionAlgorithm/DetectionParameters.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionAlgorithm/GeneralSimilarityDetection.py'),

                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionAlgorithm/GraySimilarityDetection.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionAlgorithm/JumpSimilarityDetection.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionAlgorithm/NarrowSimilarityDetection.py'),

                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionAlgorithm/SingleDirectionDetection.py'),

                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionStrategy/AbstractDetectionStrategy.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionStrategy/GNSDetectionStrategy.py'),

                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionStrategy/GRSDetectionStrategy.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionStrategy/JSDetectionStrategy.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionStrategy/NSDetectionStrategy.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionStrategy/RoadDetectionContext.py'),

                    path.join(PROJECT_MAIN_DIR, 'Core/DetectionStrategy/SDDetectionStrategy.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/Exception.py'),

                    path.join(PROJECT_MAIN_DIR, 'Core/MorphologicalFiltering.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/PeripheralCondition.py'),
                    path.join(PROJECT_MAIN_DIR, 'Core/RoadDetection.py'),
                    path.join(PROJECT_MAIN_DIR, 'DetectObjects/CircleSeed.py'),
                    path.join(PROJECT_MAIN_DIR, 'DetectObjects/Pixels.py'),
                    path.join(PROJECT_MAIN_DIR, 'DetectObjects/Utils.py'),
                    path.join(PROJECT_MAIN_DIR, 'Test/CircleSeedDetail.py'),
                    path.join(PROJECT_MAIN_DIR, 'Test/CircleSeedItem.py'),
                    path.join(PROJECT_MAIN_DIR, 'Test/OpenCVAnalysis.py'),
                    path.join(PROJECT_MAIN_DIR, 'Test/ShowResultLabel.py'),
                    path.join(PROJECT_MAIN_DIR, 'Test/ShowResultView.py'),
                    path.join(PROJECT_MAIN_DIR, 'Test/test.py'),
                    path.join(PROJECT_MAIN_DIR, 'Test/TestDirection.py')]

a = Analysis(project_py_files,
             pathex=[PROJECT_MAIN_DIR],
             binaries=[],
             datas=[],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=["torch", "PIL", "openslide"],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='RoadExtraction',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=False,
          console=False,
         )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               upx_exclude=[],
               name='RoadExtraction')
