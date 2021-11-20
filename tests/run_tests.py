import os
import shutil
import sys
import subprocess
import json
from PIL import Image, ImageChops

def chdir(dst):
    oldDir = os.getcwd()
    os.chdir(dst)
    return oldDir

def run():
    msbuild = R'C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\MSBuild\Current\Bin\MSBuild.exe'
    sln = os.path.abspath(R'..\samples\OptiX_Utility.sln')
    refImgDir = os.path.abspath(R'ref_images')

    with open(R'tests.json') as f:
        tests = json.load(f)

    configs = ['Debug', 'Release']

    # Build
    for config in configs:
        # Clean
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64', '/t:Clean']
        cmd += [sln]
        print(' '.join(cmd))
        ret = subprocess.run(cmd, check=True)

        # Build
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64']
        cmd += [sln]
        print(' '.join(cmd))
        ret = subprocess.run(cmd, check=True)

    # Run tests
    results = {}
    for config in configs:
        resultsPerConfig = {}
        outDir = os.path.abspath(os.path.join(R'..\samples\x64', config))
        for test in tests:
            testName = test['sample']
            testDir = test['sample']
            exeName = test['sample'] + '.exe'

            print('Run ' + testName + ':')

            oldDir = chdir(os.path.join(R'..\samples', testDir))
            exe = os.path.join(outDir, exeName)
            cmd = [exe]
            if 'options' in test:
                cmd.append(test['options'])
            ret = subprocess.run(cmd, check=True)

            # RGBAでdiffをとると差が無いことになってしまう。
            testImgPath = test['image']
            img = Image.open(testImgPath).convert('RGB')
            refImg = Image.open(os.path.join(refImgDir, testDir, 'reference.png')).convert('RGB')
            diffImg = ImageChops.difference(img, refImg)
            diffBBox = diffImg.getbbox()
            if diffBBox is None:
                numDiffPixels = 0
            else:
                numDiffPixels = sum(x != (0, 0, 0) for x in diffImg.crop(diffBBox).getdata())

            resultsPerConfig[testName] = {
                "success": numDiffPixels == 0,
                "numDiffPixels": numDiffPixels
            }

            for output in test['outputs']:
                if config == 'Release' and output == testImgPath and numDiffPixels > 0:
                    shutil.move(testImgPath, os.path.join(refImgDir, testDir))
                else:
                    os.remove(output)

            chdir(oldDir)
        
        results[config] = resultsPerConfig
    
    # Show results
    for config in configs:
        print('Test Results for ' + config + ':')
        resultsPerConfig = results[config]
        numSuccesses = 0
        for test, result in resultsPerConfig.items():
            print(test, result)
            if result['success']:
                numSuccesses += 1
        print('Successes: {}/{}, All Success: {}'.format(
            numSuccesses, len(resultsPerConfig), numSuccesses == len(resultsPerConfig)))
        print()

    return 0

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(e)
        