import os
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
    sln = os.path.abspath(R'samples\OptiX_Utility.sln')
    refImgDir = os.path.abspath(R'ref_images')

    with open(R'tests.json') as f:
        tests = json.load(f)

    for config in ['Debug', 'Release']:
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64', '/t:Clean']
        cmd += [sln]
        print(' '.join(cmd))
        ret = subprocess.check_call(cmd)

        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64']
        cmd += [sln]
        print(' '.join(cmd))
        ret = subprocess.check_call(cmd)

        results = {}
        outDir = os.path.abspath(os.path.join(R'samples\x64', config))
        for test in tests:
            testName = test['sample']
            testDir = test['sample']
            exeName = test['sample'] + '.exe'

            oldDir = chdir(os.path.join('samples', testDir))
            exe = os.path.join(outDir, exeName)
            cmd = [exe]
            if 'options' in test:
                cmd.append(test['options'])
            ret = subprocess.check_call(cmd)

            # RGBAでdiffをとると差が無いことになってしまう。
            img = Image.open(test['image']).convert('RGB')
            refImg = Image.open(os.path.join(refImgDir, testDir, 'reference.png')).convert('RGB')
            diffImg = ImageChops.difference(img, refImg)
            diffBBox = diffImg.getbbox()
            if diffBBox is None:
                numDiffPixels = 0
            else:
                numDiffPixels = sum(x != (0, 0, 0) for x in diffImg.crop(diffBBox).getdata())

            results[testName] = {
                "success": numDiffPixels == 0,
                "numDiffPixels": numDiffPixels
            }

            for output in test['outputs']:
                os.remove(output)

            chdir(oldDir)
        
        print('Test Results for ' + config + ':')
        for test, result in results.items():
            print(test, result)
        print()

    return 0

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(e)
        