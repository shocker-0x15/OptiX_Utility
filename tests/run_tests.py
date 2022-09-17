import sys
import os
import shutil
import pathlib
from pathlib import Path
import subprocess
import json
from PIL import Image, ImageChops

def chdir(dst):
    oldDir = os.getcwd()
    os.chdir(dst)
    return oldDir

def run_command(cmd):
    print(' '.join(cmd))
    ret = subprocess.run(cmd, check=True)

def run():
    scriptDir = Path(__file__).parent

    msbuild = R'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'

    # ----------------------------------------------------------------
    # Unit tests

    sln = scriptDir / R'optixu_unit_tests.sln'
    config = 'Release'
    exe = scriptDir / 'x64' / config / 'optixu_tests.exe'

    # Clean
    cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64', '/t:Clean']
    cmd += [str(sln)]
    run_command(cmd)

    # Build
    cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64']
    cmd += [str(sln)]
    run_command(cmd)

    print('Run unit tests')
    cmd = [str(exe)]
    run_command(cmd)

    # END: Unit tests
    # ----------------------------------------------------------------



    # ----------------------------------------------------------------
    # Sample image tests

    sln = (scriptDir / R'..\samples\optixu_samples.sln').resolve()
    refImgDir = scriptDir / R'ref_images'

    with open(scriptDir / R'tests.json') as f:
        tests = json.load(f)

    configs = ['Debug', 'Release']

    # Build
    for config in configs:
        # Clean
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64', '/t:Clean']
        cmd += [str(sln)]
        run_command(cmd)

        # Build
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64']
        cmd += [str(sln)]
        run_command(cmd)

    # Run tests
    results = {}
    for config in configs:
        resultsPerConfig = {}
        outDir = (scriptDir / R'..\samples\x64' / config).resolve()
        for test in tests:
            testName = test['name']
            testDir = Path(test['sample'])
            exeName = Path(test['sample'] + '.exe')

            print('Run ' + testName + ':')

            oldDir = chdir(scriptDir / R'..\samples' / testDir)
            exe = outDir / exeName
            cmd = [str(exe)]
            if 'options' in test:
                cmd.extend(test['options'].split())
            run_command(cmd)

            # RGBAでdiffをとると差が無いことになってしまう。
            testImgPath = Path(test['image'])
            refImgPath = refImgDir / testDir / test['reference']

            if testImgPath.exists() and refImgPath.exists():
                img = Image.open(testImgPath).convert('RGB')
                refImg = Image.open(refImgPath).convert('RGB')
                diffImg = ImageChops.difference(img, refImg)
                diffBBox = diffImg.getbbox()
                if diffBBox is None:
                    numDiffPixels = 0
                else:
                    numDiffPixels = sum(x != (0, 0, 0) for x in diffImg.crop(diffBBox).getdata())
            else:
                numDiffPixels = -1

            resultsPerConfig[testName] = {
                "success": numDiffPixels == 0,
                "numDiffPixels": numDiffPixels
            }

            # 出力されたファイルを削除する。
            # リファレンスとの相違があった場合やリファレンスが存在しない新テストに関しては出力画像を移動する。
            for output in test['outputs']:
                output = Path(output)
                if not output.exists():
                    continue
                if config == 'Release' and output == testImgPath and numDiffPixels != 0:
                    newRefImgPath = refImgPath
                    if newRefImgPath.exists():
                        newRefImgPath = newRefImgPath.parent / (newRefImgPath.stem + '_new' + newRefImgPath.suffix)
                    testImgPath.rename(newRefImgPath)
                else:
                    output.unlink()

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

    # END: Sample image tests
    # ----------------------------------------------------------------

    return 0

if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(e)
        