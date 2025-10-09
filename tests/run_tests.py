import sys
import os
import shutil
import pathlib
from pathlib import Path
import argparse
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
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent

    parser = argparse.ArgumentParser(description='Optix Utility Tests')
    parser.add_argument('--cmake-path', required=True)
    parser.add_argument('--build-dir', default=str(root_dir / 'build'))
    parser.add_argument('--cppstd', default='c++20')
    parser.add_argument('--cppstd-cuda', default='c++20')
    parser.add_argument('--cuda')
    args = parser.parse_args()

    cmake_path = Path(args.cmake_path)
    build_dir = Path(args.build_dir)



    # ----------------------------------------------------------------
    # CMake meta build

    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True)
    cmd = [
        str(cmake_path),
        '-S', str(root_dir),
        '-B', str(build_dir),
        '-G','Visual Studio 17 2022', '-A', 'x64',
        '-D', 'CPP_VER=' + args.cppstd,
        '-D', 'CPP_VER_CUDA=' + args.cppstd_cuda,
        '-T', 'cuda=' + args.cuda]
    run_command(cmd)

    # END: CMake meta build
    # ----------------------------------------------------------------



    # ----------------------------------------------------------------
    # Build

    sln = build_dir / R'OptiX_Utility_cmake.sln'
    msbuild = R'C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Current\Bin\MSBuild.exe'
    configs = ['Debug', 'Release']

    for config in configs:
        # # Clean
        # cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64', '/t:Clean']
        # cmd += [str(sln)]
        # run_command(cmd)

        # Build
        cmd = [msbuild, '/m', '/p:Configuration=' + config, '/p:Platform=x64']
        cmd += [str(sln)]
        run_command(cmd)

    # END: Build
    # ----------------------------------------------------------------



    # ----------------------------------------------------------------
    # Unit tests

    exe = build_dir / 'bin' / 'Release' / 'optixu_tests.exe'

    print('Run unit tests')
    cmd = [str(exe)]
    run_command(cmd)

    # END: Unit tests
    # ----------------------------------------------------------------



    # ----------------------------------------------------------------
    # Sample image tests

    refImgDir = script_dir / R'ref_images'

    with open(script_dir / R'tests.json') as f:
        tests = json.load(f)

    # Run tests
    results = {}
    for config in configs:
        resultsPerConfig = {}
        outDir = (build_dir / 'bin' / config).resolve()
        for test in tests:
            testName = test['name']
            testDir = Path(test['sample'])
            exeName = Path(test['sample'] + '.exe')

            print('Run ' + testName + ':')

            oldDir = chdir(build_dir / 'samples' / testDir)
            exe = outDir / exeName
            cmd = [str(exe)]
            if 'options' in test:
                cmd.extend(test['options'].split())
            run_command(cmd)

            # RGBAでdiffをとると差が無いことになってしまう。
            testImgPath = Path(test['image'])
            if not (refImgDir / testDir).exists():
                (refImgDir / testDir).mkdir(parents=True)
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
        