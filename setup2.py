#!/usr/bin/env python

import zalaiConvert
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from Cython.Build import cythonize
from zalai.tool.setup_utils import createContextManager


def getSnapInfo():
    import time
    st = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
    import subprocess
    cmd = "git rev-parse HEAD"
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    lines = p.stdout.readlines()
    commitId = lines[0].decode('gbk').strip()
    return f"{commitId} on {st}"


epy_list = [r'zalaiConvert/__main__.py', r'zalaiConvert/utils/constant.py']
pyx_list = [r'zalaiConvert/utils/constant.py']
with createContextManager(
    "zalaiConvert",
    [r"zalaiConvert/*.py", r"zalaiConvert/rknn_convert_utils/*/*.py", 
    r"zalaiConvert/zlg_convert_utils/*/*.py", r"zalaiConvert/utils/*.py"],
    exclude_list=epy_list) as value:


    fpk = find_packages(where=".", exclude=["zalaiConvert.test", "zalaiConvert.demo",
                        "zalaiConvert.bench"])
    print(fpk)
    # exit(1)
    setup(
        name='zalaiConvert',
        version=zalaiConvert.__version__,
        description=zalaiConvert.__description__,
        author=zalaiConvert.__author__,   
        keywords=zalaiConvert.__keywords__,
        packages=fpk,
        long_description=getSnapInfo(),
        include_package_data=True,
        zip_safe=False,
        python_requires='>=3.6',
        setup_requires=[],
        # install_requires=load_requirements(PATH_ROOT),
        classifiers=[
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
        entry_points={
            'console_scripts': [
                # 'zalai=zalai.main:main',
                # 'sensorFit=zalai.clt.cli_fit:main',
                # 'sensorClassify=zalai.clt.cli_classify:main',
                # 'sensorLstm=zalai.clt.cli_lstm:main',
                'zalaiConvert = zalaiConvert.__main__:main'
            ]
        },
        ext_modules=cythonize(
            pyx_list,
            quiet=False,
            language_level=3,
        ),
        package_data={
            'zalaiConvert': ['*.pyd'],
            'zalaiConvert.bin': ['*.exe', '*.dll'],            
            'zalaiConvert.zlg_convert_utils.darknet': ['*.pyd'],
            'zalaiConvert.rknn_convert_utils.darknet': ['*.pyd'],
            'zalaiConvert.utils': ['*.pyd']
        })
