#!/usr/bin/env python

import os
from io import open
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import zalaiConvert
PATH_ROOT = os.path.dirname(__file__)


fpk = find_packages(where=".", exclude=["zalaiConvert.test", "zalaiConvert.demo",
                    "zalaiConvert.bench"])
# fpk += ['zalaiConvert.cfg']
print(fpk)

setup(
    name='zalaiConvert',
    version=zalaiConvert.__version__,
    description=zalaiConvert.__description__,
    author=zalaiConvert.__author__,
    # author_email=pytorch_lightning.__author_email__,
    # url=pytorch_lightning.__homepage__,
    # download_url='https://github.com/williamFalcon/pytorch-lightning',
    # license=pytorch_lightning.__license__,
    packages=fpk,

    # long_description=open('README.md', encoding='utf-8').read(),
    # long_description_content_type='text/markdown',
    include_package_data=True,  # 将数据文件也打包
    package_data={
         'zalaiConvert.cfg': ['*.json'],
    },

    zip_safe=False,

    keywords=zalaiConvert.__keywords__,
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
            'zalaiConvert=zalaiConvert.main:main'
        ]
    }

)
