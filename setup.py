import setuptools
from io import open

requirements = [
    'numpy',
    'scipy',
    'pyfftw',
    'cctbx',
    'pathos',
    'mrcfile',
    'matplotlib',
    'setuptools'
]

setuptools.setup(
    name='tomoxtal',
    maintainer='Ariana Peck',
    version='0.1.0',
    maintainer_email='apeck@slac.stanford.edu',
    description='Cryo-ET of protein nanocrystals',
    long_description=open('README.md', encoding='utf8').read(),
    long_description_content_type="text/markdown",
    url='https://github.com/apeck12/tomoxtal.git',
    packages=setuptools.find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    zip_safe=False)
