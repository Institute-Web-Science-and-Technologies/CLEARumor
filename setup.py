"""A setuptools based setup module.
See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""
from setuptools import setup, find_packages
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='clearumor',
    version='1.0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    url='https://github.com/Institute-Web-Science-and-Technologies/CLEARumor.git',
    license='Apache License 2.0',
    description='CLEARumor implementation',
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha'],
    dependency_links=['https://github.com/erikavaris/tokenizer.git'],
    include_package_data=True
)
