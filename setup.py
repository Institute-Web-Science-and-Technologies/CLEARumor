from setuptools import setup, find_packages

setup(
    name='clearumor',
    version='1.0.1',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    url='https://github.com/Institute-Web-Science-and-Technologies/CLEARumor.git',
    description='CLEARumor implementation',
    license = 'Apache License 2.0',
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha'],
    dependency_links=['https://github.com/erikavaris/tokenizer.git'],
    package_data = {'src':['*']}
)
