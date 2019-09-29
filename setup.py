from setuptools import setup, find_packages

setup(
    name='CLEARumor',
    version='1.0.0',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    url='https://github.com/Institute-Web-Science-and-Technologies/CLEARumor.git',
    license='',
    author='',
    author_email='',
    description='CLEARumor implementation',
    classifiers = [
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha'],
    dependency_links=['https://github.com/erikavaris/tokenizer.git']
)
