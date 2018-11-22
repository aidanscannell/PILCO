from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='pilco',
    version='0.1.0',
    description='Python Implementation of PILCO',
    long_description=readme,
    author='Aidan Scannell',
    author_email='aidan.scannell@brl.ac.uk',
    # url='https://github.com/kennethreitz/samplemod',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)