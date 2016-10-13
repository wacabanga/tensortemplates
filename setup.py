from setuptools import setup

# import `__version__` from code base
exec(open('reverseflow/version.py').read())

setup(
    name='tensortemplates',
    version=__version__,
    description='Parameterized tensorflow models',
    author='Zenna Tavares',
    author_email="zenna@mit.edu",
    packages=['tensortemplates'],
    install_requires=['tensorflow>=0.11.0rc0',
                      'numpy>=1.7'],
    url='http://www.zenna.org',
    license='Apache License 2.0',
    classifiers=['License :: OSI Approved :: Apache Software License',
                 'Programming Language :: Python :: 2.7',
                 'Programming Language :: Python :: 3.4'],
)
