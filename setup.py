from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()
try:
    from pypandoc import convert
    read_md = lambda f: convert(f, 'rst')
except ImportError:
    print("warning: pypandoc module not found, could not convert Markdown to RST")
    read_md = lambda f: open(f, 'r').read()

setup(name='scanTree',
      version='0.1',
      description='scanpy compatible python suite for fast tree inference and advanced pseudotime downstream analysis',
      long_description=read_md('README.md'),
      url='https://github.com/LouisFaure/scanTree',
      author='Louis Faure',
      author_email='',
      packages=find_packages(),
      package_dir={'scanTree': 'scanTree'},
      install_requires=requirements,
      zip_safe=False)
