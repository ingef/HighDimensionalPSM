from setuptools import setup

setup(name='hdps',
      version='0.1',
      description='Functions to do high dimensional propensity score matching',
      author='Vivek Ramalingam Kailasam ',
      author_email='Vivek.Kailasam@ingef.de',
      packages=['hdps'],
      zip_safe=False, install_requires=['pandas', 'numpy', 'epydemiology'])
