from setuptools import setup

requirements = [
    'trimesh',
    'autolab_core'
]

setup(name='foxarm',
      version='0.1.0',
      description='Fox-Arm project code',
      author='Blue Fox',
      package_dir = {'': 'foxarm'},
      packages=['foxarm'],
      install_requires=requirements
     )
