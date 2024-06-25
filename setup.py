from setuptools import setup, find_packages

setup(name='korspacing',
      version=0.1,      
      python_requires='>=3.6',
      url='https://github.com/airobotlab/korspacing',
      license='GPL-3',
      author='airobotlab',
      author_email='gwy876@gmail.com',
      description='python package for korean spacing model by torch',
      packages=find_packages(),      
#       packages=['korspacing', ],
      long_description=open('README.md', encoding='utf-8').read(),
      zip_safe=False,
      include_package_data=True,
      install_requires=requirements())