from distutils.core import setup
setup(
  name = 'pyfluv',
  packages = ['pyfluv'],
  version = '0.11',
  license='GNU General Public License v3 (GPLv3)',
  description = 'A package for analyzing fluvial planforms, profiles and cross sections, with a focus on restoration.',
  author = 'Sky Jones',
  author_email = 'rsajones94@gmail.com',
  url = 'https://github.com/rsjones94',
  download_url = 'https://github.com/rsjones94/pyfluv/archive/v_011.tar.gz',
  keywords = [
              'river', 'fluvial', 'geomorphology', 'stream',
              'analysis', 'profile', 'planform', 'geometry', 'rosgen', 'restoration'
              ],
  install_requires=[
          'numpy'
          'matplotlib'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Physics',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3'
  ],
)
