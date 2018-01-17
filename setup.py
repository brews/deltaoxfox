from setuptools import setup, find_packages

setup(
    name='deltaoxfox',
    version='0.0.1',
    description='δ18Oc calibration using Bayesian regression',
    license='GPLv3',

    author='S. Brewster Malevich',
    author_email='malevich@email.arizona.edu',
    url='https://github.com/brews/deltaoxfox',
    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
    ],
    keywords='marine paleoclimate d18O',

    packages=find_packages(exclude=['docs']),

    install_requires=['numpy', 'pandas', 'scipy', 'matplotlib', 'attrs'],
    tests_require=['pytest'],
    package_data={'deltaoxfox': ['modelparams/*.h5']}
)