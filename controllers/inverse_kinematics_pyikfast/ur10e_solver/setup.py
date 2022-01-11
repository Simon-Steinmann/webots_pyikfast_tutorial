import setuptools
from distutils.core import setup, Extension


def main():
    setup(
        name='pyikfast_ur10e',
        version='0.0.1',
        description='ikfast wrapper',
        author='Cyberbotics',
        author_email='support@cyberbotics.com',
        ext_modules=[Extension('pyikfast_ur10e', ['ur_kinematics/ur_kin.cpp', 'pyikfast.cpp'], language="c++")],
        setup_requires=['wheel']
    )


if __name__ == '__main__':
    main()
