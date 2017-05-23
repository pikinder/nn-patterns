import os
import setuptools


requirements = [
    "numpy",
    "scipy",
    "lasagne",
]


def readme():
    base_dir = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(base_dir, 'README.md')) as f:
        return f.read()


def setup():
    setuptools.setup(
        name="nn-patterns",
        version="0.1",
        description=("Implementation of PatternNet and PatternLRP:"
                     " https://arxiv.org/abs/1705.05598"),
        long_description=readme(),
        classifiers=[
            "License :: OSI Approved :: ",
            "Development Status :: 3 - Alpha",
            "Environment :: Console",
            "Intended Audience :: Science/Research",
            "Operating System :: POSIX :: Linux",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3.4",
            "Topic :: Scientific/Engineering :: Artificial Intelligence",
        ],
        url="https://github.com/pikinder/nn-patterns",
        author="Pieter-Jan Kindermans, Maxmilian Alber",
        author_email="pieterjankindermans@gmail.com",
        license="MIT",
        packages=setuptools.find_packages(),
        install_requires=requirements,
        include_package_data=True,
        zip_safe=False,
    )
    pass


if __name__ == "__main__":
    setup()
