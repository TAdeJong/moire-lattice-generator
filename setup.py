import setuptools
import versioneer

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open('requirements_test.txt') as f:
    requirements_test = f.read().splitlines()

setuptools.setup(
    name="latticegen",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="T.A. de Jong",
    author_email="tobiasadejong@gmail.com",
    description="A small package to create images of atomic lattices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TAdejong/moire-lattice-generator",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.7',
    install_requires=requirements,
    test_requires=requirements_test,
    test_suite="pytest",
)
