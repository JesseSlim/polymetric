import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polymetric",
    version="0.5.0",
    author="Jesse Slim",
    author_email="jesse.j.slim@gmail.com",
    description="Generate complex polygon structures",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'shapely'
    ],
)
