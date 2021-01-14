import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ewolm",
    version="0.0.1",
    author="Tolga Dincer",
    author_email="tolgadincer@gmail.com",
    description="Ensemble Weight Optimization with Lagrange Multiplier Method",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tdincer/ewolm",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)