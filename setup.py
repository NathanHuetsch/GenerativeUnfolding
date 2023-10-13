from setuptools import setup, find_packages

HTTPS_GITHUB_URL = "https://github.com/heidelberg-hepml/memennto"

#with open("README.md", "r") as fh:
#    long_description = fh.read()

requirements = ["numpy",
                "pandas",
                "scipy",
                "tables",
                "torch",
                "FrEIA",
                "pyyaml",
                "tqdm",
                "torchdiffeq",
                "matplotlib"]

setup(
    name="GenerativeUnfolding",
    version="1.0.0",
    author="Theo Heimel, Nathan Huetsch",
    author_email="huetsch@thphys.uni-heidelberg.de",
    description="Generative Unfolding with INNs and CFMs",
    #long_description=long_description,
    long_description_content_type="text/md",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    packages=find_packages(exclude=["tests"]),
    install_requires=requirements,
    entry_points={"console_scripts": ["src=src.__main__:main"]},
)
