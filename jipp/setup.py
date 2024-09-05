from setuptools import setup, find_packages

setup(
    name="jipp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai",
        "pydantic",
        # ... other dependencies ...
    ],
)
