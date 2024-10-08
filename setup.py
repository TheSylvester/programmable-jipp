from setuptools import setup, find_packages

setup(
    name="jipp",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "PyYAML",
        "fastapi",
        "pydantic",
        # Add other dependencies as needed
    ],
    entry_points={
        "console_scripts": [
            "jipp= jipp.flow.main:main",  # Assuming you have a main function
        ],
    },
    python_requires=">=3.8",
)
