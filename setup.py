from setuptools import setup, find_packages
setup(
    name = "rl_codebase",
    version = "0.0.1",
    description = ("Simple codebase for RL projects"),
    packages=find_packages(),
    python_requires=">=3.7",
    author="giangbang",
    author_email="banggiangle2015@gmail.com",
    install_requires=[
        "torch",
        "gym",
        "numpy",
        "matplotlib"
    ],
)