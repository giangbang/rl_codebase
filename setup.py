from setuptools import setup, find_packages
setup(
    name = "rl_codebase",
    version = "0.0.1",
    description = ("Simple codebase for RL projects"),
    packages=find_packages(),
    python_requires=">=3.7",
    author="giangbang",
    author_email="banggiangle2015@gmail.com",
    url="https://github.com/giangbang/rl_codebase",
    install_requires=[
        "torch",
        "torchvision",
        "gym==0.25.2",
        "numpy",
        "matplotlib",
        "opencv-python",
    ],
    entry_points = {
        'console_scripts': ['train=rl_codebase.train:main'],
    },
)