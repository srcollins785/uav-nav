from setuptools import setup, find_packages

setup(
    name="uav_nav",
    version="0.1.0",
    description="Offline Vision-Language UAV Navigation Framework",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        "numpy>=2.0",
        "scipy>=1.13",
        "pillow>=11.0",
        "pydantic>=2.10",
        "pyyaml>=6.0",
        "filterpy>=1.4.5",
        "opencv-python-headless>=4.10",
        "ollama>=0.6",
        "matplotlib>=3.9",
    ],
    extras_require={
        "astronomy": ["skyfield>=1.54", "astropy>=6.0"],
        "dev": ["pytest>=8.0"],
    },
)
