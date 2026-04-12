from setuptools import setup, find_packages

setup(
    name="physicore",
    version="1.2.0",
    author="Prathamesh Shirbhate",
    author_email="prathamesh@physicore.ai",
    description="Hybrid Uncertainty-Aware Sim-to-Real Synchronization Engine",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/prathamesh-shirbhate/physicore",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "scipy>=1.10.0",
    ],
    extras_require={
        "bridge": [
            "pymavlink>=2.4.0",
            "websockets==12.0",
            "aiohttp>=3.9.0",
            "pyserial>=3.5",
        ],
        "api": [
            "fastapi>=0.110.0",
            "uvicorn>=0.27.0",
        ],
        "sdk": [
            "requests>=2.31.0",
        ],
        "ros2": ["rclpy"],
        "dev":  ["pytest>=7.0", "matplotlib>=3.7.0"],
        "all":  [
            "pymavlink>=2.4.0",
            "websockets==12.0",
            "aiohttp>=3.9.0",
            "pyserial>=3.5",
            "fastapi>=0.110.0",
            "uvicorn>=0.27.0",
            "requests>=2.31.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "physicore-bridge=physicore.bridge.physicore_bridge:main",
            "physicore-api=physicore.api.server:app",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Physics",
    ],
)
