from setuptools import setup, find_packages

setup(
    name="twinmind_ai_core",
    version="0.1.0",
    description="Core AI Management library for TwinMind ecosystem",
    packages=find_packages(),
    install_requires=[
        "requests",
        "google-genai",
        "python-dotenv",
        "numpy",
        "opencv-python-headless",
        "Pillow"
    ],
)
