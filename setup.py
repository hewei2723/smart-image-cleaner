from setuptools import setup, find_packages

setup(
    name="photoclean",
    version="0.1.0",
    author="hewei2723",
    author_email="hewei2723@qq.com",
    description="A smart tool to clean duplicate and blurry images",
    long_description=open("README.md", encoding="utf-8").read() if open("README.md", encoding="utf-8") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/hewei2723/smart-image-cleaner",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask>=2.0.0",
        "Pillow>=9.0.0",
        "imagehash>=4.0.0",
        "send2trash>=1.8.0",
        "werkzeug>=2.0.0",
        "opencv-python>=4.5.0",
        "numpy>=1.20.0"
    ],
    entry_points={
        'console_scripts': [
            'photoclean=photoclean.main:main',
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
