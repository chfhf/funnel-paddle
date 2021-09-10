import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="hf_paddle", # Replace with your own username
    version="0.0.4",
    author="Zhizhuo Zhang",
    author_email="zzz2010@gmail.com",
    description="paddle implementation for huggingface transformers repo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitee.com/littledesk/hf_transformers_paddle",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
