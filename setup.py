from setuptools import setup, find_packages

setup(
    name="aividence",
    version="0.1.0",
    author="Zaoqu Liu",
    author_email="liuzaoqu@163.com",
    description="Scientific claim validation for LLM-generated content",
    url="https://github.com/Zaoqu-Liu/aividence",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "biopython",
        "scikit-learn",
        "faiss-cpu",
        "sentence-transformers",
        "tqdm",
        "openai",
    ],
)