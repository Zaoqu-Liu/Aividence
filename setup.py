from setuptools import setup, find_packages

setup(
    name="aividence",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "biopython",
        "sentence-transformers",
        "faiss-cpu",
        "openai",
        "tqdm",
    ],
    author="Zaoqu Liu",
    author_email="liuzaoqu@163.com",
    description="A tool for validating scientific claims against PubMed literature using LLMs",
    keywords="science, validation, llm, pubmed, literature",
    python_requires=">=3.7",
)