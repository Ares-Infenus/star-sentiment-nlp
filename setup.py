from setuptools import setup, find_packages

setup(
    name="star-sentiment-nlp",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "spacy>=3.7.0",
        "nltk>=3.8.1",
        "contractions>=0.1.73",
        "scikit-learn>=1.4.0",
        "gensim>=4.3.0",
        "transformers>=4.40.0",
        "torch>=2.2.0",
        "xgboost>=2.0.0",
        "gradio>=4.25.0",
        "datasets>=2.18.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "joblib>=1.3.0",
        "tqdm>=4.66.0",
    ],
)
