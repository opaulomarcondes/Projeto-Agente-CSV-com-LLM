from setuptools import setup, find_packages

setup(
    name="agente-csv-llm",
    version="0.1.0",
    description="Agente de análise de arquivos CSV com LLM local via Streamlit",
    packages=find_packages(),
    python_requires=">=3.11,<3.12",
    install_requires=[
        "streamlit",
        "pandas",
        "langchain",
        "langchain-ollama",
    ],
)
