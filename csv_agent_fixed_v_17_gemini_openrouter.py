import os
import zipfile
import pandas as pd
import streamlit as st
from pathlib import Path
# from langchain.llms import Ollama
# from langchain_ollama import OllamaLLM  # Removido: não usamos mais Ollama
import requests
import tempfile
import shutil
import traceback
import locale
import re
import ast
import textwrap

# --- Cliente LLM via OpenRouter ---
class OpenRouterLLM:
    def __init__(self, model: str, api_key: str = None, temperature: float = 0.0):
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        self.model = model
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.temperature = temperature
        if not self.api_key:
            raise ValueError("Chave OPENROUTER_API_KEY não encontrada no ambiente.")

    def invoke(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        resp = requests.post(self.endpoint, json=payload, headers=headers)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


def main():
    st.set_page_config(
        page_title="Agente de Análise CSV",
        page_icon="📊",
        layout="wide"
    )

class CSVAnalysisAgent:
    def __init__(self):
        """Inicializa o agente com LLM via OpenRouter"""
        try:
            self.llm = OpenRouterLLM(
                model="openrouter/deepseek-chat-v3-0324:free",
                api_key=os.getenv("OPENROUTER_API_KEY"),
                temperature=0.0
            )
        except Exception as e:
            st.error(f"Erro ao inicializar LLM: {e}")
            st.info("Configure a variável de ambiente OPENROUTER_API_KEY com seu token.")
            return
        self.dataframes = {}
        self.current_df = None

    def extract_zip_files(self, zip_path, extract_to):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            st.error(f"Erro ao descompactar: {e}")
            return False

    def load_csv_files(self, directory):
        csv_files = {}
        for file_path in Path(directory).rglob("*.csv"):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                df.columns = df.columns.str.strip()
                csv_files[file_path.name] = df
                st.success(f"Carregado: {file_path.name} ({len(df)} linhas)")
            except UnicodeDecodeError:
                try:
                    df = pd.read_csv(file_path, encoding='latin-1')
                    df.columns = df.columns.str.strip()
                    csv_files[file_path.name] = df
                    st.success(f"Carregado: {file_path.name} ({len(df)} linhas)")
                except Exception as e:
                    st.error(f"Erro ao carregar {file_path.name}: {e}")
            except Exception as e:
                st.error(f"Erro ao carregar {file_path.name}: {e}")
        return csv_files

    def select_dataframe(self, df_name):
        if df_name in self.dataframes:
            self.current_df = self.dataframes[df_name]
            return True
        return False

    def _query_llm(self, prompt: str) -> str:
        return self.llm.invoke(prompt)

    def step0_select_file(self, question):
        # lógica inalterada, exceto chamada para invoke
        files_summary = """
ARQUIVOS DISPONÍVEIS:
1. '202401_NFs_Cabecalho.csv' ...
2. '202401_NFs_Itens.csv' ...
"""
        prompt = f"""
...PERGUNTA: "{question}"\n"""
        try:
            response = self._query_llm(prompt)
            # processa resposta...
        except Exception as e:
            return {'sucesso': False, 'erro': str(e)}

    def step1_interpret_question(self, question, selected_files):
        # substitua todas as chamadas self.llm.invoke(...) por self._query_llm(...)
        if len(selected_files) == 1:
            self.select_dataframe(selected_files[0])
            prompt = f"""
Seu prompt mestre para 1 arquivo...
PERGUNTA: "{question}"""  # mantenha template
        else:
            prompt = f"""
Seu prompt mestre para múltiplos arquivos...
PERGUNTA: "{question}"""
        response = self._query_llm(prompt)
        # limpa response e retorna código

    def step2_execute_code(self, generated_code, selected_files):
        # idêntico ao original
        namespace = {'pd': pd, 'resultado': None}
        # adiciona dataframes...
        try:
            exec(generated_code, namespace)
            return {'sucesso': True, 'resultado': namespace.get('resultado'), 'codigo_executado': generated_code}
        except Exception as e:
            return {'sucesso': False, 'erro': str(e), 'traceback': traceback.format_exc()}

    def step3_generate_response(self, user_question, execution_result):
        # troca chamadas de self.llm.invoke para self._query_llm
        if execution_result['sucesso']:
            formatted_result = execution_result['resultado']
            prompt = f"""
Sua tarefa: gerar frase em pt de dados: {formatted_result}"""
        else:
            prompt = f"Erro: {execution_result['erro']}"
        return self._query_llm(prompt)

    def query_data(self, question):
        # fluxo principal, use self._query_llm em todo lugar
        ...

    # demais métodos inalterados

if __name__ == "__main__":
    st.sidebar.markdown("PRÉ-REQUISITOS: pip install requests ...")
    main()
