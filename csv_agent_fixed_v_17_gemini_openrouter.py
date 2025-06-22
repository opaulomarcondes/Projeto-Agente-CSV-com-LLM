import os
import zipfile
import pandas as pd
import streamlit as st
from pathlib import Path
import requests
import tempfile
import shutil
import traceback
import locale
import re

# --- Cliente LLM via OpenRouter ---
class OpenRouterLLM:
    def __init__(self, model: str="deepseek/deepseek-chat-v3-0324:free", api_key: str = None, temperature: float = 0.0):
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
        response = requests.post(self.endpoint, json=payload, headers=headers)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

# --- Agente de análise de CSV ---
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
            self.llm = None
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
        if not self.llm:
            raise RuntimeError("LLM não inicializado.")
        return self.llm.invoke(prompt)

    def step0_select_file(self, question):
        files_summary = (
            "ARQUIVOS DISPONÍVEIS:\n"
            "1. '202401_NFs_Cabecalho.csv' (fornecedor, nota fiscal, data, montante recebido)\n"
            "2. '202401_NFs_Itens.csv' (item, produto, quantidade, valor unitário, valor total do item)\n"
        )
        prompt = f"""
Você é um roteador de arquivos. Selecione quais arquivos usar:\n
{files_summary}
PERGUNTA: '{question}'
Responda apenas com os nomes dos arquivos separados por vírgula.
"""
        try:
            response = self._query_llm(prompt)
            names = [n.strip().strip('"\'') for n in re.split('[,\n]', response) if n.strip()]
            valid = [n for n in names if n in self.dataframes]
            if not valid:
                return {'sucesso': False, 'erro': 'Nenhum arquivo válido selecionado.'}
            return {'sucesso': True, 'arquivos_escolhidos': valid}
        except Exception as e:
            return {'sucesso': False, 'erro': str(e)}

    def step1_interpret_question(self, question, selected_files):
        if len(selected_files) == 1:
            df_var = 'df'
            self.select_dataframe(selected_files[0])
            columns = list(self.current_df.columns)
            prompt = f"""
Você é especialista em Python/Pandas. O DataFrame '{df_var}' com colunas {columns} existe.
PERGUNTA: '{question}'
Gere apenas o código Python que responda à pergunta e atribua à variável 'resultado'.
"""
        else:
            prompt = f"""
Dois DataFrames carregados: df_cabecalho, df_itens.
Chave de junção: 'CHAVE DE ACESSO'.
PERGUNTA: '{question}'
Gere código Python usando pd.merge e atribua resultado à variável 'resultado'.
"""
        try:
            code = self._query_llm(prompt)
            return code
        except Exception as e:
            return f"# Erro ao gerar código: {e}"

    def step2_execute_code(self, generated_code, selected_files):
        namespace = {'pd': pd, 'resultado': None}
        if len(selected_files) == 1:
            namespace['df'] = self.dataframes[selected_files[0]]
        else:
            namespace['df_cabecalho'] = self.dataframes['202401_NFs_Cabecalho.csv']
            namespace['df_itens'] = self.dataframes['202401_NFs_Itens.csv']
        try:
            exec(generated_code, namespace)
            return {'sucesso': True, 'resultado': namespace.get('resultado'), 'codigo': generated_code}
        except Exception as e:
            return {'sucesso': False, 'erro': str(e), 'traceback': traceback.format_exc()}

    def step3_generate_response(self, user_question, execution_result):
        if execution_result['sucesso']:
            formatted = execution_result['resultado']
            prompt = f"""
Formate em português:
Pergunta: {user_question}
Dados: {formatted}
Responda de forma clara e direta.
"""
        else:
            prompt = f"""
Erro ao processar pergunta: {user_question}
Detalhe: {execution_result.get('erro')}
Responda em português sugerindo reformular.
"""
        try:
            return self._query_llm(prompt)
        except Exception as e:
            return f"Erro na geração da resposta: {e}"

    def query_data(self, question):
        if not self.dataframes:
            return "Carregue CSVs primeiro."
        sel = self.step0_select_file(question)
        if not sel['sucesso']:
            return sel['erro']
        files = sel['arquivos_escolhidos']
        code = self.step1_interpret_question(question, files)
        exec_res = self.step2_execute_code(code, files)
        return self.step3_generate_response(question, exec_res)

# --- Interface Streamlit ---
def run_app():
    st.set_page_config(page_title="Agente de Análise CSV", page_icon="📊", layout="wide")
    st.title("🤖 Agente Inteligente para Análise de CSV")

    if 'agent' not in st.session_state:
        st.session_state.agent = CSVAnalysisAgent()
    agent = st.session_state.agent

    with st.sidebar:
        st.header("📁 Carregar Dados")
        uploaded = st.file_uploader("Upload ZIP ou CSV", type=['zip','csv'])
        if uploaded:
            tmp = tempfile.mkdtemp()
            path = os.path.join(tmp, uploaded.name)
            with open(path, 'wb') as f:
                f.write(uploaded.getbuffer())
            if uploaded.name.endswith('.zip'):
                ext = os.path.join(tmp, 'extracted')
                if agent.extract_zip_files(path, ext):
                    agent.dataframes = agent.load_csv_files(ext)
            else:
                agent.dataframes = agent.load_csv_files(tmp)
            st.success(f"Carregados {len(agent.dataframes)} CSVs")

    if agent.dataframes:
        st.header("💬 Faça sua pergunta")
        q = st.text_area("Sua pergunta:")
        if st.button("🔍 Analisar"):
            with st.spinner("Processando..."):
                resp = agent.query_data(q)
            st.success("Resposta final:")
            st.write(resp)

if __name__ == "__main__":
    run_app()
