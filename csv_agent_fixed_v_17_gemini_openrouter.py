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
    def main():   
       st.title("🤖 Agente Inteligente para Análise de CSV")
    st.markdown("### 🔄 Sistema em 3 Etapas: Pergunta → Código → Execução → Resposta")
    
        # Inicializa o agente
 if 'agent' not in st.session_state:
            st.session_state.agent = CSVAnalysisAgent()
    
        agent = st.session_state.agent
    
     # Sidebar para upload e configuração
with st.sidebar:
            st.header("📁 Carregar Dados")
        
            # Upload de arquivo
            uploaded_file = st.file_uploader(
                "Faça upload de arquivo ZIP ou CSV",
                type=['zip', 'csv'],
                help="Aceita arquivos ZIP contendo CSVs ou arquivos CSV individuais"
            )
        
            if uploaded_file:
                # Cria diretório temporário
                temp_dir = tempfile.mkdtemp()
            
                try:
                    if uploaded_file.name.endswith('.zip'):
                        # Salva arquivo ZIP
                        zip_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(zip_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                    
                        # Descompacta
                        extract_dir = os.path.join(temp_dir, 'extracted')
                        if agent.extract_zip_files(zip_path, extract_dir):
                            # Carrega CSVs
                            agent.dataframes = agent.load_csv_files(extract_dir)
                    else:
                        # Arquivo CSV individual
                        csv_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(csv_path, 'wb') as f:
                            f.write(uploaded_file.getbuffer())
                        agent.dataframes = agent.load_csv_files(temp_dir)
                
                    st.success(f"Carregados {len(agent.dataframes)} arquivo(s) CSV")
                
                except Exception as e:
                    st.error(f"Erro ao processar arquivo: {e}")
    
        # Interface principal
        if agent.dataframes:
            col1, col2 = st.columns([1, 2])
        
            with col1:
                st.header("📋 Arquivos Disponíveis")
            
                # Seleção de arquivo
                selected_file = st.selectbox(
                    "Escolha um arquivo para analisar:",
                    list(agent.dataframes.keys())
                )
            
                if selected_file:
                    agent.select_dataframe(selected_file)
                
                    # Mostra informações do arquivo
                    with st.expander("ℹ️ Informações do Arquivo"):
                        info = agent.get_dataframe_info(selected_file)
                        stats = agent.get_quick_stats(selected_file)
                    
                        st.write(f"**Dimensões:** {info['shape'][0]} linhas × {info['shape'][1]} colunas")
                        st.write(f"**Linhas totais:** {stats['total_rows']}")
                        st.write(f"**Colunas numéricas:** {stats['numeric_columns']}")
                        st.write(f"**Colunas de texto:** {stats['text_columns']}")
                        st.write(f"**Valores nulos:** {stats['null_values']}")
                        st.write(f"**Linhas duplicadas:** {stats['duplicated_rows']}")
                    
                        st.write("**Colunas:**")
                        for col in info['columns']:
                            st.write(f"- {col}")
                
                    # Análise das colunas para debug
                    with st.expander("🔍 Análise das Colunas"):
                        col_analysis = agent.get_column_analysis(selected_file)
                        if col_analysis:
                        for col, details in col_analysis.items():
                            st.write(f"**{col}:**")
                            st.write(f"  - Tipo: {details['tipo']}")
                            st.write(f"  - Valores únicos: {details['valores_unicos']}")
                            st.write(f"  - Nulos: {details['nulos']}")
                            st.write(f"  - Exemplo: {details['exemplo']}")
                            st.write("---")
                
                    # Preview dos dados
                    with st.expander("👀 Preview dos Dados (5 primeiras linhas)"):
                        st.dataframe(agent.current_df.head())
                        st.caption(f"Mostrando 5 de {len(agent.current_df)} linhas totais")
        
            with col2:
                st.header("💬 Faça sua Pergunta")
            
                # Estatísticas rápidas
                if selected_file:
                    stats = agent.get_quick_stats(selected_file)
                    st.info(f"📊 **Dataset atual:** {stats['total_rows']} linhas × {stats['total_columns']} colunas")
            
                # Alerta sobre correção
                st.info("💡 **Correção aplicada:** Sistema otimizado para usar valores reais do dataset!")
            
                # Explicação do processo
                st.markdown("""
                **🔄 Como funciona:**
                1. **Interpretação:** LLM entende sua pergunta e gera código Python
                2. **Execução:** Código é executado no dataset real
                3. **Resposta:** LLM transforma o resultado em resposta clara
                """)
            
                # Exemplos de perguntas
                st.write("**Exemplos de perguntas:**")
                examples = [
                    "Quantas linhas tem o dataset?",
                    "Qual produto tem o maior valor unitário?",
                    "Qual fornecedor teve maior montante recebido?",
                    "Qual item teve maior volume entregue?",
                    "Mostre o top 5 produtos por quantidade",
                    "Qual a soma total dos valores?",
                    "Quantos fornecedores únicos existem?"
                ]
            
                for example in examples:
                    if st.button(example, key=f"ex_{example}"):
                       st.session_state.question = example
            
                # Input da pergunta
                question = st.text_area(
                    "Sua pergunta:",
                    value=st.session_state.get('question', ''),
                    height=100,
                    placeholder="Digite sua pergunta sobre os dados..."
                )
                if st.button("🔍 Analisar", type="primary"):
                    if question:
                        with st.spinner("🤖 Agente pensando... (Etapas 0 a 3)"):
                            response = agent.query_data(question)
                            st.success("✅ Resposta Final do Agente:")
                            st.write(response)
                    else:
                        st.warning("Por favor, digite uma pergunta.")
    
        else:
            st.info("👆 Faça upload de um arquivo CSV ou ZIP contendo CSVs para começar")
        
            # Instruções
            with st.expander("📖 Como usar"):
                st.write("""
                **Sistema em 3 Etapas:**
            
                1. **📝 Pergunta:** Você faz uma pergunta em linguagem natural
                2. **🧠 Interpretação:** LLM analisa e gera código Python específico
                3. **⚡ Execução:** Código é executado no dataset real
                4. **💬 Resposta:** LLM transforma o resultado em resposta clara
            
                **Vantagens desta abordagem:**
                - ✅ Maior precisão nas respostas
                - ✅ Transparência total (você vê o código gerado)
                - ✅ Menos erros de parsing
                - ✅ Resultados baseados nos dados reais
            
                **Exemplos de perguntas suportadas:**
                - Quantas linhas tem o dataset?
                - Qual é o maior valor na coluna X?
                - Mostre estatísticas descritivas
                - Qual categoria tem mais itens?
                - Some os valores da coluna Y
                """)

    # Configuração para executar
    if __name__ == "__main__":
        # Instruções de instalação
        st.sidebar.markdown("""
        ### 🛠️ Pré-requisitos
    
        **Instalar dependências:**
        ```bash
        pip install streamlit langchain pandas
        ```
        """)
    main()
