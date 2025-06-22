import os
import zipfile
import pandas as pd
import streamlit as st
from pathlib import Path
# from langchain.llms import Ollama
# from langchain_ollama import OllamaLLM
import requests
import tempfile
import shutil
import traceback
import locale
import re
import ast
import textwrap


def main():
    st.set_page_config(
        page_title="Agente de Análise CSV",
        page_icon="📊",
        layout="wide"
    )
    

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
        """Descompacta arquivos zip"""
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            return True
        except Exception as e:
            st.error(f"Erro ao descompactar: {e}")
            return False
    
    def load_csv_files(self, directory):
        """Carrega todos os arquivos CSV de um diretório"""
        csv_files = {}
        
        for file_path in Path(directory).rglob("*.csv"):
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                # Limpa nomes das colunas
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
        """Seleciona um DataFrame específico para análise"""
        if df_name in self.dataframes:
            self.current_df = self.dataframes[df_name]
            return True
        return False


    def step0_select_file(self, question):
        """ETAPA 0: LLM seleciona o(s) arquivo(s) CSV relevante(s) para a pergunta."""
        
        files_summary = f"""
ARQUIVOS DISPONÍVEIS:
1. '202401_NFs_Cabecalho.csv' (contém informações de 'fornecedor', 'nota fiscal', 'data', 'montante recebido')
2. '202401_NFs_Itens.csv' (contém informações de 'item', 'produto', 'quantidade', 'valor unitário', 'valor total do item')
"""
            
        prompt = f"""
Você é um roteador de arquivos. Sua única tarefa é seguir um conjunto de regras para decidir quais arquivos usar para responder a uma pergunta.

{files_summary}

REGRAS DE DECISÃO (execute na ordem):
- REGRA 1 (PERGUNTA COMPLEXA): Se a pergunta precisa de informações dos DOIS arquivos ao mesmo tempo (ex: palavras de 'fornecedor' E 'produto' na mesma pergunta), sua resposta OBRIGATÓRIA é: `202401_NFs_Itens.csv,202401_NFs_Cabecalho.csv`

- REGRA 2 (ARQUIVO CABEÇALHO): Se a REGRA 1 não se aplica, e a pergunta é APENAS sobre 'fornecedor', 'nota fiscal' ou 'montante recebido', sua resposta OBRIGATÓRIA é: `202401_NFs_Cabecalho.csv`

- REGRA 3 (ARQUIVO ITENS): Se a REGRA 1 não se aplica, e a pergunta é APENAS sobre 'item', 'produto', 'quantidade' ou 'valor', sua resposta OBRIGATÓRIA é: `202401_NFs_Itens.csv`

INSTRUÇÕES FINAIS:
- Siga as regras acima em ordem. A REGRA 1 tem a maior prioridade.
- Não adicione explicações. Apenas o(s) nome(s) do(s) arquivo(s) conforme as regras.

PERGUNTA: "{question}"

RESPOSTA:
"""
        try:
            response = self.llm.invoke(prompt)
            response_clean = response.strip().replace("'", "").replace('"',"").replace('[', '').replace(']', '')

            # Divide a string pela vírgula para obter os nomes dos arquivos
            file_names_str = response_clean.split(',')

            # Limpa espaços em branco de cada nome de arquivo e remove strings vazias
            chosen_files = [name.strip() for name in file_names_str if name.strip()]

            if not chosen_files:
                return {'sucesso': False, 'erro': 'LLM retornou uma resposta vazia.'}

            for file_name in chosen_files:
                if file_name not in self.dataframes:
                    return {'sucesso': False, 'erro': f"O LLM retornou um nome de arquivo que não existe: '{file_name}'"}
            
            return {'sucesso': True, 'arquivos_escolhidos': chosen_files}
                
        except Exception as e:
            return {'sucesso': False, 'erro': f'Erro inesperado ao processar a resposta do LLM: {str(e)}'}

# SUBSTITUA TODA A SUA FUNÇÃO step1_interpret_question PELA VERSÃO FINAL ABAIXO

# SUBSTITUA TODA A SUA FUNÇÃO step1_interpret_question PELO CÓDIGO DE OURO ABAIXO

    def step1_interpret_question(self, question, selected_files):
        """ETAPA 1: LLM interpreta a pergunta e gera código Python usando um prompt mestre com exemplos."""
        
        # Lógica para lidar com um único arquivo
        if len(selected_files) == 1:
            df_name = selected_files[0]
            self.select_dataframe(df_name)

            dataset_info = f"""
INFORMAÇÕES DO DATASET:
- Nome do DataFrame: df
- Colunas disponíveis: {list(self.current_df.columns)}
"""
            prompt = f"""
Você é um especialista em Python/Pandas que gera pequenos trechos de código para responder a uma pergunta.

{dataset_info}

REGRAS CRÍTICAS E OBRIGATÓRIAS:
1.  O DataFrame `df` já existe. Opere diretamente nele.
2.  O resultado final DEVE ser armazenado em uma variável string chamada `resultado`.
3.  Gere APENAS o código Python. NÃO inclua `import`, `print`, comentários ou qualquer texto de explicação.

---
EXEMPLOS DE GABARITO (Use como guia para responder à pergunta do usuário):

# GABARITO 1: Pergunta sobre VALOR MÁXIMO
PERGUNTA: "Qual o produto mais caro?"
CÓDIGO GERADO:
linha_maior_valor = df.loc[df['VALOR UNITÁRIO'].idxmax()]
produto = linha_maior_valor['DESCRIÇÃO DO PRODUTO/SERVIÇO']
valor = linha_maior_valor['VALOR UNITÁRIO']
resultado = f"O produto com maior valor unitário é '{{produto}}' com valor de R$ {{valor:.2f}}."

# GABARITO 2: Pergunta sobre TOP N
PERGUNTA: "Mostre o top 5 produtos por quantidade"
CÓDIGO GERADO:
top_5 = df.groupby('DESCRIÇÃO DO PRODUTO/SERVIÇO')['QUANTIDADE'].sum().nlargest(5).reset_index()
resultado = f"O top 5 produtos por quantidade são:\\n{{top_5.to_string(index=False)}}"

# GABARITO 3: Pergunta sobre SOMA TOTAL
PERGUNTA: "Qual a soma total dos valores?"
CÓDIGO GERADO:
soma_total = df['VALOR TOTAL'].sum()
resultado = f"A soma total dos valores é R$ {{soma_total:.2f}}."

# GABARITO 4: Pergunta sobre CONTAGEM
PERGUNTA: "Quantos itens existem?"
CÓDIGO GERADO:
contagem = len(df)
resultado = f"Existem {{contagem}} registros de itens."

---
PERGUNTA REAL DO USUÁRIO: "{question}"

CÓDIGO PYTHON (Siga o gabarito mais parecido com a pergunta real):
"""

    # Lógica para lidar com múltiplos arquivos
        else:
            df_cabecalho = self.dataframes['202401_NFs_Cabecalho.csv']
            df_itens = self.dataframes['202401_NFs_Itens.csv']
            dataset_info = f"INFORMAÇÕES: Dois DataFrames estão carregados: `df_cabecalho` (colunas: {list(df_cabecalho.columns)}) e `df_itens` (colunas: {list(df_itens.columns)}). A chave para junção é 'CHAVE DE ACESSO'."
            safety_rules = "REGRAS: Gere APENAS código Python. NÃO use `print`. Salve a resposta na variável `resultado`."
            
            prompt = f"""
            {dataset_info}
            {safety_rules}
        PERGUNTA DO USUÁRIO: "{question}"

        INSTRUÇÃO OBRIGATÓRIA: Primeiro, junte os dataframes com `df_merged = pd.merge(df_cabecalho, df_itens, on='CHAVE DE ACESSO')`. Depois, analise o `df_merged` para responder à pergunta.

        EXEMPLO DE CÓDIGO:
        df_merged = pd.merge(df_cabecalho, df_itens, on='CHAVE DE ACESSO')
        linha_maior_valor = df_merged.loc[df_merged['VALOR TOTAL'].idxmax()]
        fornecedor = linha_maior_valor['RAZÃO SOCIAL EMITENTE_x']
        resultado = f"O fornecedor do item de maior valor total é '{{fornecedor}}'."

        CÓDIGO PYTHON:
        """

        try:
            response = self.llm.invoke(prompt)
            cleaned_code = response.strip()
            if cleaned_code.startswith('```python'):
                cleaned_code = cleaned_code[len('```python'):].strip()
            if cleaned_code.startswith('`'):
                cleaned_code = cleaned_code.strip('`').strip()
            if cleaned_code.endswith('```'):
                cleaned_code = cleaned_code[:-len('```')].strip()
            return cleaned_code
        except Exception as e:
            return f"Erro na interpretação: {str(e)}"

    def step2_execute_code(self, generated_code, selected_files):
        """ETAPA 2: Executa o código Python gerado pela LLM com validação."""
        
        # Cria um namespace seguro
        namespace = {
            'pd': pd,
            'resultado': None
        }

        # Adiciona os DataFrames necessários ao namespace
        if len(selected_files) == 1:
            namespace['df'] = self.dataframes[selected_files[0]]
        else:
            # Garante que os nomes das variáveis correspondam aos usados no prompt da Etapa 1
            namespace['df_cabecalho'] = self.dataframes['202401_NFs_Cabecalho.csv']
            namespace['df_itens'] = self.dataframes['202401_NFs_Itens.csv']

        try:
            exec(generated_code, namespace)
            resultado = namespace.get('resultado', 'Código executado mas variável resultado não encontrada')
            
            return {
                'sucesso': True,
                'resultado': resultado,
                'codigo_executado': generated_code
            }
            
        except Exception as e:
            return {
                'sucesso': False,
                'erro': str(e),
                'traceback': traceback.format_exc(),
                'codigo_executado': generated_code
            }    
        
        
# SUBSTITUA TODA A SUA FUNÇÃO step3_generate_response PELA VERSÃO ABAIXO

    def step3_generate_response(self, user_question, execution_result):
        """ETAPA 3: Formata números e gera resposta textual baseada nos dados."""

        if execution_result['sucesso']:
            try:
                locale.setlocale(locale.LC_ALL, 'pt_BR.UTF-8')
            except locale.Error:
                try:
                    locale.setlocale(locale.LC_ALL, 'Portuguese_Brazil.1252') # Padrão para Windows
                except locale.Error:
                    st.warning("Não foi possível configurar o locale para pt_BR. A formatação de moeda pode não estar ideal.")

            formatted_result = execution_result['resultado']
            
            # Procura por números (inteiros ou decimais) no texto do resultado
            # Esta expressão regular é um pouco complexa, mas eficaz
            if formatted_result is None:
                 formatted_result = "" 
            numbers_found = re.findall(r'(\d+\.\d+|\d+)', formatted_result)
            
            if numbers_found:
                # Pega o primeiro número encontrado para formatar
                # Supõe que em respostas como "A soma é X", X é o número que queremos formatar
                number_to_format_str = numbers_found[0]
                try:
                    number_to_format_float = float(number_to_format_str)
                    
                    # Formata o número como moeda brasileira
                    formatted_currency = locale.currency(number_to_format_float, grouping=True, symbol='R$')
                    
                    # Substitui o número original no texto pelo número formatado
                    formatted_result = formatted_result.replace(number_to_format_str, formatted_currency)

                except (ValueError, IndexError):
                    # Se não for um número válido, não faz nada
                    pass
            
            # --- FIM DA NOVA LÓGICA DE FORMATAÇÃO ---

            prompt = f"""
            Sua tarefa é criar uma frase clara e amigável em português a partir dos dados fornecidos.

            PERGUNTA ORIGINAL DO USUÁRIO: {user_question}
            
            DADOS JÁ FORMATADOS: 
            {formatted_result}

            INSTRUÇÕES:
            1.  Use os "DADOS JÁ FORMATADOS" para construir sua resposta. Eles já estão no formato correto.
            2.  Apresente a resposta de forma direta e natural.
            3.  NÃO mencione aspectos técnicos. Apenas a resposta final.

            Exemplo:
            DADOS JÁ FORMATADOS: "A soma total dos valores é R$ 3.371.754,84."
            RESPOSTA FINAL: "A soma total dos valores das notas fiscais é de R$ 3.371.754,84."

            RESPOSTA FINAL EM PORTUGUÊS:
            """
        else:
            # A lógica de erro continua a mesma
            prompt = f"""       
                Houve um erro ao processar a pergunta: {user_question}
                ERRO: {execution_result['erro']}
                Explique de forma simples que houve um problema e sugira uma reformulação da pergunta.
                Responda em português brasileiro.
                RESPOSTA:
                """
        
        try:
            response = self.llm.invoke(prompt)
            return response
        except Exception as e:
            return f"Erro ao gerar resposta: {str(e)}"

    def query_data(self, question):
        """Método principal que executa o fluxo autônomo completo."""
        
        if not self.dataframes:
            return "Por favor, carregue primeiro os arquivos CSV."
        
        with st.expander("🔍 Debug - Processo Autônomo Completo"):
            
            st.write("**ETAPA 0: Agente selecionando o(s) arquivo(s)...**")
            selection_result = self.step0_select_file(question)
            
            if not selection_result['sucesso']:
                error_message = f"Não consegui determinar qual arquivo usar. Detalhe: {selection_result['erro']}"
                st.error(error_message)
                return error_message

            arquivos_escolhidos = selection_result['arquivos_escolhidos']
            st.success(f"✅ Arquivo(s) escolhido(s): **{arquivos_escolhidos}**")

            st.write("**ETAPA 1: Interpretando pergunta e gerando código...**")
            generated_code = self.step1_interpret_question(question, arquivos_escolhidos)
            st.code(generated_code, language='python')
            
            st.write("**ETAPA 2: Executando código...**")
            execution_result = self.step2_execute_code(generated_code, arquivos_escolhidos)
            
            if execution_result['sucesso']:
                st.success("✅ Código executado com sucesso!")
                st.write(f"**Resultado:** {execution_result['resultado']}")
            else:
                st.error("❌ Erro na execução do código gerado:")
                st.code(execution_result['erro'])
                if 'traceback' in execution_result:
                    st.code(execution_result['traceback'])
            
            st.write("**ETAPA 3: Gerando resposta final...**")
        
        final_response = self.step3_generate_response(question, execution_result)
        
        return final_response
    
    def get_dataframe_info(self, df_name):
        """Retorna informações sobre um DataFrame"""
        if df_name in self.dataframes:
            df = self.dataframes[df_name]
            info = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "sample": df.head(5).to_dict('records'),
                "total_rows": len(df),
                "memory_usage": df.memory_usage(deep=True).sum()
            }
            return info
        return None
    
    def get_quick_stats(self, df_name):
        """Retorna estatísticas rápidas sem usar o agente"""
        if df_name in self.dataframes:
            df = self.dataframes[df_name]
            stats = {
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "numeric_columns": len(df.select_dtypes(include=['number']).columns),
                "text_columns": len(df.select_dtypes(include=['object']).columns),
                "null_values": df.isnull().sum().sum(),
                "duplicated_rows": df.duplicated().sum()
            }
            return stats
        return None

    def get_column_analysis(self, df_name):
        """Retorna análise detalhada das colunas para ajudar na identificação"""
        if df_name in self.dataframes:
            df = self.dataframes[df_name]
            analysis = {}
            
            for col in df.columns:
                analysis[col] = {
                    "tipo": str(df[col].dtype),
                    "valores_unicos": df[col].nunique(),
                    "nulos": df[col].isnull().sum(),
                    "exemplo": str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "N/A"
                }
            
            return analysis
        return None


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