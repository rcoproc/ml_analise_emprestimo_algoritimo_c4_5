"""
Algoritmo de Classificação de Solicitações de Empréstimo usando Árvores de Decisão
Criado por Ricardo Oliveira em 19/05/25
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

# Definição do conjunto de dados
def criar_conjunto_dados():
    """
    Cria um conjunto de dados com informações de clientes:
    - idade: idade do cliente em anos (numérico)
    - renda: renda mensal do cliente em reais (numérico)
    - historico_credito: bom, regular ou ruim (categórico)
    - emprego: tempo de emprego em anos (numérico)
    - propriedade: sim ou não (categórico)
    - conceder_emprestimo: sim ou não (categórico - alvo)
    """
    dados = {
        'idade': [25, 35, 45, 28, 55, 65, 32, 39, 42, 48, 30, 60, 27, 44, 38, 52, 29, 33, 47, 36],
        'renda': [3000, 6000, 4000, 2500, 8000, 3500, 5000, 6500, 7000, 9000, 4500, 5500, 2800, 8500, 7500, 6200, 3200, 4800, 5200, 6800],
        'historico_credito': ['ruim', 'bom', 'regular', 'regular', 'bom', 'ruim', 'bom', 'bom', 'bom', 'bom', 'regular', 'bom', 'ruim', 'bom', 'bom', 'bom', 'regular', 'bom', 'regular', 'bom'],
        'emprego': [0.5, 5, 2, 1, 10, 12, 3, 7, 8, 15, 2, 20, 0.8, 9, 6, 11, 1.5, 3.5, 6.5, 5.5],
        'propriedade': ['não', 'sim', 'não', 'não', 'sim', 'sim', 'sim', 'sim', 'sim', 'sim', 'não', 'sim', 'não', 'sim', 'sim', 'sim', 'não', 'sim', 'não', 'sim'],
        'conceder_emprestimo': ['não', 'sim', 'não', 'não', 'sim', 'não', 'sim', 'sim', 'sim', 'sim', 'não', 'sim', 'não', 'sim', 'sim', 'sim', 'não', 'sim', 'não', 'sim']
    }
    
    return pd.DataFrame(dados)

def preparar_dados(df):
    """
    Prepara os dados para o treinamento:
    - Converte as variáveis categóricas em numéricas
    - Separa as features do target
    """
    # Codificar variáveis categóricas
    le_historico = LabelEncoder()
    le_propriedade = LabelEncoder()
    le_emprestimo = LabelEncoder()
    
    df_processado = df.copy()
    df_processado['historico_credito_num'] = le_historico.fit_transform(df['historico_credito'])
    df_processado['propriedade_num'] = le_propriedade.fit_transform(df['propriedade'])
    df_processado['conceder_emprestimo_num'] = le_emprestimo.fit_transform(df['conceder_emprestimo'])
    
    # Separar features e target
    X = df_processado[['idade', 'renda', 'historico_credito_num', 'emprego', 'propriedade_num']]
    y = df_processado['conceder_emprestimo_num']
    
    return X, y, le_historico, le_propriedade, le_emprestimo

def treinar_arvore_decisao(X, y):
    """
    Divide os dados em treinamento e teste (70% / 30%)
    Treina o modelo de árvore de decisão
    """
    # Dividir dados em treinamento e teste
    X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Criar e treinar o modelo de árvore de decisão
    arvore = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
    arvore.fit(X_treino, y_treino)
    
    return arvore, X_treino, X_teste, y_treino, y_teste

def avaliar_modelo(modelo, X_teste, y_teste):
    """
    Avalia o desempenho do modelo através de:
    - Acurácia
    - Relatório de classificação (precisão, recall, f1-score)
    - Matriz de confusão
    """
    y_pred = modelo.predict(X_teste)
    acuracia = accuracy_score(y_teste, y_pred)
    relatorio = classification_report(y_teste, y_pred)
    matriz = confusion_matrix(y_teste, y_pred)
    
    print(f"Acurácia do modelo: {acuracia:.4f}")
    print("\nRelatório de Classificação:")
    print(relatorio)
    print("\nMatriz de Confusão:")
    print(matriz)
    
    return acuracia, relatorio, matriz

def visualizar_arvore(modelo, X, feature_names):
    """
    Visualiza a árvore de decisão treinada
    """
    plt.figure(figsize=(20, 10))
    plot_tree(modelo, feature_names=feature_names, class_names=['Negar', 'Conceder'], filled=True, rounded=True)
    plt.title("Árvore de Decisão para Classificação de Empréstimos")
    plt.show()

def classificar_novo_cliente(modelo, novo_cliente, le_historico, le_propriedade):
    """
    Classifica um novo cliente com base no modelo treinado
    """
    # Processar os dados do novo cliente
    historico_num = le_historico.transform([novo_cliente['historico_credito']])[0]
    propriedade_num = le_propriedade.transform([novo_cliente['propriedade']])[0]
    
    # Criar array para predição
    cliente_array = np.array([[
        novo_cliente['idade'],
        novo_cliente['renda'],
        historico_num,
        novo_cliente['emprego'],
        propriedade_num
    ]])
    
    # Fazer a predição
    resultado = modelo.predict(cliente_array)[0]
    probabilidades = modelo.predict_proba(cliente_array)[0]
    
    return resultado, probabilidades

# Programa principal
if __name__ == "__main__":
    # Carregar dados
    print("Carregando conjunto de dados(base do treinamento)...")
    df = criar_conjunto_dados()
    print("\nAmostra dos dados:")
    print(df.head())
    print("\nEstatísticas descritivas:")
    print(df.describe())
    print("\nDistribuição das classes:")
    print(df['conceder_emprestimo'].value_counts())
    
    # Preparar dados
    print("\nPasso 1) Preparando dados...")
    X, y, le_historico, le_propriedade, le_emprestimo = preparar_dados(df)
    
    # Treinar modelo
    print("\nTreinando modelo de árvore de decisão...")
    modelo, X_treino, X_teste, y_treino, y_teste = treinar_arvore_decisao(X, y)
    
    # Avaliar modelo
    print("\nPasso 3) Avaliando desempenho do modelo:")
    acuracia, relatorio, matriz = avaliar_modelo(modelo, X_teste, y_teste)
    
    # Visualizar importância das características
    print("\nPasso 4)Importância das características:")
    for i, feature in enumerate(['Idade', 'Renda', 'Histórico de Crédito', 'Tempo de Emprego', 'Propriedade']):
        print(f"{feature}: {modelo.feature_importances_[i]:.4f}")
    
    # Visualizar árvore
    print("\nPasso 5) Visualizando árvore de decisão...")
    visualizar_arvore(modelo, X, ['Idade', 'Renda', 'Histórico de Crédito', 'Tempo de Emprego', 'Propriedade'])
    
    # Testar com alguns exemplos significativos
    print("\nPasso 6) Testando classificação com exemplos significativos:")
    
    exemplos = [
        {
            'idade': 30,
            'renda': 5000,
            'historico_credito': 'bom',
            'emprego': 5,
            'propriedade': 'sim'
        },
        {
            'idade': 22,
            'renda': 2000,
            'historico_credito': 'ruim',
            'emprego': 0.5,
            'propriedade': 'não'
        },
        {
            'idade': 45,
            'renda': 7000,
            'historico_credito': 'regular',
            'emprego': 8,
            'propriedade': 'sim'
        },
        {
            'idade': 53,
            'renda': 14000,
            'historico_credito': 'regular',
            'emprego': 4,
            'propriedade': 'sim'
        }
    ]
    
    for i, exemplo in enumerate(exemplos):
        resultado, probabilidades = classificar_novo_cliente(modelo, exemplo, le_historico, le_propriedade)
        decisao = 'CONCEDIDO' if resultado == 1 else 'NEGADO'
        print(f"\nExemplo {i+1}:")
        print(f"Idade: {exemplo['idade']} anos")
        print(f"Renda: R$ {exemplo['renda']},00")
        print(f"Histórico de Crédito: {exemplo['historico_credito']}")
        print(f"Tempo de Emprego: {exemplo['emprego']} anos")
        print(f"Possui Propriedade: {exemplo['propriedade']}")
        print(f"Resultado: EMPRÉSTIMO {decisao}")
        print(f"Probabilidade de negação: {probabilidades[0]:.2f}")
        print(f"Probabilidade de concessão: {probabilidades[1]:.2f}")
    
    print("\nPrograma concluído!")