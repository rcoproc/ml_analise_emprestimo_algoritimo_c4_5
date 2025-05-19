# Algoritmo de Classificação de Empréstimos com Árvore de Decisão v1.0

Algoritmo na liguagem python3 que utiliza árvores de decisão para classificar solicitações de empréstimo com base nas informações dos clientes.


## Funcionalidades implementadas:

Conjunto de dados simulado com informações relevantes:

Idade do cliente
Renda mensal
Histórico de crédito (bom, regular, ruim)
Tempo de emprego em anos
Propriedade (sim/não)
Resultado (conceder/negar empréstimo)
Processamento dos dados:

## Conversão de variáveis categóricas para numéricas

Divisão em conjuntos de treinamento (70%) e teste (30%)
Treinamento da árvore de decisão:

## Uso do algoritmo C4.5 (através da implementação DecisionTreeClassifier do scikit-learn)

Critério de divisão baseado em entropia
Profundidade máxima de 4 níveis para evitar overfitting
Avaliação de desempenho:

## Cálculo da acurácia do modelo
Relatório completo com precisão, recall e f1-score
Matriz de confusão para análise de erros
Visualização da árvore:

## Geração de um gráfico da árvore treinada com todas as regras de decisão
Identificação da importância de cada característica
Classificação de novos clientes:

## Função para classificar novas solicitações
Exemplos significativos de diferentes perfis de clientes
Exibição das probabilidades de concessão/negação
Exemplo de uso:


Para executar o algoritmo, basta rodar o script no terminal:

```bash
python3 exercicio_py_charm.py
