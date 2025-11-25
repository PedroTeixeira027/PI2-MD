PROJETO INDIVIDUAL 2 (PI2)
TÉCNICAS DE AGRUPAMENTO – MACHINE LEARNING NÃO SUPERVISIONADO
1. Introdução

Este projeto aplica técnicas de aprendizado de máquina não supervisionado para analisar um conjunto de dados fictício relacionado ao perfil de clientes de uma empresa de varejo online. O objetivo central é identificar padrões de comportamento que permitam segmentar os consumidores em grupos com características semelhantes, auxiliando decisões estratégicas de marketing, retenção e personalização de ofertas.

Foram selecionadas duas técnicas de agrupamento:

K-Means

DBSCAN

Esses algoritmos permitem diferentes perspectivas sobre os dados:
K-Means busca grupos compactos e esféricos; DBSCAN identifica regiões densas e detecta ruídos/outliers.

2. Descrição do Problema

Empresas de comércio eletrônico acumulam grandes volumes de dados sobre hábitos de compra e navegação. Entretanto, esses dados costumam ser pouco organizados e demandam métodos capazes de descobrir padrões escondidos.

O problema proposto consiste em:

Segmentar consumidores a partir de características como número de compras, ticket médio, frequência e tempo de uso da plataforma, identificando grupos de perfis semelhantes.

Os grupos esperados podem revelar clientes VIP, compradores ocasionais, usuários em risco de churn, entre outros.

3. Conjunto de Dados Fictício

O dataset fictício contém as seguintes variáveis:

Variável	Descrição
cliente_id	Identificador do usuário
compras_ano	Número de compras realizadas no ano
ticket_medio	Valor médio gasto por compra
tempo_plataforma	Meses usando a plataforma
frequencia_acesso	Número médio de acessos mensais

Todas as variáveis são contínuas, exceto o ID.

4. Processo de ETL e Limpeza dos Dados
4.1. Extração

Os dados são gerados de forma programática no próprio script Python, simulando comportamento realista.

4.2. Transformação

Foram aplicadas:

Remoção de valores nulos;

Padronização das escalas utilizando StandardScaler;

Exclusão do identificador cliente_id para treinar os modelos.

4.3. Carregamento

Os dados tratados foram armazenados em um DataFrame pronto para análise.

5. Código Completo

O código foi desenvolvido em Python, utilizando:
pandas, numpy, matplotlib, seaborn, sklearn e scipy.

(O código integral já foi fornecido anteriormente. Caso deseje a versão comentada linha a linha, posso gerar.)

6. Visualização dos Agrupamentos

Foram gerados gráficos:

Dispersão 2D após redução de dimensionalidade usando PCA;

Cores representando os clusters formados por cada algoritmo;

Identificação visual clara de agrupamentos e outliers.

7. Resultados e Análise
7.1. K-Means

O algoritmo de K-Means dividiu os clientes em 3 clusters principais:

Cluster 0 – Clientes de alto valor:
Alto ticket médio, alta frequência e grande volume de compras.

Cluster 1 – Clientes moderados:
Compram regularmente, mas gastam menos.

Cluster 2 – Clientes ocasionais:
Baixa frequência e pouco engajamento.

K-Means demonstrou estrutura bem definida e interpretável, refletindo perfis claros.

7.2. DBSCAN

DBSCAN identificou:

Grupos compactos de comportamento denso, semelhantes aos clusters do K-Means.

Outliers representando usuários atípicos, como clientes com gastos extremamente altos ou extremamente baixos.

Sua capacidade de detectar ruído é vantagem relevante em análises de comportamento humano.

8. Comparação Entre K-Means e DBSCAN
Aspecto	K-Means	DBSCAN
Detecta outliers?	Não	Sim
Requer número de clusters definido?	Sim	Não
Forma dos clusters	Esféricos	Arbitrária
Interpretação	Intuitiva	Moderada

Em combinação, os dois métodos fornecem visão completa:
K-Means revela perfis típicos; DBSCAN destaca exceções e anomalias.

9. Conclusão

A análise demonstrou que técnicas de aprendizagem não supervisionada são eficazes para segmentação de clientes. Os grupos encontrados podem orientar ações como:

Campanhas específicas para clientes VIP;

Estratégias de retenção para perfis de risco;

Personalização de ofertas por comportamento.

A utilização conjunta de K-Means e DBSCAN oferece resultados robustos, permitindo identificar perfis gerais e indivíduos atípicos que exigem tratamento diferenciado.

10. Referências

SKLEARN. Scikit-learn Machine Learning in Python.

HASTIE, Tibshirani & Friedman. The Elements of Statistical Learning.

TAN, Steinbach & Kumar. Introduction to Data Mining.
