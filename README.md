📘 Projeto Individual 2 (PI2) — Mineração de Dados (Não Supervisionado)
Este projeto aplica técnicas de aprendizado de máquina não supervisionado utilizando um dataset fictício de clientes de uma empresa digital. O objetivo é identificar padrões de comportamento e segmentar usuários por similaridade.

📌 1. Descrição do Problema

A empresa fictícia ShopNow deseja identificar grupos naturais de clientes para apoiar estratégias de marketing.
O dataset contém variáveis como:

- Compras mensais
- Gasto médio
- Tempo de cadastro
- Visitas mensais
- Avaliação média

Os métodos aplicados foram:
- K-Means
- DBSCAN

🧹 2. ETL — Preparação e Limpeza

- Dataset fictício gerado artificialmente.
- Remoção de duplicatas.
- Tratamento de valores ausentes usando mediana.
- Padronização para os algoritmos que exigem escala.

O objetivo foi garantir dados consistentes para o processo de clusterização.

📊 3. Análise Exploratória

Foram produzidas:
- Heatmap de correlação
- Pairplot
- Histogramas das variáveis

As análises mostraram diferentes padrões de comportamento, sugerindo a presença de múltiplos grupos possíveis.

🌈 4. Resultados — K-Means

O algoritmo formou 3 clusters principais representando perfis distintos de clientes:
- Usuários de baixo engajamento
- Usuários moderados
- Usuários de alto valor

Os grupos ficaram bem separados após a padronização, e os centróides ajudaram a interpretar cada segmento.

🌀 5. Resultados — DBSCAN

- Detectou grupos menores e mais densos.
- Identificou pontos de ruído/outliers, úteis para identificar comportamentos atípicos.
- Revelou padrões que não aparecem tão claramente no K-Means.

🧠 Conclusão

Os dois métodos se complementam:
- K-Means oferece uma segmentação clara e estruturada.
- DBSCAN revela outliers e agrupamentos de alta densidade.

O projeto demonstra como técnicas não supervisionadas podem extrair insights relevantes mesmo sem variáveis-alvo.
