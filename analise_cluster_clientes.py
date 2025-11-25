# ===============================================================
# PI2 - Mineração de Dados Não Supervisionada
# ===============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ajuste de estilo
sns.set(style="whitegrid")

# ---------------------------------------------------------------
# 1. CRIAÇÃO DO DATASET FICTÍCIO
# ---------------------------------------------------------------

np.random.seed(42)
n = 700  # quantidade de clientes fictícios

data = pd.DataFrame({
    "compras_mensais": np.random.randint(1, 30, n),
    "gasto_medio": np.random.uniform(20, 600, n),
    "tempo_de_cadastro": np.random.randint(1, 60, n),  # meses
    "visitas_mensais": np.random.randint(1, 50, n),
    "avaliacao_media": np.random.uniform(1, 5, n)
})

# Simular valores faltantes (apenas para demonstrar ETL real)
for col in data.columns:
    if np.random.rand() < 0.15:
        data.loc[np.random.choice(n, 5), col] = np.nan

print("Prévia dos dados:")
print(data.head())

# ---------------------------------------------------------------
# 2. ETL - LIMPEZA E TRATAMENTO
# ---------------------------------------------------------------

# Verificação de valores ausentes
print("\nValores faltantes por coluna:")
print(data.isnull().sum())

# Tratamento de valores faltantes (substituindo pela mediana)
data = data.fillna(data.median())

# Remoção de duplicatas
data = data.drop_duplicates()

print("\nDados após limpeza:")
print(data.head())

# ---------------------------------------------------------------
# 3. ANÁLISE E VISUALIZAÇÃO INICIAL
# ---------------------------------------------------------------

plt.figure(figsize=(10, 6))
sns.heatmap(data.corr(), annot=True, cmap="Blues")
plt.title("Correlação entre Variáveis")
plt.show()

# Visualização com pairplot
sns.pairplot(data)
plt.show()

# Histogramas das variáveis
data.hist(figsize=(12, 7), bins=20)
plt.suptitle("Distribuição das Variáveis", y=1.02)
plt.show()

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

sns.set(style="whitegrid")
plt.rcParams.update({"figure.max_open_warning": 0})

try:
    data  # check if data variable exists
except NameError:
    np.random.seed(42)
    n = 700
    data = pd.DataFrame({
        "compras_mensais": np.random.randint(1, 30, n),
        "gasto_medio": np.random.uniform(20, 600, n),
        "tempo_de_cadastro": np.random.randint(1, 60, n),
        "visitas_mensais": np.random.randint(1, 50, n),
        "avaliacao_media": np.random.uniform(1, 5, n)
    })
    # introduzindo alguns NaNs para demonstrar ETL
    for col in data.columns:
        if np.random.rand() < 0.15:
            data.loc[np.random.choice(n, 5), col] = np.nan
    data = data.fillna(data.median()).drop_duplicates()

# -------------------------
# (B) Pré-processamento (ETL final)
# -------------------------
features = data.columns.tolist()
X = data[features].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# (C) Redução de dimensionalidade para visualização (PCA 2 componentes)
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df = pd.concat([pca_df, data.reset_index(drop=True)], axis=1)

# Visualização inicial PCA (sem clusters)
plt.figure(figsize=(8, 6))
sns.scatterplot(x="PC1", y="PC2", data=pca_df, alpha=0.7)
plt.title("PCA 2D - Distribuição dos clientes (sem clusters)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()

# -------------------------
# (D) K-MEANS
# -------------------------
# 1) Escolher k via elbow + silhouette
inertia = []
sil_scores = []
K_range = range(2, 11)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

# Plot elbow
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(list(K_range), inertia, marker="o")
plt.title("Elbow Method (Inertia) - KMeans")
plt.xlabel("Número de clusters k")
plt.ylabel("Inertia")

# Plot silhouette by k
plt.subplot(1, 2, 2)
plt.plot(list(K_range), sil_scores, marker="o")
plt.title("Silhouette Score x k")
plt.xlabel("Número de clusters k")
plt.ylabel("Silhouette Score")
plt.tight_layout()
plt.show()

# Escolha de k: pega o k com maior silhouette (automático), mas o usuário pode ajustar
best_k = K_range[int(np.argmax(sil_scores))]
print(f">> k escolhido automaticamente (melhor silhouette): {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
k_labels = kmeans.fit_predict(X_scaled)

# Plot clusters em PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=k_labels, palette="tab10", legend="full", alpha=0.8)
plt.title(f"K-Means (k={best_k}) - clusters em PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()

# Perfil dos clusters (médias das features)
kmeans_profile = pd.DataFrame(X_scaled).groupby(k_labels).mean()
kmeans_profile.columns = features
kmeans_profile.index.name = "cluster"
print("\nPerfil médio (padronizado) por cluster - KMeans:")
print(kmeans_profile)

# Mostrar médias em escala original para interpretação
orig_profile = pd.DataFrame(X, columns=features).groupby(k_labels).mean()
orig_profile.index.name = "cluster"
print("\nPerfil médio (original) por cluster - KMeans:")
print(orig_profile)

print("\nSilhouette score (KMeans):", silhouette_score(X_scaled, k_labels))

# -------------------------
# (E) DBSCAN
# -------------------------
# 1) k-distance plot (k = min_samples) para ajudar a escolher eps
min_samples = max(5, X_scaled.shape[1] + 1)  # regra prática: dim+1 ou 5
nbrs = NearestNeighbors(n_neighbors=min_samples).fit(X_scaled)
distances, indices = nbrs.kneighbors(X_scaled)
# distances to k-th nearest neighbor
k_distances = np.sort(distances[:, -1])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.ylabel(f"Distância para o {min_samples}º vizinho")
plt.xlabel("Pontos ordenados por distância")
plt.title("k-distance plot (ajuda a escolher eps para DBSCAN)")
plt.show()

# 2) Testar vários eps e escolher pelo silhouette (apenas para os pontos que não são ruído)
eps_values = np.linspace(k_distances.mean()*0.5, k_distances.mean()*1.6, 12)
db_results = []
for eps in eps_values:
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db_labels = db.fit_predict(X_scaled)
    n_clusters = len(set(db_labels)) - (1 if -1 in db_labels else 0)
    # compute silhouette only when at least 2 clusters (excluding noise)
    if n_clusters >= 2:
        mask = db_labels != -1
        try:
            sil = silhouette_score(X_scaled[mask], db_labels[mask])
        except Exception:
            sil = -1
    else:
        sil = -1
    db_results.append((eps, n_clusters, sil))

db_results_df = pd.DataFrame(db_results, columns=["eps", "n_clusters", "silhouette"])
print("\nResultados DBSCAN testando eps (min_samples={}):".format(min_samples))
print(db_results_df)

# Escolher eps com maior silhouette (>=0), se existir
valid = db_results_df[db_results_df["silhouette"] > -0.5]  # filtro conservador
if not valid.empty:
    best_eps = valid.sort_values("silhouette", ascending=False).iloc[0]["eps"]
else:
    # fallback para eps mediana do k-dist se nenhum for satisfatório
    best_eps = float(k_distances.mean())
print(f"\n>> eps escolhido automaticamente para DBSCAN: {best_eps:.4f}")

db = DBSCAN(eps=best_eps, min_samples=min_samples)
db_labels = db.fit_predict(X_scaled)

# Plot DBSCAN clusters in PCA
plt.figure(figsize=(8, 6))
palette = sns.color_palette("tab10", np.unique(db_labels).size)
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=db_labels, palette=palette, legend="full", alpha=0.8)
plt.title(f"DBSCAN (eps={best_eps:.3f}, min_samples={min_samples}) - clusters em PCA 2D")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.legend(title="Cluster")
plt.show()

# DBSCAN profile
n_clusters_db = len(set(db_labels)) - (1 if -1 in db_labels else 0)
n_noise = sum(1 for l in db_labels if l == -1)
print(f"\nDBSCAN encontrou {n_clusters_db} clusters e {n_noise} pontos de ruído (-1).")

if n_clusters_db >= 1:
    # medias por cluster (original scale)
    db_profile = pd.DataFrame(X, columns=features).groupby(db_labels).mean().sort_index()
    db_profile.index.name = "cluster_label"
    print("\nPerfil médio (original) por label - DBSCAN (inclui -1 ruído):")
    print(db_profile)

# Silhouette para DBSCAN (excluindo ruído)
if n_clusters_db >= 2:
    mask = db_labels != -1
    sil_db = silhouette_score(X_scaled[mask], db_labels[mask])
    print("\nSilhouette score (DBSCAN, sem ruído):", sil_db)
else:
    print("\nSilhouette score (DBSCAN): não aplicável (menos de 2 clusters)")

# -------------------------
# (F) Comparação resumida
# -------------------------
print("\n--- Resumo comparativo ---")
print(f"K-Means: k = {best_k}, clusters = {len(set(k_labels))}")
print(f"DBSCAN: eps = {best_eps:.4f}, min_samples = {min_samples}, clusters = {n_clusters_db}, noise points = {n_noise}")

# Salvar resultados (opcional)
results_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "kmeans_label": k_labels,
    "dbscan_label": db_labels
})
results_df = pd.concat([results_df.reset_index(drop=True), data.reset_index(drop=True)], axis=1)
results_df.to_csv("pi2_cluster_results.csv", index=False)
print("\nArquivo 'pi2_cluster_results.csv' salvo com os rótulos dos clusters e as features originais.")
