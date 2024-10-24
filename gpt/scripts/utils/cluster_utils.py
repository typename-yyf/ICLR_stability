import numpy as np
from sklearn.decomposition import PCA
import torch
from sklearn.cluster import DBSCAN, KMeans
from sklearn.manifold import TSNE


def cal_cluster_center(embeddings, labels):
    labelwise_embeddings = [[] for _ in set(labels)]
    for label, emb in zip(labels, embeddings):
        labelwise_embeddings[label].append(emb)

    centers = []
    for i in range(len(labelwise_embeddings)):
        labelwise_embeddings[i] = np.array(labelwise_embeddings[i])
        centers.append(labelwise_embeddings[i].mean(0))
    
    radius = []
    for i in range(len(labelwise_embeddings)):
        radius.append(np.linalg.norm(labelwise_embeddings[i] - centers[i], axis=1).mean())

    return centers, radius


def normalize(embeddings, new_min=-1, new_max=1):
    for i in range(embeddings.shape[1]):
        old_min = embeddings[:, i].min()
        old_max = embeddings[:, i].max()
        embeddings[:, i] = (embeddings[:, i] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return embeddings


def np_PCA(embeddings, n_comp = 4):
    pca = PCA(n_components=n_comp)  # 
    pca.fit(embeddings)
    new_emb = pca.transform(embeddings)
    new_emb = normalize(new_emb)
    return new_emb


def torch_PCA(embeddings, n_comp = 4):
    embeddings = embeddings.cpu().numpy()
    pca = PCA(n_components=n_comp)
    pca.fit(embeddings)
    new_emb = pca.transform(embeddings)
    new_emb = normalize(new_emb)

    return new_emb


def embed_space_2_cluster_space(embeddings):
    if isinstance(embeddings, np.ndarray):
        return np_PCA(embeddings, 4)
    elif isinstance(embeddings, torch.Tensor):
        return torch_PCA(embeddings, 4)



def cluster_embedding_space(rank, model, source, reflayer, clust_num):
    model.eval()
    model = model.to(rank)

    embeddings = []
    with torch.no_grad():
        batch_size = 100
        source = source.to(rank)
        for i in range(0, len(source), batch_size):
            embedding = model.get_attn_output(source[i:i+batch_size], reflayer)
            embeddings.append(embedding.mean(1, keepdim = False).to('cpu').numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    clust_alg = KMeans(n_clusters=clust_num, random_state=0, n_init=10).fit(embeddings)
    cluster_labels = clust_alg.labels_
    # cal number of each cluster
    unique, counts = np.unique(cluster_labels, return_counts=True)
    count_dict = dict(zip(unique, counts))
    # sort count_dict by count
    count_dict = dict(sorted(count_dict.items(), key=lambda x: x[1], reverse=True))
    # print(count_dict)

    centers, radius = cal_cluster_center(embeddings, clust_alg.labels_)

    return centers, radius, count_dict

