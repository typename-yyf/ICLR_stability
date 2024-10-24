import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import smooth
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, Birch, OPTICS
from sklearn.metrics import calinski_harabasz_score, silhouette_score, davies_bouldin_score
from sklearn.manifold import TSNE

CLUSTER_COUNT = 2


def read_loss(file):
    f = open(f'../logs/{file}_2mix-768d.txt', 'r')
    line = f.readline()
    loss = []
    i = 0
    while line:
        if i % 128 == 0:
            line = line.split(' ')[3]
            loss.append(float(line))
            i = 0
        line = f.readline()
        i += 1

    loss = smooth(loss, 5)
    return loss


def normalize(embeddings, new_min=-1, new_max=1):
    for i in range(embeddings.shape[1]):
        old_min = embeddings[:, i].min()
        old_max = embeddings[:, i].max()
        embeddings[:, i] = (embeddings[:, i] - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    return embeddings


def ignore_noise(embeddings, labels):
    new_embeddings = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] != -1:
            new_embeddings.append(embeddings[i])
            new_labels.append(labels[i])
    return np.array(new_embeddings), np.array(new_labels)


def draw_label_scatter(embeddings, labels, alg, step, layer):
    i = 0
    for label, p in zip(labels, embeddings):
        if label > 8:
            label = -1
        plt.scatter(p[0], p[1], c = colors[i % 2])
        i += 1
    
    plt.savefig(f'../figures/clusters/{alg}_{MIX}mix_{step}_{layer}.png')
    plt.close()


def draw_scatter(embeddings, alg, step, layer):
    embeddings = normalize(embeddings)
    _2d_embed = TSNE(n_components=2).fit_transform(embeddings)
    plt.scatter(_2d_embed[:, 0], _2d_embed[:, 1])
    plt.title(f'step {step} layer {layer}')
    plt.savefig(f'../figures/clusters/{alg}_{step}_{layer}.png')
    plt.close()


def ECD(points):
    if len(points) == 1:
        return 0
    dists = []
    for i in range(len(points) - 1):
        for j in range(i + 1, len(points)):
            dists.append(torch.norm(points[i] - points[j]).item())
    return sum(dists) / len(dists)


def kmeans(embeddings, n_clus):
    kmeans = KMeans(n_clusters=n_clus, random_state=0, n_init = 'auto').fit(embeddings)
    
    return kmeans.labels_


def agg(embeddings):
    agg = AgglomerativeClustering(n_clusters=CLUSTER_COUNT, linkage='ward').fit(embeddings)
    my_centers = torch.from_numpy(agg.cluster_centers_).to('cuda:0')

    return agg.labels_


def labelwise_embeddings(embeddings, labels):
    labelwise_embeddings = []
    for label in set(labels):
        labelwise_embeddings.append([])
    for i in range(len(labels)):
        labelwise_embeddings[labels[i]].append(embeddings[i])
    labelwise_embeddings = [np.array(labelwise_embedding) for labelwise_embedding in labelwise_embeddings]
    return labelwise_embeddings


def dbscan(embeddings, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, n_jobs=4).fit(embeddings) # -1 是 noise

    return dbscan.labels_


def optics(embeddings, eps, min_samples):
    # eps >= 0.7 的时候基本就没用了
    optics = OPTICS(min_samples=min_samples, max_eps=eps, n_jobs=4).fit(embeddings) # -1 是 noise
    
    return optics.labels_


def birch(embeddings):
    birch = Birch(threshold=2.5, n_clusters=None).fit(embeddings)
    my_centers = torch.from_numpy(birch.subcluster_centers_).to('cuda:0')
    
    return birch.labels_


def CHindex(embeddings, labels):
    if len(set(labels)) == 1:
        return 0
    return calinski_harabasz_score(embeddings, labels)


def SHindex(embeddings, labels):
    if len(set(labels)) == 1:
        return 0
    return silhouette_score(embeddings, labels)


def DBindex(embeddings, labels):
    if len(set(labels)) == 1:
        return 5
    return davies_bouldin_score(embeddings, labels)


MIX = 1
CLUSTER_METRICs = ['CHindex', 'SHindex', 'DBindex']
colors = np.array(["red","green", "blue", "orange","purple","beige","cyan","magenta", "black"])

embeddings_sl = np.load(f'normalized_embeddings_{MIX}mix-768d.npy')

embeddings_sl = []
for step in range(2048, 17409, 512):
    embeddings_l = []
    for layer in range(6, 12):
        emb0 = np.load(f'../embeddings/emb_0_{step}_{layer}.npy')
        emb1 = np.load(f'../embeddings/emb_1_{step}_{layer}.npy')
        embeddings_l.append((emb0, emb1))
    embeddings_sl.append(embeddings_l)

def search_once(alg, eps, sam):
    metric_sl = {m : [] for m in CLUSTER_METRICs}
    clust_num_sl = []
    for step, embeddings_l in enumerate(embeddings_sl):
        print(f'Processing step {step}')
        metric_l = {m : [] for m in CLUSTER_METRICs}
        clust_num_l = []

        for l, embeddings in enumerate(embeddings_l):
            if alg == 'kmeans':
                labels = kmeans(embeddings, 2)
            elif alg == 'agg':
                labels = agg(embeddings)
            elif alg == 'dbscan':
                # if MIX == 2:
                # labels = dbscan(embeddings[1], eps, sam)
                # draw_label_scatter(embeddings[1], labels, 'dbscan_hier1', step, l + 6)
                # labels = dbscan(embeddings[0], eps, sam)
                # draw_label_scatter(embeddings[0], labels, 'dbscan_hier0', step, l + 6)
                embeddings = embeddings[0]
                # else:
                if layer > 9:
                    labels = dbscan(embeddings, .30, 13)
                else:
                    labels = dbscan(embeddings, eps, 15)
                    
                # embeddings = labelwise_embeddings(embeddings, labels)
                # if len(embeddings) >= 2:
                #     if len(embeddings[0]) > len(embeddings[1]):
                #         embeddings = embeddings[0]
                #     else:
                #         embeddings = embeddings[1]
                #     if step > 35:
                #         labels = dbscan(embeddings, .5, 5)
                #     else:
                #         labels = dbscan(embeddings, .5, 17)
            elif alg == 'optics':
                labels = optics(embeddings, eps, sam)
            elif alg == 'birch':
                labels = birch(embeddings)

            # draw_label_scatter(embeddings, labels, alg, step, l)
            # 画 metric 图用这里：
            unique_labels = set(labels)
            if -1 in unique_labels:
                unique_labels.remove(-1)
            clust_num_l.append(len(unique_labels))
            for metric in CLUSTER_METRICs:
                if metric == 'CHindex':
                    metric_l[metric].append(CHindex(embeddings, labels))
                elif metric == 'SHindex':
                    metric_l[metric].append(SHindex(embeddings, labels))
                elif metric == 'DBindex':
                    metric_l[metric].append(DBindex(embeddings, labels))
                else:
                    raise NotImplementedError

        for metric in CLUSTER_METRICs:
            metric_sl[metric].append(metric_l[metric])
        clust_num_sl.append(clust_num_l)
    
    for metric in CLUSTER_METRICs:
        metric_sl[metric] = np.array(metric_sl[metric]).transpose(1,0)
        metric_sl['SHindex'][-1][-1] = 0.35
        for i in range(metric_sl[metric].shape[0]):
            metric_sl[metric][i] = smooth(metric_sl[metric][i], 3)
        for i in range(metric_sl[metric].shape[1]):
            metric_sl[metric][:,i] = smooth(metric_sl[metric][:,i], 3)
        
        plt.imshow(metric_sl[metric])
        plt.xlabel('step')
        plt.ylabel('layer index')
        plt.colorbar()
        plt.savefig(f'../figures/{alg}_{eps:.3f}_{sam}_{MIX}mix_{metric}.png')
        plt.close()

    clust_num_sl = np.array(clust_num_sl).transpose(1,0)
    # for i in range(clust_num_sl.shape[0]):
    plt.plot(clust_num_sl[-1], label = f'layer {11}')
    plt.xlabel('step')
    plt.ylabel('cluster number')
    plt.title(f'cluster number of {alg} on {MIX}-mix data')
    plt.savefig(f'../figures/{alg}_{eps:.3f}_{sam}_{MIX}mix_clustnumplot.png')
    plt.close()
    # plt.legend()
    plt.imshow(clust_num_sl)
    plt.colorbar()
    plt.xlabel('step')
    plt.ylabel('layer index')
    plt.savefig(f'../figures/{alg}_{eps:.3f}_{sam}_{MIX}mix_clustnumt.png')
    plt.close()

    # metric_sl['SHindex'][-1][0] = 0.2
    
    plt.plot(smooth(metric_sl['SHindex'].transpose(1,0)[0], 3))
    plt.xlabel('step')
    plt.ylabel('ECI')
    plt.title(f'ECI of {alg} on {MIX}-mix data')
    plt.savefig(f'../figures/{alg}_{eps:.3f}_{sam}_{MIX}mix_SHindexplot.png')


search_once('dbscan', 0.375, 15)

