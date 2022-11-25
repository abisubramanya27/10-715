import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from itertools import cycle, islice

def pts_distance(x, y, mode='euclidean'):
    if mode.lower() == 'euclidean':
        return np.sqrt(np.sum((x-y)**2))
    elif mode.lower() == 'cosine':
        x_mod = np.linalg.norm(x)
        y_mod = np.linalg.norm(y)
        return 1 - np.dot(x/x_mod, y/y_mod)
    else: # 'manhattan' or 'cityblock'
        return np.sum(np.abs(x-y))

def cluster_distance(X, Y, linkage='single', dist=None, dist_betw_center=None):
    # If dist is not None, we use indices from X and Y to get the distance from dist matrix
    # dist_betw_center is needed for using Ward linkage
    if linkage.lower() in ['single', 'min']:
        return np.min([pts_distance(x, y) if dist is None else dist[x][y] for x in X for y in Y])
    elif linkage.lower() in ['max', 'complete']:
        return np.max([pts_distance(x, y) if dist is None else dist[x][y] for x in X for y in Y])
    elif linkage.lower() == 'ward':
        return (len(X)*len(Y)/(len(X)+len(Y))) * dist_betw_center
    else: # 'average'
        return np.mean([pts_distance(x, y) if dist is None else dist[x][y] for x in X for y in Y])
        
        
class LinkageClustering:
    def __init__(self, n_clusters, cluster_dist_linkage='single', pts_dist_mode='euclidean'):
        # assert cluster_dist_linkage != 'ward' or pts_dist_mode == 'euclidean'
        self.n_clusters = n_clusters
        self.cluster_dist_linkage = cluster_dist_linkage
        self.pts_dist_mode = pts_dist_mode
            
    def fit(self, X):
        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        else:
            self.X = X
        
        self.fill_pts_distances()
            
        self.clusters = {
            id: [id] for id in range(len(X))
        }
        
        self.distances_ = []
        self.children_ = []
        
        self.cluster_distances = {}
        for id_X, clusterX in self.clusters.items():
            self.cluster_distances[id_X] = {}
            for id_Y, clusterY in self.clusters.items():
                if id_X > id_Y:
                    self.cluster_distances[id_X][id_Y] = self.cluster_distances[id_Y][id_X]
                    continue
                elif id_X == id_Y:
                    continue
                
                dist_betw_center = None
                if self.cluster_dist_linkage == 'ward':
                    m_X = np.mean([self.X[i] for i in self.clusters[id_X]], axis=0)
                    m_Y = np.mean([self.X[i] for i in self.clusters[id_Y]], axis=0)
                    dist_betw_center = pts_distance(m_X, m_Y, self.pts_dist_mode)
                self.cluster_distances[id_X][id_Y] = cluster_distance(
                    clusterX, clusterY, self.cluster_dist_linkage, self.pts_distances, dist_betw_center
                )
        
        while len(self.clusters) > self.n_clusters:
            id_X, id_Y = self.find_closest_clusters()
            self.merge_clusters(id_X, id_Y)
             
        self.re_index_clusters()
        self.fill_labels_from_clusters()
    
    def fill_pts_distances(self):
        self.pts_distances = [
            [pts_distance(self.X[i], self.X[j], self.pts_dist_mode) for j in range(len(self.X))] 
            for i in range(len(self.X))
        ]
    
    def find_closest_clusters(self):
        best_cluster_pair_idx = (-1, -1)
        best_distance = np.inf
        for i in self.clusters:
            for j in self.clusters:
                if i >= j:
                    continue
                if self.cluster_distances[i][j] < best_distance:
                    best_distance = self.cluster_distances[i][j]
                    best_cluster_pair_idx = (i, j)
        
        self.distances_.append(best_distance)
        self.children_.append(list(best_cluster_pair_idx))
        
        return best_cluster_pair_idx
    
    def merge_clusters(self, id_X, id_Y):
        for ind in self.clusters:
            if ind == id_X or ind == id_Y:
                continue
                        
            if self.cluster_dist_linkage in ['single', 'min']:
                self.cluster_distances[id_X][ind] = self.cluster_distances[ind][id_X] = min(
                    self.cluster_distances[id_X][ind], self.cluster_distances[id_Y][ind]
                )
            elif self.cluster_dist_linkage in ['max', 'complete']:
                self.cluster_distances[id_X][ind] = self.cluster_distances[ind][id_X] = max(
                    self.cluster_distances[id_X][ind], self.cluster_distances[id_Y][ind]
                )
            elif self.cluster_dist_linkage == 'ward':
                m_X = [self.X[i] for i in self.clusters[id_X]]
                m_X.extend([self.X[i] for i in self.clusters[id_Y]])
                m_X = np.mean(m_X, axis=0)
                m_Y = np.mean([self.X[i] for i in self.clusters[ind]], axis=0)
                dist_betw_center = pts_distance(m_X, m_Y, self.pts_dist_mode)
                n1 = len(self.clusters[id_X]) + len(self.clusters[id_Y])
                n2 = len(self.clusters[ind])
                self.cluster_distances[id_X][ind] = self.cluster_distances[ind][id_X] = (n1*n2/(n1+n2)) *\
                    dist_betw_center
            else: # Average Linkage
                nr1 = len(self.clusters[id_X])*len(self.clusters[ind])
                nr2 = len(self.clusters[id_Y])*len(self.clusters[ind])
                dr = nr1+nr2
                self.cluster_distances[id_X][ind] = self.cluster_distances[ind][id_X] = \
                    (nr1/dr)*self.cluster_distances[id_X][ind] + (nr2/dr)*self.cluster_distances[id_Y][ind]      

            self.cluster_distances[ind].pop(id_Y, None)
        
        self.cluster_distances[id_X].pop(id_Y, None)    
        self.cluster_distances.pop(id_Y, None)
        self.clusters[id_X].extend(self.clusters[id_Y])
        self.clusters.pop(id_Y, None)
        
    
    def re_index_clusters(self):
        mapper = {
            old_key: new_key for new_key, old_key in enumerate(self.clusters)
        }
        self.clusters = {
            mapper[old_key]: self.clusters[old_key] for old_key in self.clusters
        }
        self.cluster_distances = {
            mapper[id_X]: { 
                mapper[id_Y]: dist for id_Y, dist in distances.items()
            } for id_X, distances in self.cluster_distances.items()
        }
    
    def fill_labels_from_clusters(self):
        self.cluster_labels = [-1 for _ in range(len(self.X))]
        for cluster_id in self.clusters:
            for X_idx in self.clusters[cluster_id]:
                self.cluster_labels[X_idx] = cluster_id
    

def scatter_plot(X, y=None, fig_idx=0, merge_outliers=False, tsne=False):
    if y is None:
        y = [0 for _ in range(len(X))]
    if merge_outliers:
        y = [0 if yi == 0 else 1 for yi in y]
    
    pca = PCA(n_components=2) if not tsne else TSNE(n_components=2)
    X_red = pca.fit_transform(X)
    
    plt.figure(fig_idx)
    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "olive",
                        "orange",
                        "blue",
                        "red",
                        "brown",
                        "mediumseagreen",
                        "#377eb8",
                        "pink",
                        "m",
                        "black"
                    ]
                ),
                int(max(y) + 1),
            )
        )
    )
    plt.scatter(X_red[:, 0], X_red[:, 1], s=12, color=colors[y])
    plt.xlabel('First Dimension')
    plt.ylabel('Second Dimension')
    plt.title('Visualization of datapoints (dimension reduced using PCA)')
    plt.show()
    plt.close()

def pca_plot(variances, threshold=0.95, fig_idx=0):
    plt.figure(fig_idx)
    npc = np.argmax(variances >= threshold)+1
    ax = plt.subplot(1, 1, 1)
    ax.set_xlabel('number of components (log scale)')
    ax.set_ylabel('cumulative % explained variance')
    ax.set_title('PCA')
    ax.semilogx(list(range(1,variances.shape[0]+1)), variances*100)
    ax.axhline(variances[npc-1]*100, c='red', linestyle='dashed', label=f'cum-var {variances[npc-1]*100:.2f}% @ {npc} PC')
    plt.legend()
    plt.show()
    plt.close()


EPS = 1e-8
if __name__ == "__main__":
    train = pd.read_csv('Q2_data/train.csv', header=None)
    print(f'Total number of datapoints: {len(train)}')
    # Removing columns that don't have any variations (since they dont contribute to distinguishing datapoints)
    columns = list(train.columns)
    columns_to_drop = [column for column in columns if train[column].max()-train[column].min() < EPS]
    train.drop(columns_to_drop, axis=1, inplace=True)
    print(f'{len(columns_to_drop)} out of {len(columns)} columns dropped due to their zero contribution to distinguishing datapoints')
    min_val = math.inf
    max_val = -math.inf
    for column in train.columns:
        min_val = min(min_val, train[column].min())
        max_val = max(max_val, train[column].max())
    print(f'Min of column values: {min_val} | Max of column values: {max_val}')
    
    # Stats about vector magnitudes
    X = train.to_numpy()
    mag = np.linalg.norm(X, axis=1)
    print(f'Min of point vector magnitudes: {np.min(mag):.3f}')
    print(f'Max of point vector magnitudes: {np.max(mag):.3f}')
    print(f'Mean of point vector magnitudes: {np.mean(mag):.3f}')
    print(f'Variance of point vector magnitudes: {np.var(mag):.3f}')
    
    # To plot PCA
    dim_redn = PCA()
    X = dim_redn.fit_transform(train)
    expl_var = np.cumsum(dim_redn.explained_variance_ratio_)
    pca_plot(expl_var, fig_idx=0)

    # To plot true labels
    true_labels = pd.read_csv('Q2_data/labels.csv', header=None)
    true_labels[0] = true_labels[0].astype(int)
    scatter_plot(train, true_labels[0], fig_idx=1)

    # Scatter Plot of Clustering Results
    lc = LinkageClustering(2, 'complete', 'cosine')
    lc.fit(train)
    scatter_plot(train, lc.cluster_labels, fig_idx=2)
    