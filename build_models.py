import numpy as np
from sklearn.neighbors import LSHForest, BallTree
from sklearn.metrics.pairwise import euclidean_distances
from utils import timeit, elbow_plot
from sklearn.cluster import MiniBatchKMeans
import pickle


# Paths to data
PATH_TO_IDS = './item_id.npy'
PATH_TO_VECTORS = './item_vector.npy'


#lambdafn = lambda x : ', '.join(map(str, x))


def load_dataset(path_to_ids, path_to_vectors):
    """

    :param path_to_ids: path to item_id.npy
    :param path_to_vectors: path to item_vector.npy
    :return: numpy arrays item_id, item_vector, and dictionary ids_vecs with item_ids as keys and item_vectors as values
    """
    item_ids = np.load(path_to_ids)
    item_vectors = np.load(path_to_vectors)
    ids_vecs = dict(zip(item_ids, item_vectors))

    return item_ids, item_vectors, ids_vecs


item_ids, item_vecs, ids_vecs = load_dataset(PATH_TO_IDS, PATH_TO_VECTORS)
pickle.dump(ids_vecs, open("ids_vecs.pkl", "wb"))



@timeit
def buildkNNModel(data):
    """
    Pickles the ball Tree model constructed with the given data
    :param data: takes input data
    :return: ball Tree model constructed with the given data
    """
    tree = BallTree(data, metric='euclidean')
    pickle.dump(tree, open("balltree_euclidean.pkl", "wb"))
    #nbrs = LSHForest(n_estimators=10, radius=1.0, n_candidates=50, n_neighbors=10, random_state=333)
    return tree


@timeit
def buildKMeans(data):
    """
    Pickles the  k_means model, k centroid vectors, dictionary of centroid labels and correspoinding item_ids
    :param data: takes input data
    :return: k_means model, k centroid vectors, dictionary of centroid labels and correspoinding item_ids
    """
    k_means = MiniBatchKMeans(n_clusters=30, init='k-means++', max_no_improvement=100, random_state=0, batch_size=10000, verbose=True)
    k_means = k_means.fit(data)
    k_centroids, labels = k_means.cluster_centers_, k_means.labels_
    k_cluster_pts = {i: np.where(k_means.labels_ == i)[0] for i in range(k_means.n_clusters)}

    pickle.dump(k_means, open("k_means.pkl", "wb"))
    pickle.dump(k_centroids, open("k_centroids.pkl", "wb"))
    pickle.dump(k_cluster_pts, open("k_cluster_pts.pkl", "wb"))

    return  k_means, k_centroids, k_cluster_pts



@timeit
def findKSimilarImages(query, tree, item_ids, K):
    """
     Returns K similar images
    :param query: query item's vector
    :param tree: constructed ball tree model
    :param item_ids: given item_ids data
    :param K: # of similar results
    :return: K similar results
    """
    dist, ind = tree.query([query], k = K + 1)
    ind = ind.flatten()
    result = [item_ids[i] for i in ind[1:]]
    return result


@timeit
def findKSimilarDiverseImages(query, centroids, cluster_pts, K):
    """
     Picks 2 (3 for the closest centroid if K is odd) random results from each centroid and returns K similar (and diverse) images
    :param query: query item's vector
    :param centroids: k centroid vectors
    :param cluster_pts: dictionary of centroid labels and correspoinding item_ids
    :param K: # of similar (and diverse) results
    :return: K similar (and diverse) results
    """
    temp_centroids = []
    for centre in centroids:
        temp_centroids.append(float(euclidean_distances([centre], [query])[0]))

    # Pick K/2 closest centroids
    choosen_labels = np.argsort(temp_centroids)[: int(K / 2)]

    choosen_ids = []

    # Handling if request is made for odd number of results
    odd_check = (K % 2 == 1)
    for i in choosen_labels:
        # Randomly picks 2 results from K/2 centroids if K is even
        # Picks 3 results from the closest centroid if K is odd
        if odd_check:
            random_sampling = np.random.choice(cluster_pts[i], 1)
            choosen_ids.extend(random_sampling)
            odd_check = False
        random_sampling = np.random.choice(cluster_pts[i], 2)
        choosen_ids.extend(random_sampling)

    return choosen_ids



buildkNNModel(item_vecs)

# Elobow method to pick k-value in k-means clustering
# elbow_plot(item_vectors, step=3, maxK=90)

buildKMeans(item_vecs)