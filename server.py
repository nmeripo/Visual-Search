from flask import Flask, jsonify, request
from utils import timeit
import pickle
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import json

app = Flask(__name__)

# Paths to data
PATH_TO_IDS = './item_id.npy'
PATH_TO_VECTORS = './item_vector.npy'


@app.route('/visual_search', methods=['POST'])
def apicall():
    """
    API Call
    """
    try:
        query_json = request.get_json()
        item_id = int(query_json["item_id"])
        item_vec = ids_vecs[item_id]
        query_type = query_json["query_type"]

        if query_type == "similar":
            # Number of results
            K = 10
            # (K + 1) because ball tree also returns the query item in results
            dist, ind = ball_tree.query([item_vec], k = K + 1)
            ind = ind.flatten()
            # Retrieving item_ids from indices
            result = [item_ids[i] for i in ind[1:]]
            resp = {'item_id': str(item_id), 'similar': str(result)}


        elif query_type == "diverse":
            K = 10

            # Measuring distance from the query all (30) centroids
            temp_centroids = []
            for centre in k_centroids:
                temp_centroids.append(float(euclidean_distances([centre], [item_vec])[0]))

            # Pick K/2 closest centroids
            choosen_labels = np.argsort(temp_centroids)[: int(K / 2)]

            choosen_ids = []

            # Handling if request is made for odd number of results
            odd_check = (K % 2 == 1)

            for i in choosen_labels:
                # Randomly picks 2 results from K/2 centroids if K is even
                # Picks 3 results from the closest centroid if K is odd
                if odd_check:
                    random_sampling = np.random.choice(k_cluster_pts[i], 1)
                    choosen_ids.extend(random_sampling)
                    odd_check = False
                random_sampling = np.random.choice(k_cluster_pts[i], 2)
                choosen_ids.extend(random_sampling)

            resp = {'item_id': str(item_id), 'diverse': str(choosen_ids)}


    except Exception as e:
        raise e

    responses = jsonify(json.dumps(resp))
    responses.status_code = 200

    return responses




if __name__ == '__main__':
    try:
        # Load all the pretrained models and dataset
        item_ids = np.load(PATH_TO_IDS)
        item_vectors = np.load(PATH_TO_VECTORS)

        with open('./ids_vecs.pkl', 'rb') as f:
            ids_vecs = pickle.load(f)

        with open('./balltree_euclidean.pkl', 'rb') as f:
            ball_tree = pickle.load(f)

        with open('./k_centroids.pkl', 'rb') as f:
            k_centroids = pickle.load(f)

        with open('./k_cluster_pts.pkl', 'rb') as f:
            k_cluster_pts = pickle.load(f)

    except Exception as e:
        print(str(e))

    app.run(debug=True)
