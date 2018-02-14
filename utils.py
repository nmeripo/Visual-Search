import functools
import time
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans


def elbow_plot(data, maxK=80, step=2):
    """
    Returns Elbow Method Plot (Number of clusters K vs Sum of Squares Errors SSE)
    :param data: input data
    :param maxK: maximum number of clusters
    :param step: k value in k means increments with step size
    """
    sse = {}
    for k in range(1, maxK):
        if k % step == 0:
            print("k: ", k)
            kmeans = MiniBatchKMeans(n_clusters=k, init='k-means++', max_no_improvement=10, random_state=0).fit(data)
            #data["clusters"] = kmeans.labels_
            # Inertia: Sum of distances of samples to their closest cluster center
            sse[k / step] = kmeans.inertia_
    actual_k = list(sse.keys())
    actual_k[:] = [x * step for x in actual_k]
    fig = plt.figure()
    plt.plot(actual_k, list(sse.values()))
    fig.suptitle('Elbow method', fontsize=22)
    plt.xlabel('Number of Clusters (k)', fontsize=18)
    plt.ylabel('Sum of squared Errors (SSE)', fontsize=18)
    plt.show()
    return

def timeit(func):
    """
    Prints time taken for a function call
    :param func: function call
    :return: Computes the time taken for each function call
    """
    @functools.wraps(func)
    def newfunc(*args, **kwargs):
        startTime = time.time()
        func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
    return newfunc


