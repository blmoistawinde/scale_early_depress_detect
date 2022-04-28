import os
import re
import json
import numpy as np
from numpy.core.arrayprint import array2string
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans, Birch

def get_kmeans_centroid_ids(embs, K=32):
    # if len(embs) <= K:
    #     return list(range(len(embs)))
    kmeans = KMeans(K)
    cluster_dists = kmeans.fit_transform(embs)
    labels = kmeans.labels_
    ret = []
    for k in range(K):
        curr_members = np.where(labels == k)[0]
        if len(curr_members) == 0:
            continue
        local_id = cluster_dists[curr_members,k].argmin()
        global_id = curr_members[local_id]
        ret.append(global_id)
    return sorted(list(set(ret)))

def get_cluster_summary(user_posts, user_embs, K=32):
    # note that the posts and embs are already selected using mappings
    if len(user_posts) <= K:
        return user_posts
    centroid_ids = get_kmeans_centroid_ids(user_embs, K)
    return [user_posts[i] for i in centroid_ids]


if __name__ == "__main__":
    arr = np.random.randn(200, 100)
    print(get_kmeans_centroid_ids(arr))