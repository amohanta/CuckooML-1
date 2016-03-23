#! /usr/bin/python2
"""
    KMeans Clustering Class
"""

from sklearn import cluster
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from matplotlib import pyplot

class KMeans(object):
    """Apply KMeans Clustering """

    def __init__(self, data, arg):
        super(KMeans, self).__init__()
        self.no_clusters = arg.num_class
        self.data = np.array(data)
        self.cluster_labels = []
        self.centroids = []

    def cluster_data(self):
        '''
            Run Clustering to calculate labels and centroids for the data.
        '''
        k_means = cluster.KMeans(n_clusters=self.no_clusters)
        k_means.fit(self.data)

        print self.data
        
        self.centroids = k_means.cluster_centers_
        self.cluster_labels = k_means.labels_

        print self.cluster_labels

        silhouette_avg = silhouette_score(self.data, self.cluster_labels)
        print silhouette_avg

        #print normalized_mutual_info_score

    def find_best_k(self):
        '''
            Find the best fit value for num_cluster for KMeans Clustering.
        '''
        print ""

    def plot_cluster(self):
        '''
            Plot the Cluster.
        '''

        for i in range(self.no_clusters):
            label_data = self.data[np.where(self.cluster_labels == i)]
            #print len(label_data)
            # plot the data observations
            pyplot.plot(label_data[:, 2], label_data[:, 3], 'o')
            # plot the centroids
            lines = pyplot.plot(self.centroids[i, 2], self.centroids[i, 3], \
                'kx')
            # make the centroid x's bigger
            pyplot.setp(lines, ms=15.0)
            pyplot.setp(lines, mew=2.0)
        pyplot.show()
