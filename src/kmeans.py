#! /usr/bin/python2
"""
    KMeans Clustering Class
"""

from sklearn import cluster
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
from matplotlib import pyplot

class KMeans(object):
    """Apply KMeans Clustering """

    def __init__(self, data, arg):
        super(KMeans, self).__init__()
        self.no_clusters = arg.num_class
        self.data = np.array(data)
        self.cluster_labels = []
        self.centroids = []
        self.k_means = None

    def cluster_data(self):
        '''
            Run Clustering to calculate labels and centroids for the data.
        '''
        self.k_means = cluster.KMeans(n_clusters=self.no_clusters)
        self.k_means.fit(self.data)
        
        self.centroids = self.k_means.cluster_centers_
        self.cluster_labels = self.k_means.labels_

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

        if len(self.data[0])==2:
            # The following code was based in:
            # http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#example-cluster-plot-kmeans-digits-py

            gap = 1

            x_min, x_max = self.data[:, 0].min() - 0.1, self.data[:, 0].max() + 0.1
            y_min, y_max = self.data[:, 1].min() - 0.1, self.data[:, 1].max() + 0.1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, gap), np.arange(y_min, y_max, gap))

            Z = self.k_means.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            plt.figure(1)
            plt.clf()
            plt.imshow(Z, interpolation='nearest',
                   extent=(xx.min(), xx.max(), yy.min(), yy.max()),
                   cmap='jet',  
                   aspect='auto', origin='lower')

            plt.plot(self.data[:, 0], self.data[:, 1], 'w.', markersize=3)
            centroids = self.k_means.cluster_centers_

            plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', \
                color='w', s=180, linewidths=3, zorder=15)

            plt.title('K-means clustering (n_clusters={}, no_features={})'\
                .format(self.no_clusters, len(self.data[0])))
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)
            plt.xticks(())
            plt.yticks(())
            plt.show()

        else:
            pyplot.title("K-Means Clustering  (n_clusters={}, no_features={})"\
                .format(self.no_clusters, len(self.data[0])))

            for i in range(self.no_clusters):
                label_data = self.data[np.where(self.cluster_labels == i)]
                #print len(label_data)
                # plot the data observations
                pyplot.plot(label_data[:, 1], label_data[:, 2], 'o')
                # plot the centroids
                lines = pyplot.plot(self.centroids[i, 1], self.centroids[i, 2], \
                    'kx')
                # make the centroid x's bigger
                pyplot.setp(lines, ms=15.0)
                pyplot.setp(lines, mew=2.0)
            pyplot.show()
