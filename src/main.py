#! /usr/bin/python2

import os
import logging
import argparse
import json
from sklearn.feature_extraction import DictVectorizer
from sklearn import cluster
from matplotlib import pyplot
import numpy as np

logger = logging.getLogger("CuckooML")
logger.setLevel(logging.WARN)

class readable_dir(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        prospective_dir = values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError(\
                "readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace, self.dest, prospective_dir)
        else:
            raise argparse.ArgumentTypeError(\
                "readable_dir:{0} is not a readable dir".format(prospective_dir))

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

def parse_args():
    parser = argparse.ArgumentParser(description="CuckooML")

    parser.add_argument("-I", "--input-dir", action=readable_dir, \
        required=True, metavar="input-dir", \
        help="Input Directory containing reports")
    parser.add_argument("-n", "--num_class", metavar="num_class", default=5, \
        type=int, help="Number of Classes")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    return parser.parse_args()

def cluster_data(data, args):
    k_means = cluster.KMeans(n_clusters=args.num_class)
    k_means.fit(data)

    centroids = k_means.cluster_centers_
    labels = k_means.labels_

    print labels
    
    data = np.array(data)
    
    #plot_cluster(data, labels, centroids, args.num_class)

def plot_cluster(data, labels, centroids, no_clusters):
    #print len(data)

    for i in range(no_clusters):
        ds = data[np.where(labels==i)]
        #print len(ds)
        # plot the data observations
        pyplot.plot(ds[:,1],ds[:,2],'o')
        # plot the centroids
        lines = pyplot.plot(centroids[i,1],centroids[i,2],'kx')
        # make the centroid x's bigger
        pyplot.setp(lines, ms=15.0)
        pyplot.setp(lines, mew=2.0)
    pyplot.show()

def feature_extraction(data, args):
    vectorizer = DictVectorizer(sparse=False)
    vectorized_data = vectorizer.fit_transform(data)

    cluster_data(vectorized_data, args)

def main(args):
    files = os.listdir(args.input_dir)
    features = ['info.score', 'info.duration']
    scanners = []

    data = []

    for file_name in files:
        file_path = os.path.join(args.input_dir, file_name)
        
        #print file_name

        with open(file_path) as input_file:
            file_content = json.load(input_file)

            data.append({})

            for feature in features:
                feature_split = feature.split('.')
                temp = file_content
                for key in feature_split:
                    temp = temp[str(key)]
                data[-1][feature] = temp

            try:
                temp = file_content["virustotal"]["scans"]
                if not scanners:
                    for key in temp:
                        scanners.append(key)                    
                
                for key in scanners:
                    if temp.has_key(key):
                        if temp[key]["detected"]==True:
                            data[-1][key + ".result"] = temp[key]["result"]
                        else:
                            data[-1][key + ".result"] = "not_detection"
                    else:
                        data[-1][key + ".result"] = "unknown"

            except Exception, e:
                for key in scanners:
                    data[-1][key + ".result"] = "unknown"
                pass

    feature_extraction(data, args)

if __name__ == '__main__':
    arguments = parse_args()
    main(arguments)
