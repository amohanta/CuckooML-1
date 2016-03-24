#! /usr/bin/python2

import os, sys
import logging
import argparse
import json
from sklearn.feature_extraction import DictVectorizer
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
from kmeans import KMeans
from flask import Flask
from utils import construct_W, MCFS

logger = logging.getLogger("CuckooML")
logger.setLevel(logging.WARN)

flask_app = Flask(__name__, template_folder='templates')

def check_perm(writable=True):
    class CheckDirPermission(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            prospective_dir = values

            if not os.path.isdir(prospective_dir):
                raise argparse.ArgumentTypeError(\
                    "check_dir_permission:{0} is not a \
                    valid path".format(prospective_dir))

            if writable is True:
                perm = os.W_OK
            else:
                perm = os.R_OK

            if os.access(prospective_dir, perm):
                setattr(namespace, self.dest, prospective_dir)
            else:
                raise argparse.ArgumentTypeError(\
                    "check_dir_permission:{0} is not a \
                    readable dir".format(prospective_dir))

    return CheckDirPermission

def is_valid_file(parser, arg):
    '''
        Check if File Path Exists and is readable.
    '''
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

def parse_args():
    '''
        Arguments Parser
    '''
    parser = argparse.ArgumentParser(description="CuckooML")

    parser.add_argument("-I", "--input-dir", action=check_perm(False), \
        required=True, metavar="input-dir", \
        help="Input Directory containing reports")
    parser.add_argument("-O", "--output-dir", action=check_perm(True), \
        required=False)
    parser.add_argument("-n", "--num_class", metavar="num_class", default=5, \
        type=int, help="Number of Class(es)")
    parser.add_argument("-f", "--num_features", metavar="num_features", \
        default=25, type=int, help="Number of Feature(s)")
    parser.add_argument("-v", "--verbosity", action="count", default=0)
    parser.add_argument('--version', action='version', version='%(prog)s 0.1')
    return parser.parse_args()

def feature_extraction(data, args):
    vectorizer = DictVectorizer(sparse=False)
    vectorized_data = vectorizer.fit_transform(data)

    kwargs = {"metric": "euclidean", "neighborMode": "knn", \
        "weightMode": "heatKernel", "k": 5, 't': 1}

    weights = construct_W.construct_W(vectorized_data, **kwargs)

    S = MCFS.mcfs(vectorized_data, args.num_features, \
        W=weights, n_clusters=args.num_class)

    idx = MCFS.feature_ranking(S)
    #idx = np.array(idx)

    #Feature Ranking: Extract important num_features features from data.
    new_data = vectorized_data[:, idx[0:args.num_features]]

    kmeans_obj = KMeans(new_data, args)
    kmeans_obj.cluster_data()
    kmeans_obj.plot_cluster()

def main(args):
    files = os.listdir(args.input_dir)
    features = ['info.score', 'info.duration']
    scanners = []

    data = []

    for file_name in files:

        file_path = os.path.join(args.input_dir, file_name)

        with open(file_path) as input_file:
            file_content = json.load(input_file)

            data.append({})

            for feature in features:
                feature_split = feature.split('.')
                temp = file_content
                for key in feature_split:
                    temp = temp[str(key)]
                data[-1][feature] = temp


            '''
            # Include the results of each scan.
            try:
                temp = file_content["virustotal"]["scans"]
                if not scanners:
                    for key in temp:
                        scanners.append(key)

                for key in scanners:
                    if temp.has_key(key):
                        if temp[key]["detected"] == True:
                            data[-1][key + ".result"] = temp[key]["result"]
                        else:
                            data[-1][key + ".result"] = "not_detection"
                    else:
                        data[-1][key + ".result"] = "unknown"

            except Exception, e:
                for key in scanners:
                    data[-1][key + ".result"] = "unknown"
            '''


            # Addition of the Normalized Virustotal result with their respective count value.
            try:
                norm_results = file_content["virustotal"]["scans"]
                
                for scan_result in norm_results:
                    if norm_results[scan_result]["detected"]==False:
                        continue
                    for result in norm_results[scan_result]["normalized"]:
                        if str("normalized=" + result) in data[-1]:
                            data[-1]["normalized=" + result] += 1
                        else:
                            data[-1]["normalized=" + result] = 1
            except Exception, e:
                pass


            #Add Signature Values in Virustotal Results
            try:
                signature_result = file_content["signatures"]

                for signature in signature_result:
                    if str("signature=" + signature["name"]) in data[-1]:
                        data[-1]["signature=" + signature["name"]] += 1
                    else:
                        data[-1]["signature=" + signature["name"]] = 1
            except Exception, e:
                pass


            #Add Behavioural ApiStats
            try:
                behavioural_info = file_content["behavior"]["apistats"]
                for err_no in behavioural_info:
                    for key in behavioural_info[err_no]:
                        if key not in data[-1]:
                            data[-1][key] = behavioural_info[err_no][key]
                        else:
                            data[-1][key] += behavioural_info[err_no][key]
            except Exception, e:
                pass


    feature_extraction(data, args)

if __name__ == '__main__':
    global arguments
    arguments = parse_args()
    main(arguments)
