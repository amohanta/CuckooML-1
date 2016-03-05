#! /usr/bin/python

import sys
import os
import getopt
import logging
import argparse

log = logging.getLogger("CuckooML")
log.setLevel(logging.WARN)

class readable_dir(argparse.Action):
    def __call__(self,parser, namespace, values, option_string=None):
        prospective_dir=values
        if not os.path.isdir(prospective_dir):
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a valid path".format(prospective_dir))
        if os.access(prospective_dir, os.R_OK):
            setattr(namespace,self.dest,prospective_dir)
        else:
            raise argparse.ArgumentTypeError("readable_dir:{0} is not a readable dir".format(prospective_dir))

def is_valid_file(parser, arg):
    if not os.path.exists(arg):
        parser.error("The file %s does not exist!" % arg)
    else:
        return open(arg, 'r')  # return an open file handle

def parse_args():
	parser = argparse.ArgumentParser(description="CuckooML")
	
	parser.add_argument("-I", "--input-dir", action=readable_dir, required=True, \
		metavar="input-dir", help="Input Directory containing reports")
	parser.add_argument("-v", "--verbosity", action="count", default=0)
	args = parser.parse_args()

def main():
	print ''

if __name__ == '__main__':
	parse_args()
	main()