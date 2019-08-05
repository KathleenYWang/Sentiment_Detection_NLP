import argparse
import pickle
import pprint

parser = argparse.ArgumentParser(description='Decode a Pickle file',
                                     prog='decode_pickle.py',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument(
        "-f",
        "--file",
        dest='file',
        required=True,
        type=str,
        help='Input pickle file')

args = parser.parse_args()

with open(args.file, 'rb') as pickle_file:
    obj = pickle.load(pickle_file)
    pprint.pprint(obj[20000:20020])
