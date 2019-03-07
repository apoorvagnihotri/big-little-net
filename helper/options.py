import os
import argparse

home_dir = '../'

def read_args():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_loc", default=os.path.join(home_dir,'data'),
                        help="Location from where to load the data")
    args = parser.parse_args()
    return args