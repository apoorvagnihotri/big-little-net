import os
import urllib.request as testfile

def downloader(uplink, down_fold, local_name):
    """Helpful to download files located at uplink

    eg
    uplink = '''https://archive.ics.uci.edu/ml/'''
             '''machine-learning-databases/iris/iris.data'''
    down_fold = "../data/iris/"
    local_name = "iris.csv"
    """
    local_file_path = down_fold + '/' + local_name
    if not os.path.exists(local_file_path):
        os.makedirs(down_fold)
        print ("Downloading...")
        testfile.urlretrieve(uplink, local_file_path)
        print ("Downloaded at", down_fold)
