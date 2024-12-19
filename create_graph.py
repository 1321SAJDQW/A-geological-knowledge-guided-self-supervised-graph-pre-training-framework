import libpysal
import pandas as pd
import pickle as pkl
import numpy as np

# Read data
inputfile = "HT_label.csv"
dataframe = pd.read_csv(inputfile)

# read coordinates
xy = dataframe[['POINT_X', 'POINT_Y']]
xy = xy.values
for distance in [750]:
# for distance in [500]:
    wid = libpysal.weights.distance.DistanceBand.from_array(xy, threshold=distance, p=2, binary=False)
    dict = wid.neighbor_offsets
    sparse = wid.sparse

    file = open("graph_"+str(distance)+".pkl", 'wb')
    r = pkl.dump(dict, file)
    file.close()

    print(distance)
