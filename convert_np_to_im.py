import numpy as np
import pickle
import os
import sys
import re
import importlib
from mayavi import mlab
import matplotlib.pyplot as plt
import argparse
import glob

def save_img_from_arr(points, outfile):
    print('Saving {}'.format(outfile))
    if points.shape[1]==3:
        mlab.points3d((points[:,0]),(points[:,1]),(points[:,2]))
    else:
        mlab.points3d((points[:,0]),(points[:,1]),(points[:,2]),(points[:,3])  )
    mlab.savefig(outfile)
    mlab.clf()
    return

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filepath', type=str, default='/home/ananya/Documents/titan/code/pointnet2/output/modelnet40_features/plant')
    parser.add_argument('--ignore_previous', action='store_true')
    args = parser.parse_args()

    np_files = glob.glob(os.path.join(args.filepath,'**/*.npy'))

    for np_file in np_files:
        outfile = re.sub('.npy', '.png', np_file)
        if os.path.isfile(outfile) and not args.ignore_previous:
            print('{} already exists'.format(outfile))
            continue
        points = np.load(np_file)
        save_img_from_arr(points, outfile)

        




if __name__ == '__main__':
    main()