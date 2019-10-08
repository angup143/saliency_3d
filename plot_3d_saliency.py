##https://github.com/pair-code/saliency/blob/master/Examples.ipynb

import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import importlib
from mayavi import mlab
import matplotlib.pyplot as plt
from docx import Document
from io import StringIO, BytesIO
import tracemalloc
import linecache


cwd = os.getcwd()
ROOT_DIR = os.path.join(cwd,'external', 'pointnet2')

sys.path.append(os.path.join(cwd,'external', 'saliency'))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import provider
import tf_util
import modelnet_dataset
import modelnet_h5_dataset
import saliency

model = 'pointnet2_cls_ssg'
BATCH_SIZE = 1
NUM_POINT = 1024
NUM_CLASSES = 40

MODEL = importlib.import_module(model)
DATA_PATH = os.path.join(ROOT_DIR, 'data/modelnet40_ply_hdf5_2048')
print(DATA_PATH)
TRAIN_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(DATA_PATH, 'train_files_local.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=True)
TEST_DATASET = modelnet_h5_dataset.ModelNetH5Dataset(os.path.join(DATA_PATH, 'test_files_local.txt'), batch_size=BATCH_SIZE, npoints=NUM_POINT, shuffle=False)

def display_top(snapshot, key_type='lineno', limit=10):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))

def test_data_loading():
    while TEST_DATASET.has_next_batch():
        print('here')

        print(TEST_DATASET._get_data_filename())
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        print(batch_data[0,:].shape)
        mlab.points3d((batch_data[0,:,0]),(batch_data[0,:,1]),(batch_data[0,:,2]))
        mlab.show()

def VisualisePcdGrad(pcd, grad, percentile=99, show=True, save=False, output_filename='x'): #assumes both are 1024x3
    grad_val = np.sum(np.abs(grad), axis=1)
    vmax = np.percentile(grad_val, percentile)
    vmin = np.min(grad_val)
    grad_val = np.clip((grad_val - vmin) / (vmax - vmin), 0, 1)
    grad_val = np.expand_dims(grad_val, axis=1)

    pc_grad = np.concatenate((pcd,grad_val),axis=1)

    mlab.points3d(pc_grad[:,0],pc_grad[:,1],pc_grad[:,2],pc_grad[:,3] )
    if show:
        mlab.show()
    if save:
        mlab.savefig(output_filename)
        mlab.clf()
        # mlab.close()
        return None, None
    else:
        img = mlab.screenshot()
        mlab.clf()
        # mlab.close()
        return pc_grad, img
    mlab.clf()
    return None, None

    
        # print(np.shape(pc_grad))

def plot_3d(points, show=False,save=True, output_filename='x'): # assumes points=x1024x3
    tracemalloc.start()
    if points.ndim==3:
        points = points[0,:]
    # fig = mlab.figure()
    
    mlab.points3d((points[:,0]),(points[:,1]),(points[:,2]))
    if show:
        mlab.show()
    if save:
        mlab.savefig(output_filename)

    # img = mlab.screenshot()
    mlab.clf()
    # mlab.close(all=True)

    return None


def rotate_point_cloud(pcd, angle):
    angle_rad = np.deg2rad(angle)
    cosval = np.cos(angle_rad)
    sinval = np.sin(angle_rad)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
    rotated_pcd = np.dot(pcd, rotation_matrix)
    return rotated_pcd



def main():
    # tracemalloc.start()
    # mlab.options.offscreen=True


    ckpt_file = '/home/ananya/Documents/titan/code/pointnet2/log/model.ckpt'
    output_folder =  '/home/ananya/Documents/titan/code/pointnet2/output/modelnet_features/'
    is_training = False
    visualise = False
    save = True
    graph = tf.Graph()
    # output_filename = '/home/ananya/Desktop/pointnet_fetures.docx'
    document = Document()
    rotation = [20,40,60,80,100,120,140,160,180]
    classes = {1: {'name':'bathtub','count':0},
               2:{'name':'bed','count':0}, 
               8:{'name':'chair','count':0},
               12:{'name':'desk','count':0},
               14:{'name':'dresser','count':0}, 
               22:{'name':'monitor','count':0},
               23:{'name':'night_stand','count':0},
               30:{'name':'sofa','count':0},
               33:{'name':'table','count':0},
               35:{'name':'toilet','count':0}}

    with graph.as_default():
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
        batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)

         # Restore the checkpoint
        sess = tf.Session(graph=graph)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)
        
        neuron_selector = tf.placeholder(tf.int32)
        y = pred[0][neuron_selector]

   
        while TEST_DATASET.has_next_batch():
            cur_batch_data, cur_batch_label = TEST_DATASET.next_batch(augment=False)
            #only test for certain classes
            label = cur_batch_label[0]
            if label in classes.keys():
                cur_output_folder = os.path.join(output_folder, classes[label]['name'])
                label_count = classes[label]['count']
                print('saving to ',cur_output_folder)
                if not os.path.exists(cur_output_folder):
                    os.makedirs(cur_output_folder)
                    os.makedirs(os.path.join(cur_output_folder, 'point_cloud'))
                    os.makedirs(os.path.join(cur_output_folder, 'vanilla_grad'))
                    os.makedirs(os.path.join(cur_output_folder, 'guided_backprop'))
                    os.makedirs(os.path.join(cur_output_folder, 'integrated_grad'))

                for rot in rotation:
                    cur_batch_data[0,:] = rotate_point_cloud(cur_batch_data[0,:], rot)

                    feed_dict = {pointclouds_pl: cur_batch_data,
                                labels_pl: cur_batch_label,
                                is_training_pl: is_training}
                    pred_val = sess.run(pred, feed_dict=feed_dict)
                    pred_class = np.argmax(pred_val, 1)[0]      

                    pcl = cur_batch_data[0,:]
                    output_filename = '{}/point_cloud/{}_{}.png'.format(cur_output_folder, label_count, rot)
                    base_img = plot_3d(pcl, show=visualise, save=save, output_filename=output_filename)

                    # # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
                    gradient_saliency = saliency.GradientSaliency(graph, sess, y, pointclouds_pl)
                    feed_dict = {neuron_selector: pred_class, is_training_pl: is_training}

                    # # # Compute the vanilla mask and the smoothed mask.
                    output_filename = '{}/vanilla_grad/{}_{}.png'.format(cur_output_folder, label_count, rot)
                    vanilla_gradients_3d = gradient_saliency.GetMask(pcl, 
                    feed_dict=feed_dict)
                    # pc_grad_van, img_van = VisualisePcdGrad(pcl, vanilla_gradients_3d,show=visualise, save=save, output_filename=output_filename)
                    # smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(pcl, feed_dict=feed_dict)
                    # VisualisePcdGrad(pcl, smoothgrad_mask_3d)


                    ##guided backprop
                    guided_backprop = saliency.GuidedBackprop(graph, sess, y, pointclouds_pl)
                    feed_dict = {neuron_selector: pred_class, is_training_pl: is_training}
                    output_filename = '{}/guided_backprop/{}_{}.png'.format(cur_output_folder, label_count, rot)
                    vanilla_gradients_3d = guided_backprop.GetMask(pcl, 
                    feed_dict=feed_dict )
                    # pc_grad_van_guided, img_van_guided = VisualisePcdGrad(pcl, vanilla_gradients_3d,show=visualise,save=save, output_filename=output_filename )

                    ##integrated grad

                    integrated_gradient = saliency.IntegratedGradients(graph, sess, y, pointclouds_pl)
                    output_filename = '{}/integrated_grad/{}_{}.png'.format(cur_output_folder, label_count, rot)
                    vanilla_gradients_3d = integrated_gradient.GetMask(pcl, 
                    feed_dict=feed_dict )
                    pc_grad_van_guided, img_van_guided = VisualisePcdGrad(pcl, vanilla_gradients_3d,show=visualise,save=save, output_filename=output_filename )

                    smoothgrad_mask_3d = guided_backprop.GetSmoothedMask(pcl, feed_dict=feed_dict)
                    pc_grad_smooth_guided, img_smooth_guided = VisualisePcdGrad(pcl, smoothgrad_mask_3d)

                    # fig = plt.figure()
                    # ax1 = fig.add_subplot(121)
                    # ax1.imshow(img_van_guided)
                    # ax1.set_axis_off()

                    # # add the screen capture
                    # ax2 = fig.add_subplot(122)
                    # ax2.imshow(base_img)
                    # ax2.set_axis_off()

                    # # plt.show()
                    # memfile = BytesIO()
                    # plt.savefig(memfile)
                    
                    # document.add_picture(memfile)
                    # document.save(output_filename)
                    # plt.close('all')
                    # memfile.close()
                classes[label]['count'] = label_count+1
                # snapshot = tracemalloc.take_snapshot()
                # display_top(snapshot)




if __name__ == '__main__':
    main()


