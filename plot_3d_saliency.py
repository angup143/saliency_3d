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

def test_data_loading():
    while TEST_DATASET.has_next_batch():
        print('here')

        print(TEST_DATASET._get_data_filename())
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        print(batch_data[0,:].shape)
        mlab.points3d((batch_data[0,:,0]),(batch_data[0,:,1]),(batch_data[0,:,2]))
        mlab.show()

def VisualisePcdGrad(pcd, grad, percentile=99, show=True): #assumes both are 1024x3
    grad_val = np.sum(np.abs(grad), axis=1)
    vmax = np.percentile(grad_val, percentile)
    vmin = np.min(grad_val)
    grad_val = np.clip((grad_val - vmin) / (vmax - vmin), 0, 1)
    grad_val = np.expand_dims(grad_val, axis=1)

    pc_grad = np.concatenate((pcd,grad_val),axis=1)

    mlab.points3d(pc_grad[:,0],pc_grad[:,1],pc_grad[:,2],pc_grad[:,3] )
    if show:
        mlab.show()
    # else:
    img = mlab.screenshot()
    mlab.close()
        # print(np.shape(pc_grad))
    return pc_grad, img

def plot_3d(points, show=True): # assumes points=x1024x3
    if points.ndim==3:
        points = points[0,:]
    
    mlab.points3d((points[:,0]),(points[:,1]),(points[:,2]))
    if show:
        mlab.show()
    # else:
    img = mlab.screenshot()
    mlab.close()
    return img


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

    ckpt_file = '/home/ananya/Documents/titan/code/pointnet2/log/model.ckpt'
    is_training = False
    graph = tf.Graph()
    output_filename = '/home/ananya/Desktop/pointnet_fetures.docx'
    document = Document()
    rotation = [20,40,60,80,100,120,140,160,180]

    with graph.as_default():
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        is_training_pl = tf.placeholder(tf.bool, shape=())
         # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter
            # for you every time it trains.
        batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)

         # Restore the checkpoint
        sess = tf.Session(graph=graph)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)
        
       
        neuron_selector = tf.placeholder(tf.int32)
        y = pred[0][neuron_selector]


    # for op in graph.get_operations():
    #     print(op.name)
    # names = [n.name for n in graph.as_graph_def().node]
    # print(names)

    # Construct the scalar neuron tensor.
   
        while TEST_DATASET.has_next_batch():
            cur_batch_data, cur_batch_label = TEST_DATASET.next_batch(augment=False)
            for rot in rotation:
                cur_batch_data[0,:] = rotate_point_cloud(cur_batch_data[0,:], rot)

                feed_dict = {pointclouds_pl: cur_batch_data,
                            labels_pl: cur_batch_label,
                            is_training_pl: is_training}
                pred_val = sess.run(pred, feed_dict=feed_dict)
                pred_class = np.argmax(pred_val, 1)[0]      

                pcl = cur_batch_data[0,:]
                base_img = plot_3d(pcl)


                # # Construct the saliency object. This doesn't yet compute the saliency mask, it just sets up the necessary ops.
                # gradient_saliency = saliency.GradientSaliency(graph, sess, y, pointclouds_pl)
                # feed_dict = {neuron_selector: pred_class, is_training_pl: is_training}

                # # # Compute the vanilla mask and the smoothed mask.
                # vanilla_gradients_3d = gradient_saliency.GetMask(pcl, 
                # feed_dict=feed_dict )
                # VisualisePcdGrad(pcl, vanilla_gradients_3d)

                # smoothgrad_mask_3d = gradient_saliency.GetSmoothedMask(pcl, feed_dict=feed_dict)
                # VisualisePcdGrad(pcl, smoothgrad_mask_3d)


                ##guided backprop

                guided_backprop = saliency.GuidedBackprop(graph, sess, y, pointclouds_pl)
                feed_dict = {neuron_selector: pred_class, is_training_pl: is_training}

                vanilla_gradients_3d = guided_backprop.GetMask(pcl, 
                feed_dict=feed_dict )
                pc_grad_van_guided, img_van_guided = VisualisePcdGrad(pcl, vanilla_gradients_3d)

                # smoothgrad_mask_3d = guided_backprop.GetSmoothedMask(pcl, feed_dict=feed_dict)
                # pc_grad_smooth_guided, img_smooth_guided = VisualisePcdGrad(pcl, smoothgrad_mask_3d)

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




if __name__ == '__main__':
    main()


