##https://github.com/pair-code/saliency/blob/master/Examples.ipynb

import tensorflow as tf
import numpy as np
import pickle
import os
import sys
import re
import importlib
# from mayavi import mlab
import matplotlib.pyplot as plt
# from docx import Document
# from io import StringIO, BytesIO
# import tracemalloc
# import linecache


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
        print('Plotting frist sample of {} samples'. format(batch_data.shape[0]))

        print(TEST_DATASET._get_data_filename())
        batch_data, batch_label = TEST_DATASET.next_batch(augment=False)
        print(batch_data[0,:].shape)
        mlab.points3d((batch_data[0,:,0]),(batch_data[0,:,1]),(batch_data[0,:,2]))
        mlab.show()

def VisualisePcdGrad(pcd, grad, percentile=99, show=False, save=True, save_image=False, output_filename='x'): #assumes both are 1024x3
    grad_val = np.sum(np.abs(grad), axis=1)
    vmax = np.percentile(grad_val, percentile)
    vmin = np.min(grad_val)
    grad_val = np.clip((grad_val - vmin) / (vmax - vmin), 0, 1)
    grad_val = np.expand_dims(grad_val, axis=1)
    pc_grad = np.concatenate((pcd,grad_val),axis=1)
    
    if show:
        mlab.points3d(pc_grad[:,0],pc_grad[:,1],pc_grad[:,2],pc_grad[:,3] )
        mlab.show()
        mlab.clf()
    if save:
        output_npfile = re.sub('.jpg','.npy', output_filename)
        print('saving to {}'.format(output_npfile))
        np.save(output_npfile, pc_grad)
        return pc_grad, None
    if save_image:
        mlab.points3d(pc_grad[:,0],pc_grad[:,1],pc_grad[:,2],pc_grad[:,3] )
        mlab.savefig(output_filename)
        mlab.clf()
        return pc_grad, None
    else:
        mlab.points3d(pc_grad[:,0],pc_grad[:,1],pc_grad[:,2],pc_grad[:,3] )
        img = mlab.screenshot()
        mlab.clf()
        return pc_grad, img
    return None, None

#plot single point cloud, assumes batch but no grad
def plot_3d(points, show=False,save=True,save_image=False, output_filename='x'): # assumes points=x1024x3
    # tracemalloc.start()
    if points.ndim==3:
        points = points[0,:]
    
    if show:
        mlab.points3d((points[:,0]),(points[:,1]),(points[:,2]))
        mlab.show()
        mlab.clf()
    if save:
        output_npfile = re.sub('.jpg','.npy', output_filename)
        print('saving to {}'.format(output_npfile))
        np.save(output_npfile, points)
        return None
    if save_image:
        mlab.points3d((points[:,0]),(points[:,1]),(points[:,2]))
        mlab.savefig(output_filename)
        mlab.clf()
    
    return None


# def rotate_batch_point_cloud(pcd, angle): #shape = batch*1024*3

#     angle_rad = np.deg2rad(angle)
#     cosval = np.cos(angle_rad)
#     sinval = np.sin(angle_rad)
#     rotation_matrix = np.array([[cosval, sinval, 0],
#                                     [-sinval, cosval, 0],
#                                     [0, 0, 1]])
#     rotated_pcd = np.zeros(pcd.shape)
#     for n in range(pcd.shape[0]):
#         rotated_pcd[n,...] = np.dot(pcd[n,...], rotation_matrix)
#     return rotated_pcd



def main():
    # tracemalloc.start()
    # mlab.options.offscreen=True


    ckpt_file = '/home/ananya/Documents/titan/code/pointnet2/log/model.ckpt'
    output_folder =  '/home/ananya/Documents/titan/code/pointnet2/output/modelnet40_incorrect_features/'
    is_training = False
    visualise = False
    save = True
    save_image = False
    graph = tf.Graph()
    # output_filename = '/home/ananya/Desktop/pointnet_fetures.docx'
    # document = Document()
    # rotation = [20,40,60,80,100,120,140,160,180]
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

    modelnet40_classes = {0: {'name':'airplane','count':0},
               1: {'name':'bathtub','count':0},
               2:{'name':'bed','count':0}, 
               3:{'name':'bench','count':0}, 
               4:{'name':'bookshelf','count':0}, 
               5:{'name':'bottle','count':0}, 
               6:{'name':'bowl','count':0}, 
               7:{'name':'car','count':0}, 
               8:{'name':'chair','count':0},
               9:{'name':'cone','count':0}, 
               10:{'name':'cup','count':0}, 
               11:{'name':'curtain','count':0}, 
               12:{'name':'desk','count':0},
               13:{'name':'door','count':0},
               14:{'name':'dresser','count':0}, 
               15:{'name':'flower_pot','count':0},
               16:{'name':'glass_box','count':0},
               17:{'name':'guitar','count':0},
               18:{'name':'keyboard','count':0},
               19:{'name':'lamp','count':0},
               20:{'name':'laptop','count':0},
               21:{'name':'mantel','count':0},
               22:{'name':'monitor','count':0},
               23:{'name':'night_stand','count':0},
               24:{'name':'person','count':0},
               25:{'name':'piano','count':0},
               26:{'name':'plant','count':0},
               27:{'name':'radio','count':0},
               28:{'name':'range_hood','count':0},
               29:{'name':'sink','count':0},
               30:{'name':'sofa','count':0},
               31:{'name':'stairs','count':0},
               32:{'name':'stool','count':0},
               33:{'name':'table','count':0},
               33:{'name':'tent','count':0},
               35:{'name':'toilet','count':0},
               36:{'name':'tv_stand','count':0},
               37:{'name':'vase','count':0},
               38:{'name':'wardrobe','count':0},
               39:{'name':'xbox','count':0}}

    with graph.as_default():
        pointclouds_pl, labels_pl = MODEL.placeholder_inputs(None, NUM_POINT)
        # single_pointcloud_pl, single_label_pl =  MODEL.placeholder_inputs(1, NUM_POINT)

        is_training_pl = tf.placeholder(tf.bool, shape=())
        batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
        pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl)
        # single_pred, single_end_points = MODEL.get_model(single_pointcloud_pl, is_training_pl)

         # Restore the checkpoint
        sess = tf.Session(graph=graph)
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_file)
        
        neuron_selector = tf.placeholder(tf.int32)
        input_selector = tf.placeholder(tf.int32)

        
   
        while TEST_DATASET.has_next_batch():
            cur_batch_data, cur_batch_label = TEST_DATASET.next_batch(augment=False)
            batch_size = cur_batch_data.shape[0]
            #only test for certain modelnet40_classes

            feed_dict = {pointclouds_pl: cur_batch_data,
                        labels_pl: cur_batch_label,
                        is_training_pl: is_training}
            pred_val = sess.run(pred, feed_dict=feed_dict)
            pred_class = np.argmax(pred_val, 1)
            correct_sum = np.sum(pred_class[0:batch_size] == cur_batch_label[0:batch_size])
            if correct_sum == batch_size: #if all predictions are correct
                print('all predictions are correct, not computing grads')
                continue
            incorrect_idxs = np.not_equal(pred_class, cur_batch_label)
            incorrect_preds = pred_class[incorrect_idxs]
            incorrect_idx_vals = np.argwhere(incorrect_idxs).astype(np.int32)
            # incorrect_pred_vals = pred[incorrect_idxs,...]
            incorrect_labels = cur_batch_label[incorrect_idxs]
            incorrect_pcls = cur_batch_data[incorrect_idxs,...]
            for i, in_pcl,  in enumerate(incorrect_pcls):
                in_pred, in_label = incorrect_preds[i], incorrect_labels[i]
                y = pred[incorrect_idx_vals[i][0]][neuron_selector]
                gradient_saliency = saliency.GradientSaliency(graph, sess, y, pointclouds_pl)
                guided_backprop = saliency.GuidedBackprop(graph, sess, y,pointclouds_pl)
                integrated_gradient = saliency.IntegratedGradients(graph, sess, y, pointclouds_pl)

                cur_output_folder = os.path.join(output_folder, modelnet40_classes[in_label]['name']) 
                label_count = modelnet40_classes[in_label]['count']   
                if not os.path.exists(cur_output_folder):
                    os.makedirs(cur_output_folder)
                    os.makedirs(os.path.join(cur_output_folder, 'point_cloud'))
                    os.makedirs(os.path.join(cur_output_folder, 'vanilla_grad'))
                    os.makedirs(os.path.join(cur_output_folder, 'guided_backprop'))
                    os.makedirs(os.path.join(cur_output_folder, 'integrated_grad'))

                feed_dict = {neuron_selector: in_pred, is_training_pl: is_training}
                
                output_filename = '{}/point_cloud/{}_{}.png'.format(cur_output_folder, label_count, modelnet40_classes[in_pred]['name'])
                base_img = plot_3d(in_pcl, show=visualise, save=save, save_image=save_image, output_filename=output_filename)

                # # # Compute the vanilla mask and the smoothed mask.
                output_filename = '{}/vanilla_grad/{}_{}.png'.format(cur_output_folder, label_count, modelnet40_classes[in_pred]['name'])
                vanilla_gradients_3d = gradient_saliency.GetMask(in_pcl, 
                feed_dict=feed_dict)
                pc_grad_van, img_van = VisualisePcdGrad(in_pcl, vanilla_gradients_3d,show=visualise, save=save, output_filename=output_filename, save_image=save_image)

                # ##guided backprop
                output_filename = '{}/guided_backprop/{}_{}.png'.format(cur_output_folder, label_count, modelnet40_classes[in_pred]['name'])
                vanilla_gradients_3d = guided_backprop.GetMask(in_pcl, feed_dict=feed_dict)
                pc_grad_van_guided, img_van_guided = VisualisePcdGrad(in_pcl, vanilla_gradients_3d,show=visualise,save=save, output_filename=output_filename, save_image=save_image )

                # ##integrated grad
                output_filename = '{}/integrated_grad/{}_{}.png'.format(cur_output_folder, label_count, modelnet40_classes[in_pred]['name'])
                vanilla_gradients_3d = integrated_gradient.GetMask(in_pcl, 
                feed_dict=feed_dict )
                pc_grad_van_guided, img_van_guided = VisualisePcdGrad(in_pcl, vanilla_gradients_3d,show=visualise,save=save, output_filename=output_filename, save_image=save_image)

                modelnet40_classes[in_label]['count'] = label_count+1
                # snapshot = tracemalloc.take_snapshot()
                # display_top(snapshot)




if __name__ == '__main__':
    main()


