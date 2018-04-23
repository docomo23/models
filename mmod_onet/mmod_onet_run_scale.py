
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import keras
from keras.models import load_model, Model
from PIL import Image
import numpy as np
from keras import layers
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import glob
import functools
import keras.backend as K
from PIL import Image
from operator import itemgetter
import sys
import tools_matrix as tools
import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
# from MTCNN import create_Kao_Onet, create_Kao_Rnet, create_Kao_Pnet
import re
import argparse
from PIL import ImageOps

K.clear_session()
mmod = load_model("mmod.hdf5")
Onet = load_model("Onet.hdf5")


# In[3]:


def run_mmod(filename, model, input_size, scale, detecting_window_size=(40,40), display_result=False):
    def img_preprocess(img, input_size, scale):
        """
        image processing including resize, padding, normalization
        """
        ## scale down
        w, h = img.size
        img_scaled = img.resize((int(img.size[0]*scale), int(img.size[1]*scale)))
        
        ## padding
        background = Image.new("RGB", (input_size, input_size))
        padding_x = (input_size-img_scaled.size[0])//2
        padding_y = (input_size-img_scaled.size[1])//2
        background.paste(img_scaled, (padding_x, padding_y))
        img = background
        ## normalization
        mean=(122.782, 117.001, 104.298)
        img = np.array(img, dtype=float)
        img[:,:,0] = (img[:,:,0] - mean[0]) / 256.
        img[:,:,1] = (img[:,:,1] - mean[1]) / 256.
        img[:,:,2] = (img[:,:,2] - mean[2]) / 256.
        img = np.expand_dims(img, axis=0)
        return img, padding_x, padding_y
    
    def map_output_to_input(point, strides=(2,2,2), paddings=(0,0,0), ncs=(5,5,5), nrs=(5,5,5)):
        """
        trackback its corresponding point on input tensor untill the very begining layer
        """
        for stride, padding, nc, nr in zip(strides[::-1], paddings[::-1], ncs[::-1], nrs[::-1]):
            point[0] = point[0]*stride - padding + nr//2
            point[1] = point[1]*stride - padding + nc//2
        return [int(val) for val in point]
    
    def to_label(output_tensor, window=(40, 40), adj_threshold=0.):
        """
        find points over threshold and get their corresponding point on input tensor
        """
        output_tensor = np.squeeze(output_tensor)
        nr, nc, nk = output_tensor.shape
        dets = []
        for k in range(nk):
            # currently we only use the first kernel
            if k == 0: 
                for r in range(nr):
                    for c in range(nc):
                        if output_tensor[r,c,k] > adj_threshold:
                            r_, c_ = map_output_to_input([r,c])
                            # detected box is stored as (top, left, width, height, score)
                            dets.append((r_ - window[1]//2,c_ - window[0]//2, *window, output_tensor[r,c,k]))
        return dets 
    
    ## open image and regularize its size
    img = Image.open(filename)
    W_original, H_original  = img.size
    img.thumbnail((input_size, input_size), Image.ANTIALIAS)
    W_resize, H_resize = img.size
    
    ## preprocessing image
    img_processed, padding_x, padding_y = img_preprocess(img, input_size, scale)
#     print(scale, img_processed.shape)
    _, H, W, _ = img_processed.shape

    ## do inference and get the detected bounding box
    output = model.predict(img_processed)
    dets = to_label(output, detecting_window_size)

    ## shift and scale bounding box to match original image
    dets = [(det[0]-padding_y, det[1]-padding_x, *det[2:]) for det in dets]
    dets = [(int(det[0]/scale), int(det[1]/scale), int(det[2]/scale), int(det[3]/scale), det[4]) for det in dets]
    scale_x, scale_y = W_resize / float(W_original), H_resize / float(H_original)
    dets = [(int(det[0]/scale_y), int(det[1]/scale_x), int(det[2]/scale_x), int(det[3]/scale_y), det[4]) for det in dets]
    ## last element has the biggest score
    dets.sort(key=itemgetter(4))
    
    return dets


def run_onet(filename, rectangle, Onet, display_result=True):
    img = cv2.imread(filename)
    
    origin_h, origin_w, ch = img.shape

    ## mmod's output rectangle is (top, left, width, height), but Onet requires rectangle to be (left, top, right, bottom)
    rectangle = (rectangle[1], rectangle[0], (rectangle[1]+rectangle[3]), (rectangle[0]+rectangle[2]))
    ## regularize rectangle to make it stay in image
    rectangle = (max(0,rectangle[0]), max(0, rectangle[1]), min(origin_w, rectangle[2]), min(origin_h, rectangle[3]))

    ## crop and scale the rectangle area of input image
    crop_img = img[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
    scale_img = cv2.resize(crop_img, (48,48))
    x = np.array(scale_img)
    x = (x - 127.5)/127.5
    x = np.expand_dims(x, axis=0)

    ## conducting inference
    y = Onet.predict(x)
    
    ## nms and post-processing the output
    cls_prob = y[0]
    roi_prob = y[1]
    pts_prob = y[2] 
    rectangles = [rectangle]
    y = tools.filter_face_48net(cls_prob, roi_prob, pts_prob, rectangles, origin_w, origin_h, 0)
    y = y[0]

    ## return and/or display landmarks
    output = []
    cv2.rectangle(img, (int(y[0]), int(y[1])), (int(y[2]), int(y[3])), (0, 0, 255), int(2./200*min(origin_h,origin_w)))
    for i in range(5, 15, 2):
        cv2.circle(img, (int(y[i + 0]), int(y[i + 1])), int(2./200*min(origin_h,origin_w)), (0, 0, 255), -1)
        output.append(int(y[i+0]))
        output.append(int(y[i+1]))
    
    if display_result:
        print("==== showing Onet's detection ====")
        fig = plt.figure()
        plt.imshow(img[...,::-1])
        plt.show()
    return output

def display(img, dets, name=""):
    """
    show mmod's detection result, bounding boxes are red except for the max score one, whose color is blue and who will be the input to Onet.
    """
    img = np.squeeze(img)
    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.imshow(img)
    for i, box in enumerate(dets):
        if i != len(dets) - 1:
            rect = patches.Rectangle((box[1], box[0]), box[2], box[3], linewidth=1, edgecolor='r', fill=False)
        else:
            rect = patches.Rectangle((box[1], box[0]), box[2], box[3], linewidth=2,edgecolor='b', fill=False)
        ax.add_patch(rect)
    plt.show()

## call this function to do detection
def face_detection(filename, pyramid_down, display_mmod_result=False,display_mtcnn_result=False):
    ## do MMOD detection
    input_size, scale_base, num_resize = pyramid_down
    scale_list = [scale_base**n for n in range(num_resize)]
    dets_total = []
    for scale in scale_list:
        dets = run_mmod(filename, mmod,
                        input_size=input_size,
                        scale=scale
                        )
        dets_total.extend(dets)
        
    if display_mmod_result and len(dets_total) > 0:
        print("==== showing mmod's detection ====")
        display(np.array(np.array(Image.open(filename))), dets_total, name="")
    
    dets_total.sort(key=itemgetter(4))

    ## do MTCNN [Onet] detection
    if len(dets_total) > 0:
        rectangle = dets_total[-1]
        return run_onet(filename, rectangle, Onet, display_result=display_mtcnn_result)
    return []


# In[4]:

filename = 'b.png'
pyramid_down = (200, 6/7., 10)
face_detection(filename,
               pyramid_down,
               display_mmod_result=True,
               display_mtcnn_result=True)



# pyramid_down = (200, 6/7., 10)
# src = './images_shuffle(20,60)/'
# file_list = glob.glob('%s/**/*' % src, recursive=True)
# false = 0
# for filename in file_list:
#     dets = face_detection(filename,
#                            pyramid_down,
#                            display_mmod_result=False,
#                            display_mtcnn_result=False)
#     if len(dets) == 0:
#         false += 1
#         img = Image.open(filename)
#         plt.imshow(np.array(img))
#         plt.show()
#         print(filename, false)


# In[ ]:


# src1 = './images_shuffle(20,60)/'
# src2 = '/home/luna/Work/Deling/3D_face_data/3d_face_dataset/3DTEC/textureimages'
# src3 = '/home/luna/Work/Deling/face_detection_dataset/actors'
# src4 = '/home/luna/Work/Deling/face_detection_dataset/actresses'
# file_list = glob.glob('%s/**/*' % src1, recursive=True) + glob.glob('%s/**/*' % src2, recursive=True) + glob.glob('%s/**/*' % src3, recursive=True) + glob.glob('%s/**/*' % src4, recursive=True)
# for init_size in range(140, 180, 10):
#     for scale in range(5, 10, 1):
#         for num in range(6, 12, 1):
#             pyramid_down = (init_size, float(scale-1) / scale, num)
#             for filename in file_list:
#                 dets = face_detection(filename,
#                                     pyramid_down,
#                                     display_mmod_result=False,
#                                     display_mtcnn_result=False)
#                 if len(dets) == 0:
#                     # fig = plt.figure(figsize=(2,2))
#                     false += 1
#                     # img = Image.open(filename)
#                     # plt.imshow(np.array(img))
#                     # plt.show()
#                     print('failed case: ', filename)
#                 print(pyramid_down, "->failed case total_number : ", false)


# get_ipython().run_line_magic('timeit', 'dets = face_detection(filename, pyramid_down,display_mmod_result=False,display_mtcnn_result=False)')

