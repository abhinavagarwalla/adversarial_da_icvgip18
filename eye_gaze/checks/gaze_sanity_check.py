""" Script to check the discrepency in real and synthetic gaze vectors """

import numpy as np
from PIL import Image
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import glob
from scipy.io import loadmat
from os.path import expanduser
from scipy.misc import imresize
import json
import cv2

def get_syn_img_lbl(addr):
    with open(addr) as json_f:
        j = json.loads(json_f.read())
        j = map(float, j["eye_details"]["look_vec"].encode('utf-8')[1:-1].split(','))
        look_vec = [j[0], -j[1], j[2]] #j[:3]
    return np.array(look_vec).astype(np.float32)

def load_image_label_real(addr, lbl):
    print("Addr: ", addr)
    data_arr = loadmat(addr)
    data = data_arr['data'][0][0][0][0][0]
    imgs, look_vecs, l2loss = [], [], []
    for idx in range(len(data[0])):
        look_vecs.append(data[0][idx])
        imgs.append(imresize(data[1][idx], (35, 55)))
        l2loss.append(np.mean(np.abs(data[0][idx]-lbl)))
    l2loss = np.array(l2loss)
    idx = np.where(l2loss==np.min(l2loss))[0][0]
    print("Index: ", idx, imgs[idx].shape, look_vecs[idx].shape, look_vecs[idx], l2loss[idx].shape, l2loss[idx])
    return imgs[idx], look_vecs[idx], l2loss[idx]

def get_mat_real(lbl):
    img_data_path = expanduser('~') + '/domain_adaptation_datasets/MPIIGaze/Data/Normalized/'
    patients = os.listdir(img_data_path)
    addrs = [glob.glob(img_data_path + i + '/*') for i in patients]
    for idx in range(len(addrs)):
        for idx2 in range(len(addrs[idx])):
            imgs, look_vecs, l2loss = load_image_label(addrs[idx][idx2], lbl)
            print(imgs.shape, look_vecs.shape, l2loss.shape, l2loss)
            eye_c = np.array([55/2, 35/2])
            cv2.line(imgs, tuple(eye_c), tuple(eye_c+(np.array(look_vecs[:2])*80).astype(int)), (255,255,255), 3)
            im = Image.fromarray(imgs.reshape(35, 55))
            im.save(str(l2loss) + '_' + np.array_str(look_vecs) + '_' + np.array_str(lbl) + '.jpg')

def check_from_real_to_syn():
    lbl = get_syn_img_lbl('check/100001.json')
    get_mat(lbl)

def get_reference_real_vec():
    data_arr = loadmat(expanduser('~') + '/domain_adaptation_datasets/MPIIGaze/Data/Normalized/p10/day12.mat')
    data = data_arr['data'][0][0][0][0][0]
    look_vec = data[0][10]
    img = imresize(data[1][10], (35, 55))
    return np.array(img), np.array(look_vec).astype(np.float32)

def check_from_syn_to_real():
    img, vec = get_reference_real_vec()
    eye_c = np.array([55/2, 35/2])
    print("Original Vector: ", vec)
    cv2.line(img, tuple(eye_c), tuple(eye_c+(np.array(vec[:2])*80).astype(int)), (255,255,255), 3)
    im = Image.fromarray(img.reshape(35, 55))
    im.save(np.array_str(vec) + '_original.jpg')
    
    syn_data_path = expanduser('~') + '/domain_adaptation_datasets/eye_gaze/imgs_cropped/*.png'
    addrs = glob.glob(syn_data_path)
    labels = [ad.replace('imgs_cropped', 'imgs').replace('_cropped.png', '.json') for ad in addrs]

    minloss = 10.
    minaddr = None
    minlookvec = None
    for i in labels:
        look_vec = get_syn_img_lbl(i)
        l2loss = np.mean(np.abs(look_vec - vec))
        if l2loss < minloss:
            print("loss: ", l2loss, ", i: ", i, ", look:, vec: ", look_vec, vec, look_vec-vec)
            minloss = l2loss
            minaddr = i
            minlookvec = look_vec

    print("MinLoss, MinAddr, Minlookvec: ", minloss, minaddr, minlookvec)

check_from_syn_to_real()
