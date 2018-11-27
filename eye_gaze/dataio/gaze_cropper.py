""" Crop and resize image to the required size parallely """

import os
import sys
import json
import fnmatch
import tarfile
from PIL import Image
from glob import glob
from tqdm import tqdm
from six.moves import urllib
from os.path import expanduser
from multiprocessing import Pool
import cv2

import numpy as np
from scipy.misc import imread, imresize

folder_name = 'imgs_cropped'

def save_array_to_grayscale_image(array, path):
  Image.fromarray(array).convert('L').save(path)

def process_json_list(json_list, img):
  ldmks = [eval(s) for s in json_list]
  return np.array([(x, img.shape[0]-y, z) for (x,y,z) in ldmks])

def crop_parallel(json_path):
  """ Center Crop and resize the eye image using extracted center from correposding JSON file
  If image already processed, it is skipped
  """
  jpg_path = json_path.replace('json', 'jpg')

  if not os.path.exists(jpg_path):
    return

  if os.path.exists(jpg_path.replace(".jpg", "_cropped.png").replace('imgs', folder_name)):
    return

  try:
    with open(json_path) as json_f:
      img = imread(jpg_path)
      j = json.loads(json_f.read())

      key = "interior_margin_2d"
      j[key] = process_json_list(j[key], img)

      x_min, x_max = int(min(j[key][:,0])), int(max(j[key][:,0]))
      y_min, y_max = int(min(j[key][:,1])), int(max(j[key][:,1]))

      x_center, y_center = (x_min + x_max)/2, (y_min + y_max)/2
      cropped_img = img[y_center-63: y_center+63, x_center-105:x_center+105]

      cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
      cropped_img = imresize(cropped_img, (35, 55))
      cropped_img = cv2.normalize(cropped_img, np.zeros_like(cropped_img), alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
      img_path = jpg_path.replace(".jpg", "_cropped.png").replace('imgs', folder_name)

      save_array_to_grayscale_image(cropped_img, img_path)
      print('.')
  except:
    print("Exception raised! missing JSON file: ", json_path)

def maybe_preprocess(data_path, syn_data_path):
  """
  Args:
    data_path: Folder containing raw images from simulator
    syn_data_path: Folder to save the processed images to
  """
  jpg_paths = glob(os.path.join(data_path, '*.jpg'))
  cropped_jpg_paths = glob(os.path.join(syn_data_path, '*_cropped.png'))

  if len(jpg_paths) == 0:
    print("[!] No images in ./{}. Skip.".format(data_path))
  else:
    print("[!] Found images in ./{}.".format(data_path))
    if len(cropped_jpg_paths) != len(jpg_paths):
      json_paths = glob(os.path.join(
          data_path, '{}/*.json'.format(data_path)))

      p = Pool(8)
      p.map(crop_parallel, json_paths)

      jpg_paths = glob(os.path.join(data_path, '*.jpg'))
      cropped_jpg_paths = glob(os.path.join(syn_data_path, '*_cropped.png'))

      print("[*] # of synthetic data: {}, # of cropped_data: {}". \
          format(len(jpg_paths), len(cropped_jpg_paths)))
      print("[*] Finished preprocessing synthetic `gaze` data.")
    return

  raise Exception("[!] Failed to found proper synthetic_image_path")

if __name__=="__main__":
  home = expanduser("~")
  data_path = home + '/domain_adaptation_datasets/eye_gaze_4_april_2018/imgs/'
  syn_data_path = home + '/domain_adaptation_datasets/eye_gaze_4_april_2018/' + folder_name + '/'
  maybe_preprocess(data_path, syn_data_path)
