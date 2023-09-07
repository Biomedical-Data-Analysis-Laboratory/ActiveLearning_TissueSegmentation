"""
## Importing required libraries
"""
import warnings

warnings.simplefilter(action='ignore',
                      category=FutureWarning)  # Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore',
                      category=DeprecationWarning)  # Ignore all DeprecationWarning warnings that might flood the console log

import os

import tensorflow as tf
tf.enable_eager_execution()
import pyvips

import sys
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
import keras
sys.stderr = stderr

import numpy as np
from skimage.draw import polygon2mask, polygon, polygon_perimeter
from skimage.transform import resize
from skimage import img_as_bool
from scipy import ndimage
import xml.etree.ElementTree as ET

import json
import mysql.connector
from mysql.connector import errorcode
import my_constants


"""
## Functions
"""

############

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value ist tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def serialize_array(array):
    array = tf.io.serialize_tensor(array)
    return array


############

def parse_single_tile(tile_info):
    # extract tile info
    wsi_filename = tile_info['wsi_filename']
    label = tile_info['label']
    tissue_type = tile_info['tissue_type']
    image_400x = tile_info['image_400x']
    image_100x = tile_info['image_100x']
    image_25x = tile_info['image_25x']
    (tile_x_400, tile_y_400) = tile_info['coordinates_400x']
    (tile_x_100, tile_y_100) = tile_info['coordinates_100x']
    (tile_x_25, tile_y_25) = tile_info['coordinates_25x']

    data = {
        'wsi_filename': _bytes_feature(serialize_array(wsi_filename)),
        'label': _int64_feature(label),
        'tissue_type': _bytes_feature(serialize_array(tissue_type)),
        'image_400x': _bytes_feature(serialize_array(image_400x)),
        'image_100x': _bytes_feature(serialize_array(image_100x)),
        'image_25x': _bytes_feature(serialize_array(image_25x)),
        'tile_x_400': _int64_feature(tile_x_400),
        'tile_y_400': _int64_feature(tile_y_400),
        'tile_x_100': _int64_feature(tile_x_100),
        'tile_y_100': _int64_feature(tile_y_100),
        'tile_x_25': _int64_feature(tile_x_25),
        'tile_y_25': _int64_feature(tile_y_25)
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


############

def write_images_to_tfr_short(tiles, filename: str = "tiles"):
    filename = filename + ".tfrecords"
    writer = tf.io.TFRecordWriter(filename)  # create a writer that'll store our data to disk
    count = 0

    for index in range(len(tiles)):
        # get the data we want to write
        current_tile = images[index]

        out = parse_single_tile(current_tile)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} elements to TFRecord")


############

def parse_tfr_element(element):
    # use the same structure as above; it's kinda an outline of the structure we now want to create
    data = {
        'wsi_filename': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
        'tissue_type': tf.io.FixedLenFeature([], tf.string),
        'image_400x': tf.io.FixedLenFeature([], tf.string),
        'image_100x': tf.io.FixedLenFeature([], tf.string),
        'image_25x': tf.io.FixedLenFeature([], tf.string),
        'tile_x_400': tf.io.FixedLenFeature([], tf.int64),
        'tile_y_400': tf.io.FixedLenFeature([], tf.int64),
        'tile_x_100': tf.io.FixedLenFeature([], tf.int64),
        'tile_y_100': tf.io.FixedLenFeature([], tf.int64),
        'tile_x_25': tf.io.FixedLenFeature([], tf.int64),
        'tile_y_25': tf.io.FixedLenFeature([], tf.int64),
    }

    content = tf.io.parse_single_example(element, data)

    wsi_filename = content['wsi_filename']
    label = content['label']
    tissue_type = content['tissue_type']
    image_400x = content['image_400x']
    image_100x = content['image_100x']
    image_25x = content['image_25x']
    tile_x_400 = content['tile_x_400']
    tile_y_400 = content['tile_y_400']
    tile_x_100 = content['tile_x_100']
    tile_y_100 = content['tile_y_100']
    tile_x_25 = content['tile_x_25']
    tile_y_25 = content['tile_y_25']

    filename = tf.io.parse_tensor(wsi_filename, out_type=tf.string)
    tile_tissue_type = tf.io.parse_tensor(tissue_type, out_type=tf.string)

    image_400x_feature = tf.io.parse_tensor(image_400x, out_type=tf.int16)
    image_400x_feature = tf.reshape(image_400x_feature, shape=[256, 256, 3])

    image_100x_feature = tf.io.parse_tensor(image_100x, out_type=tf.int16)
    image_100x_feature = tf.reshape(image_100x_feature, shape=[256, 256, 3])

    image_25x_feature = tf.io.parse_tensor(image_25x, out_type=tf.int16)
    image_25x_feature = tf.reshape(image_25x_feature, shape=[256, 256, 3])

    # get our 'feature'
    return (filename, label, tile_tissue_type, image_400x_feature, image_100x_feature, image_25x_feature, tile_x_400,
            tile_y_400, tile_x_100, tile_y_100, tile_x_25, tile_y_25)


############

def get_dataset_small(filename):
    # create the dataset
    dataset = tf.data.TFRecordDataset(filename)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset


############

"""
## Main
"""

wsi_directory = "WSI_DIRECTORY/"
TILE_SIZE = 8
threshold = 0.7
tile_threshold = int(TILE_SIZE * TILE_SIZE * threshold)
save_tfrecords_folder = "active_tfrecords_test_blood/"
recreate_tfrecords = True

# Retrieve annotations
# In our case, we had coordinates in a SQL database
try:
    cnx = mysql.connector.connect(user='user',
                                  password='pass',
                                  database='db',
                                  host='dummy.com',
                                  port=0000
                                  )
except mysql.connector.Error as err:
    if err.errno == errorcode.ER_ACCESS_DENIED_ERROR:
        print("Something is wrong with your user name or password")
    elif err.errno == errorcode.ER_BAD_DB_ERROR:
        print("Database does not exist")
    else:
        print(err)
else:
    cursor = cnx.cursor()
    query = ("SELECT `mySlice` FROM `keyvalue` WHERE `finished`=1 GROUP BY `mySlice`")
    cursor.execute(query)

    slide_ids = []
    for myValue in cursor:
        slide_ids.append(myValue[0])

# Iterate over slides with annotations
for slide_id in slide_ids:

        query = ("SELECT myValue FROM keyvalue WHERE mySlice={} ORDER BY myTimestamp DESC LIMIT 1".format(slide_id))
        cursor.execute(query)

    for myValue in cursor:
        db_value = json.loads(myValue[0])

    # Extract WSI info
    filename = db_value['filename']
    filename = filename.split('/')[-1][:-4]
    regionsDict = db_value['Regions']
    print('------')
    print(filename)

    # Let's create a dictionary that contains the masks of the regions grouped per label
    masks = dict()
    # Look for image dimensions in DZI file
    tree = ET.parse('root/images/' + filename + ".dzi")
    root = tree.getroot()
    height, width = int(root[0].attrib["Height"]), int(root[0].attrib["Width"])
    shape = (int(width / 100), int(height / 100))

    # Get label and region
    for region in regionsDict:

        # Obtain label and segments that delimit the region
        region_label = region['name']
        if "Untitled" in region_label or "Copy" in region_label: continue
        segments = region['path'][1]['segments']

        if not isinstance(segments[0], list):
            segment_list = []
            for segment in segments:
                segment_list.append([segment['x'], segment['y']])
            points = np.array(segment_list)
        else:
            points = np.array(segments)
        points = np.transpose(points)

        # Draw a poligon mask
        imgp = np.full(shape, False)
        rr, cc = polygon(*points, shape=shape)
        imgp[rr, cc] = True

        # Check whether the maks of the label already exists and apply logical OR
        if region_label not in masks:
            masks[region_label] = np.full(shape, False)
        masks[region_label] = masks[region_label] | imgp

    # Read WSI
    print('Reading slide...')
    wsi_filepath = wsi_directory + filename
    slide_400 = pyvips.Image.new_from_file(wsi_filepath, autocrop=True, level=1).flatten()
    slide_100 = pyvips.Image.new_from_file(wsi_filepath, autocrop=True, level=3).flatten()
    slide_25 = pyvips.Image.new_from_file(wsi_filepath, autocrop=True, level=5).flatten()

    slide25_height = slide_25.height
    slide25_width = slide_25.width

    x_coordinates = list(range(0, slide25_width - TILE_SIZE, TILE_SIZE))
    y_coordinates = list(range(0, slide25_height - TILE_SIZE, TILE_SIZE))

    print('Eliminating overlapping areas...')
    for mask in masks:

        # Apply XOR operator for CIA over anything else
        if 'Cancerous invasive area' not in mask and 'Cancerous invasive area' in masks.keys():
            masks[mask][masks['Cancerous invasive area'] == True] = False

        # Apply XOR operator for blood over lamina propria
        if 'Blood' not in mask and 'Blood' in masks.keys():
            masks[mask][masks['Blood'] == True] = False

        # Apply XOR operator for muscle over lamina propria
        if 'Muscle' not in mask and 'Muscle' in masks.keys():
            masks[mask][masks['Muscle'] == True] = False

    print('Generating background mask...')
    background_mask = np.ones((slide25_height, slide25_width))
    full_slide = slide_25.extract_area(0, 0, slide25_width, slide25_height)

    slide_numpy = full_slide.write_to_memory()
    slide_numpy = np.fromstring(slide_numpy, dtype=np.uint8).reshape(full_slide.height, full_slide.width, 3)

    background_mask[slide_numpy[:, :, 1] > 250] = 0
    background_mask = ndimage.binary_closing(background_mask, structure=np.ones((4, 4))).astype(background_mask.dtype)

    print('Extracting tiles...')
    # TFrecords
    current_shard_name = save_tfrecords_folder + "{}.tfrecords".format(filename)
    writer = tf.io.TFRecordWriter(current_shard_name)

    current_all_annotations_mask = np.ones((slide25_height, slide25_width))
    current_stored_annotations = np.ones((slide25_height, slide25_width))
    current_total_tiles = 0
    for mask in masks:

        if 'Cancerous invasive area' in mask:
            current_mask_label = 1
        else:
            current_mask_label = 0

        # current_mask = img_as_bool(resize(masks[mask], (slide_width,slide_height))).T
        current_mask = img_as_bool(resize(masks[mask], (slide25_width, slide25_height))).T
        current_mask[background_mask == 0] = False
        current_all_annotations_mask[current_mask == True] = 0

        current_number_of_tiles = 0
        for x_coor in x_coordinates:
            for y_coor in y_coordinates:
                if sum(sum(current_mask[y_coor:y_coor + TILE_SIZE, x_coor:x_coor + TILE_SIZE])) > tile_threshold:
                    current_stored_annotations[y_coor:y_coor + TILE_SIZE, x_coor:x_coor + TILE_SIZE] = 0
                    x_coor_temp = x_coor + int(TILE_SIZE / 2)
                    y_coor_temp = y_coor + int(TILE_SIZE / 2)

                    tile_dict = dict()
                    tile_dict['wsi_filename'] = wsi_directory
                    tile_dict['tissue_type'] = mask
                    tile_dict['label'] = current_mask_label

                    image_x400_tile_x_pos = max(int(x_coor_temp * 16 - TILE_SIZE * 8), 0)
                    image_x400_tile_x_pos = min(image_x400_tile_x_pos, slide_400.width - int(TILE_SIZE * 16))
                    image_x400_tile_y_pos = max(int(y_coor_temp * 16 - TILE_SIZE * 8), 0)
                    image_x400_tile_y_pos = min(image_x400_tile_y_pos, slide_400.height - int(TILE_SIZE * 16))

                    tile_400x = slide_400.extract_area(image_x400_tile_x_pos, image_x400_tile_y_pos,
                                                       int(TILE_SIZE * 16), int(TILE_SIZE * 16))
                    tile_dict['image_400x'] = np.ndarray(buffer=tile_400x.write_to_memory(),
                                                         dtype=my_constants.format_to_dtype[tile_400x.format],
                                                         shape=[tile_400x.height, tile_400x.width, tile_400x.bands])

                    image_x100_tile_x_pos = max(int(x_coor_temp * 4 - TILE_SIZE * 8), 0)
                    image_x100_tile_x_pos = min(image_x100_tile_x_pos, slide_100.width - int(TILE_SIZE * 16))
                    image_x100_tile_y_pos = max(int(y_coor_temp * 4 - TILE_SIZE * 8), 0)
                    image_x100_tile_y_pos = min(image_x100_tile_y_pos, slide_100.height - int(TILE_SIZE * 16))

                    tile_100x = slide_100.extract_area(image_x100_tile_x_pos, image_x100_tile_y_pos,
                                                       int(TILE_SIZE * 16), int(TILE_SIZE * 16))
                    tile_dict['image_100x'] = np.ndarray(buffer=tile_100x.write_to_memory(),
                                                         dtype=my_constants.format_to_dtype[tile_100x.format],
                                                         shape=[tile_100x.height, tile_100x.width, tile_100x.bands])

                    image_x25_tile_x_pos = max(int(x_coor_temp - TILE_SIZE * 8), 0)
                    image_x25_tile_x_pos = min(image_x25_tile_x_pos, slide_25.width - int(TILE_SIZE * 16))
                    image_x25_tile_y_pos = max(int(y_coor_temp - TILE_SIZE * 8), 0)
                    image_x25_tile_y_pos = min(image_x25_tile_y_pos, slide_25.height - int(TILE_SIZE * 16))

                    tile_25x = slide_25.extract_area(image_x25_tile_x_pos, image_x25_tile_y_pos, int(TILE_SIZE * 16),
                                                     int(TILE_SIZE * 16))
                    tile_dict['image_25x'] = np.ndarray(buffer=tile_25x.write_to_memory(),
                                                        dtype=my_constants.format_to_dtype[tile_25x.format],
                                                        shape=[tile_25x.height, tile_25x.width, tile_25x.bands])

                    tile_dict['coordinates_400x'] = (int(x_coor_temp * 16), int(y_coor_temp * 16))
                    tile_dict['coordinates_100x'] = (int(x_coor_temp * 4), int(y_coor_temp * 4))
                    tile_dict['coordinates_25x'] = (int(x_coor_temp), int(y_coor_temp))

                    out = parse_single_tile(tile_info=tile_dict)
                    writer.write(out.SerializeToString())
                    current_number_of_tiles += 1
                    current_total_tiles += 1

        print("{}: {} tiles".format(mask, str(current_number_of_tiles)))
    print("TOTAL NUMBER OF TILES: {}".format(str(current_total_tiles)))
    print("")
    writer.close()

cursor.close()
cnx.close()