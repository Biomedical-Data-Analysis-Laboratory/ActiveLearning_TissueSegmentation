"""
## Importing required libraries
"""
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) # Ignore all FutureWarning warnings that might flood the console log
warnings.simplefilter(action='ignore', category=DeprecationWarning) # Ignore all DeprecationWarning warnings that might flood the console log

import os

import tensorflow as tf
import pyvips

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow import keras

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from scipy import ndimage

import time
import pickle
import my_constants


"""
## Functions
"""

# Data generator for prediction
def inference_generator_py(i, coordinates_400x, coordinates_100x, coordinates_25x, wsi_filepath, TILE_SIZE):
    i = i.numpy()

    filepath = wsi_filepath.numpy().decode('utf-8')
    coordinates_400x = coordinates_400x.numpy()
    coordinates_100x = coordinates_100x.numpy()
    coordinates_25x = coordinates_25x.numpy()
    TILE_SIZE = TILE_SIZE.numpy()

    # Split coordinates
    (image_x400_tile_x_pos, image_x400_tile_y_pos), (image_x100_tile_x_pos, image_x100_tile_y_pos), (
    image_x25_tile_x_pos, image_x25_tile_y_pos) = coordinates_400x[i], coordinates_100x[i], coordinates_25x[i]

    slide_400 = pyvips.Image.new_from_file(filepath, autocrop=True, level=1).flatten()
    slide_100 = pyvips.Image.new_from_file(filepath, autocrop=True, level=3).flatten()
    slide_25 = pyvips.Image.new_from_file(filepath, autocrop=True, level=5).flatten()


    # Extract tiles
    tile_400x = slide_400.extract_area(image_x400_tile_x_pos,image_x400_tile_y_pos,int(TILE_SIZE*16),int(TILE_SIZE*16))
    tile_400x_numpy = np.ndarray(buffer=tile_400x.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_400x.format],
                                        shape=[tile_400x.height, tile_400x.width, tile_400x.bands])

    tile_100x = slide_100.extract_area(image_x100_tile_x_pos,image_x100_tile_y_pos,int(TILE_SIZE*16),int(TILE_SIZE*16))
    tile_100x_numpy = np.ndarray(buffer=tile_100x.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_100x.format],
                                        shape=[tile_100x.height, tile_100x.width, tile_100x.bands])

    tile_25x = slide_25.extract_area(image_x25_tile_x_pos,image_x25_tile_y_pos,int(TILE_SIZE*16),int(TILE_SIZE*16))
    tile_25x_numpy = np.ndarray(buffer=tile_25x.write_to_memory(),
                                        dtype=my_constants.format_to_dtype[tile_25x.format],
                                        shape=[tile_25x.height, tile_25x.width, tile_25x.bands])


    return tile_400x_numpy, tile_100x_numpy, tile_25x_numpy
    

# TRI-CNN model architecture
def get_tri_scale_model():
    # Model input
    image_input_400x = Input(shape=(128, 128, 3), name='input_400x')
    image_input_100x = Input(shape=(128, 128, 3), name='input_100x')
    image_input_25x = Input(shape=(128, 128, 3), name='input_25x')
           
    base_model_400x = VGG16(input_tensor=image_input_400x, include_top=False, weights='imagenet', pooling='avg')
    base_model_100x = VGG16(input_tensor=image_input_100x, include_top=False, weights='imagenet', pooling='avg')
    base_model_25x = VGG16(input_tensor=image_input_25x, include_top=False, weights='imagenet', pooling='avg')

    last_layer_400x = base_model_400x.layers[-1].output
    last_layer_100x = base_model_100x.layers[-1].output
    last_layer_25x = base_model_25x.layers[-1].output

    # Rename all layers in first model
    for layer in base_model_100x.layers:
        layer._name = layer._name + str("_100x")

    # Rename all layers in second model
    for layer in base_model_25x.layers:
        layer._name = layer._name + str("_25x")

    # Concatenate models
    flatten_layer = keras.layers.concatenate([last_layer_400x, last_layer_100x, last_layer_25x], axis=-1)

    # Define new classifier architecture
    my_dense = Dense(4096, activation='relu', name='my_dense1')(flatten_layer)
    # my_dense = Dropout(rate=0.4, noise_shape=None, seed=None)(my_dense)
    my_dense = Dense(4096, activation='relu', name='my_dense2')(my_dense)
    # my_dense = Dropout(rate=0.4, noise_shape=None, seed=None)(my_dense)

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(6, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(inputs=[image_input_400x, image_input_100x, image_input_25x], outputs=my_output)

    return TL_classifier_model


# Make colormap based on live predictions
def make_colormap(N_CLASSES_ALL, NAME_OF_CLASSES_ALL, threshold,  all_colormap_images, current_save_folder):

    print('Making colormap')

    # Check that data exist in the probability images
    if not (len(all_colormap_images) > 0):
        print('Probability pickle file is empty. Stopping program.', error=True)
        exit()

    # Get seg mask width/height
    seg_mask_width = all_colormap_images[0].shape[1]
    seg_mask_heights = all_colormap_images[0].shape[0]

    for cur_class in range(N_CLASSES_ALL):
        curr_prob_img = all_colormap_images[cur_class]
        for row in curr_prob_img:
            row[row < threshold] = 0
            row[row >= threshold] = cur_class

    colors = [
        (0, 0, 0),  # Black     - Background
        (0.5, 0, 0),  # Maroon    - Blood
        (0, 0.51, 0.7843),  # Blue      - Damaged
        (0.23529, 0.70588, 0.294117),  # Green     - Muscle
        (0.9412, 0.196, 0.9019),  # Magenta   - Stroma
        (0.96078, 0.51, 0.18823),  # Orange    - Urothelium
        (0.5, 0.5, 0.5)  # Grey      - Undefined
    ]

    # RGB Values
    # colors = [
    #     (0, 0, 0),  # Black     - Background
    #     (128, 0, 0),  # Maroon    - Blood
    #     (0, 130, 200),  # Blue      - Damaged
    #     (60, 180, 75),  # Green     - Muscle
    #     (240, 50, 230),  # Magenta   - Stroma
    #     (244, 130, 47),  # Orange    - Urothelium
    #     (128, 128, 128)  # Grey      - Undefined
    # ]

    # Create an empty mask of same size as image
    seg_img = np.zeros(shape=(seg_mask_heights, seg_mask_width, 3))

    for c in range(N_CLASSES_ALL):
        segc = (all_colormap_images[c] == c)
        seg_img[:, :, 0] += (segc * (colors[c][0]))
        seg_img[:, :, 1] += (segc * (colors[c][1]))
        seg_img[:, :, 2] += (segc * (colors[c][2]))

    # Backup color
    # (1, 0.98, 0.7843),  # Beige
    # (0.9412, 0.196, 0.9019),  # Magenta   - Stroma
    # (0.23529, 0.70588, 0.294117),  # Green     - Muscle
    # (0.96078, 0.51, 0.18823),  # Orange    - Urothelium

    # Create legend
    legend_patch = []
    for n in range(N_CLASSES_ALL):
        legend_patch.append(mpatches.Patch(color=colors[n], label=NAME_OF_CLASSES_ALL[n]))

    # Create segmentation image
    filename = 'Colormap_image_thres_' + str(threshold) + '.png'
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.imshow(seg_img)
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
    plt.savefig(current_save_folder + filename, bbox_inches='tight', pad_inches=0)

    ax2.legend(handles=legend_patch, loc='lower left')
    filename = 'Colormap_image_legend_thres_' + str(threshold) + '.png'
    plt.savefig(current_save_folder + filename, bbox_inches='tight', pad_inches=0)

#####################

"""
## Main
"""

N_CLASSES_TRAINING = 6 # All tissue types
N_CLASSES_ALL = 7 # We include undefined
UNDEFINED_CLASS_THRESHOLD = 0.6
DAMAGED_CLASS_THRESHOLD = 0.9
PREDICT_WINDOW_SIZE = 128
TILE_SIZE = 8
threshold = 0.7
tile_threshold = int(TILE_SIZE*TILE_SIZE*threshold)

EXTRACTION_TILE_SIZE = 64 # This corresponds to 25x view, 100x will be four times larger
TILE_THRESHOLD = 0.7 # For extraction at 10x

weights_filename = 'AL_Model.h5'
BATCH_SIZE = 512
SAVE_FOLDER = 'history/'
wsi_directory = 'DATASET_DIRECTORY/'
list_of_wsi_files = os.listdir(wsi_directory)

for filename in list_of_wsi_files:

    script_start_time = time.time()

    preds_and_coords_dict = dict()
    current_save_folder = SAVE_FOLDER + filename + '/'
    if not os.path.exists(current_save_folder):
        os.makedirs(current_save_folder)
    else:
        continue

    print('')
    print('*'*100)
    print('Processing {}'.format(filename))
    print('*'*100)

    # Read WSI
    print('Reading slide...')
    wsi_filepath = wsi_directory + filename
    slide_400 = pyvips.Image.new_from_file(wsi_filepath, autocrop=True, level=1).flatten()
    slide_100 = pyvips.Image.new_from_file(wsi_filepath, autocrop=True, level=3).flatten()
    slide_25 = pyvips.Image.new_from_file(wsi_filepath, autocrop=True, level=5).flatten()

    slide25_height = slide_25.height
    slide25_width = slide_25.width

    print('Generating background mask...')

    background_mask = np.ones((slide25_height, slide25_width))
    full_slide = slide_25.extract_area(0,0,slide25_width,slide25_height)
    slide_numpy = full_slide.write_to_memory()
    slide_numpy = np.fromstring(slide_numpy, dtype=np.uint8).reshape(full_slide.height, full_slide.width, 3)

    # Apply threshold and morphological closing
    background_mask[slide_numpy[:,:,1] > 250] = 0
    background_mask = ndimage.binary_closing(background_mask, structure=np.ones((4,4))).astype(background_mask.dtype)

    non_background_mask = np.ones((slide25_height, slide25_width))
    non_background_mask[background_mask == 0] = 0

    print('Extracting tiles...')

    x_coordinates = list(range(0,slide25_width-TILE_SIZE,TILE_SIZE))
    y_coordinates = list(range(0,slide25_height-TILE_SIZE,TILE_SIZE))

    current_number_of_tiles = 0
    coordinates_400x_all = []
    coordinates_400x = []
    coordinates_100x = []
    coordinates_25x = []

    for x_coor in x_coordinates:
        for y_coor in y_coordinates:

            x_coor_temp = x_coor + int(TILE_SIZE/2)
            y_coor_temp = y_coor + int(TILE_SIZE/2)
            coordinates_400x_all.append((min(int(x_coor_temp*16 - TILE_SIZE*8),slide_400.width-int(TILE_SIZE*16)), min(int(y_coor_temp*16 - TILE_SIZE*8),slide_400.height-int(TILE_SIZE*16))))

            if sum(sum(non_background_mask[y_coor:y_coor+TILE_SIZE,x_coor:x_coor+TILE_SIZE])) > tile_threshold:

                image_x400_tile_x_pos = max(int(x_coor_temp*16 - TILE_SIZE*8), 0)
                image_x400_tile_x_pos = min(image_x400_tile_x_pos, slide_400.width-int(TILE_SIZE*16))
                image_x400_tile_y_pos = max(int(y_coor_temp*16 - TILE_SIZE*8), 0)
                image_x400_tile_y_pos = min(image_x400_tile_y_pos, slide_400.height-int(TILE_SIZE*16))

                image_x100_tile_x_pos = max(int(x_coor_temp*4 - TILE_SIZE*8), 0)
                image_x100_tile_x_pos = min(image_x100_tile_x_pos, slide_100.width-int(TILE_SIZE*16))
                image_x100_tile_y_pos = max(int(y_coor_temp*4 - TILE_SIZE*8), 0)
                image_x100_tile_y_pos = min(image_x100_tile_y_pos, slide_100.height-int(TILE_SIZE*16))

                image_x25_tile_x_pos = max(int(x_coor_temp - TILE_SIZE*8), 0)
                image_x25_tile_x_pos = min(image_x25_tile_x_pos, slide_25.width-int(TILE_SIZE*16))
                image_x25_tile_y_pos = max(int(y_coor_temp - TILE_SIZE*8), 0)
                image_x25_tile_y_pos = min(image_x25_tile_y_pos, slide_25.height-int(TILE_SIZE*16))

                coordinates_400x.append((image_x400_tile_x_pos,image_x400_tile_y_pos))
                coordinates_100x.append((image_x100_tile_x_pos,image_x100_tile_y_pos))
                coordinates_25x.append((image_x25_tile_x_pos,image_x25_tile_y_pos))

                current_number_of_tiles += 1

    print("TOTAL NUMBER OF TILES: {}".format(str(current_number_of_tiles)))
    print("Number of tiles including background: {}".format(int(len(x_coordinates)*len(y_coordinates))))
    print("")

    if not os.path.exists(current_save_folder + 'preds_and_coords.obj'):

        wsi_dataset = tf.data.Dataset.from_generator(lambda: list(range(len(coordinates_400x))), tf.int32)
        wsi_dataset = wsi_dataset.map(lambda i: tf.py_function(func=inference_generator_py,
                                        inp=[i, coordinates_400x, coordinates_100x, coordinates_25x, wsi_filepath, TILE_SIZE],
                                        Tout=[tf.uint8, tf.uint8, tf.uint8]
                                        )
                                    )
        wsi_dataset = wsi_dataset.map(lambda x, y, z: {'input_400x': x, 'input_100x_100x': y, 'input_25x_25x': z})

        model = get_tri_scale_model()

        initial_weights = [layer.get_weights() for layer in model.layers]
        model.load_weights(weights_filename, by_name=True)
        for layer, initial in zip(model.layers, initial_weights):
                weights = layer.get_weights()
                if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                    print(f'Checkpoint contained no weights for layer {layer.name}!')

        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.SGD(lr=0.000015, decay=1e-6, momentum=0.9, nesterov=True),
            metrics=[
                'accuracy'
            ],
        )

        # Getting predictions from previously trained model
        print('Starting prediction')
        all_predictions = model.predict(wsi_dataset.batch(BATCH_SIZE))

        preds_and_coords_dict['preds'] = all_predictions
        preds_and_coords_dict['tile_size'] = PREDICT_WINDOW_SIZE
        preds_and_coords_dict['coords_400x'] = coordinates_400x
        preds_and_coords_dict['coords_100x'] = coordinates_100x
        preds_and_coords_dict['coords_25x'] = coordinates_25x
        with open(current_save_folder + 'preds_and_coords.obj', 'wb') as handle:
            pickle.dump(preds_and_coords_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else: # WSI already predicted

        with open(os.path.join(current_save_folder, 'preds_and_coords.obj'), 'rb') as f:
            preds_and_coords_dict = pickle.load(f)

        all_predictions = preds_and_coords_dict['preds']
        PREDICT_WINDOW_SIZE = preds_and_coords_dict['tile_size']
        coordinates_400x = preds_and_coords_dict['coords_400x']
        coordinates_100x = preds_and_coords_dict['coords_100x']
        coordinates_25x = preds_and_coords_dict['coords_25x']

    """COLORMAP"""
    if not os.path.exists(current_save_folder + 'colormap.obj'):

        # Make a empty list to store the probability array in
        colormap_image_all_classes_list = []
        for _ in range(N_CLASSES_ALL):
            colormap_image_all_classes_list.append(np.zeros(shape=(len(y_coordinates), len(x_coordinates)), dtype=float))

        for current_xy_pos in coordinates_400x_all:

            # Save background
            if current_xy_pos not in coordinates_400x:
                # Update colormap for background positions
                colormap_image_all_classes_list[0][
                    (current_xy_pos[1]) // PREDICT_WINDOW_SIZE,
                    (current_xy_pos[0]) // PREDICT_WINDOW_SIZE
                ] = 1

            # It's tissue
            else:
                xy_pred_index = coordinates_400x.index(current_xy_pos)

                # Check if the largest prediction is below the threshold. If yes, it is defined as undefined class. Add prediction to the array.
                curr_max_pred = max(all_predictions[xy_pred_index])
                curr_class_id = np.where(all_predictions[xy_pred_index] == curr_max_pred)
                if curr_max_pred < UNDEFINED_CLASS_THRESHOLD:
                    colormap_image_all_classes_list[6][
                        (current_xy_pos[1]) // PREDICT_WINDOW_SIZE,
                        (current_xy_pos[0]) // PREDICT_WINDOW_SIZE
                    ] = 1

                else:
                    if curr_class_id == 2 and curr_max_pred < DAMAGED_CLASS_THRESHOLD:
                        curr_max_pred = np.partition(all_predictions[xy_pred_index].flatten(), -2)[-2]

                    for i in range(N_CLASSES_TRAINING):
                        if curr_max_pred == all_predictions[xy_pred_index][i]:
                            colormap_image_all_classes_list[i][
                                (current_xy_pos[1]) // PREDICT_WINDOW_SIZE,
                                (current_xy_pos[0]) // PREDICT_WINDOW_SIZE
                            ] = 1

        with open(current_save_folder + 'colormap.obj', 'wb') as handle:
            pickle.dump(colormap_image_all_classes_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

    else: # Colormap already exists

        # Load colormap
        with open(current_save_folder + 'colormap.obj', 'rb') as f:
            colormap_image_all_classes_list = pickle.load(f)

    # COLORMAP
    make_colormap(N_CLASSES_ALL=N_CLASSES_ALL,
                    NAME_OF_CLASSES_ALL=['Background', 'Blood', 'Damaged', 'Muscle', 'Stroma', 'Urothelium', 'Undefined'],
                    threshold=0.2,
                    COLORMAP_IMAGES_PICKLE_FILE=colormap_image_all_classes_list,
                    current_save_folder=current_save_folder)

    elapse_time = time.time() - script_start_time
    m, s = divmod(elapse_time, 60)
    h, m = divmod(m, 60)
    model_time = '%02d:%02d:%02d' % (h, m, s)
    print('Prediction finished. Duration: {}'.format(model_time))
    print("")