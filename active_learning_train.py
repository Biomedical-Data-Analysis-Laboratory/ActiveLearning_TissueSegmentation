"""
## Importing required libraries
"""
import os

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Input

from tensorflow.keras import Sequential
from tensorflow.keras.layers.experimental import preprocessing

import tensorflow as tf
from tensorflow import keras
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import pickle

import gc
import random
import my_constants

random.seed(3)

script_start_time = time.time()

"""
## Functions
"""

# Dataset generator from TFRecrods
def generate_set(list_of_tfrecord_files, set_split, is_it_training):
    dataset_counter_dict = dict()
    x_blood_400x = []
    x_blood_100x = []
    x_blood_25x = []
    y_blood = []
    x_damaged_400x = []
    x_damaged_100x = []
    x_damaged_25x = []
    y_damaged = []
    x_muscle_400x = []
    x_muscle_100x = []
    x_muscle_25x = []
    y_muscle = []
    x_stroma_400x = []
    x_stroma_100x = []
    x_stroma_25x = []
    y_stroma = []
    x_uro_400x = []
    x_uro_100x = []
    x_uro_25x = []
    y_uro = []

    for tfrecord_count, tfrecord_filename in enumerate(list_of_tfrecord_files):

        wsi_counter_dict = dict()

        raw_dataset = tf.data.TFRecordDataset(tfrecord_filename)

        for raw_record in raw_dataset:
            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            for k, v in example.features.feature.items():
                if k == 'image_400x':
                    tile_400x_numpy = v.bytes_list.value[0]
                    tile_400x_numpy = np.frombuffer(tf.io.parse_tensor(tile_400x_numpy, out_type=tf.uint8),
                                                    dtype=np.uint8).reshape(TILE_SIZE, TILE_SIZE, 3)
                elif k == 'image_100x':
                    tile_100x_numpy = v.bytes_list.value[0]
                    tile_100x_numpy = np.frombuffer(tf.io.parse_tensor(tile_100x_numpy, out_type=tf.uint8),
                                                    dtype=np.uint8).reshape(TILE_SIZE, TILE_SIZE, 3)
                elif k == 'image_25x':
                    tile_25x_numpy = v.bytes_list.value[0]
                    tile_25x_numpy = np.frombuffer(tf.io.parse_tensor(tile_25x_numpy, out_type=tf.uint8),
                                                   dtype=np.uint8).reshape(TILE_SIZE, TILE_SIZE, 3)
                elif k == 'tissue_type':
                    label = v.bytes_list.value[0].decode('ascii')

            if 'Blood' in label:
                label = 1
                x_blood_400x.append(tile_400x_numpy)
                x_blood_100x.append(tile_100x_numpy)
                x_blood_25x.append(tile_25x_numpy)
                y_blood.append(label)
            elif 'Cauterized' in label or 'Blurry' in label or 'Folding' in label or 'Others' in label:
                label = 2
                if label in wsi_counter_dict.keys():
                    if wsi_counter_dict[label] == LIMIT_NUMBER_OF_SAMPLES: continue
                x_damaged_400x.append(tile_400x_numpy)
                x_damaged_100x.append(tile_100x_numpy)
                x_damaged_25x.append(tile_25x_numpy)
                y_damaged.append(label)
            elif 'Muscle' in label:
                label = 3
                if label in wsi_counter_dict.keys():
                    if wsi_counter_dict[label] == LIMIT_NUMBER_OF_SAMPLES: continue
                x_muscle_400x.append(tile_400x_numpy)
                x_muscle_100x.append(tile_100x_numpy)
                x_muscle_25x.append(tile_25x_numpy)
                y_muscle.append(label)
            elif 'Lamina' in label:
                label = 4
                if label in wsi_counter_dict.keys():
                    if wsi_counter_dict[label] == LIMIT_NUMBER_OF_SAMPLES: continue
                x_stroma_400x.append(tile_400x_numpy)
                x_stroma_100x.append(tile_100x_numpy)
                x_stroma_25x.append(tile_25x_numpy)
                y_stroma.append(label)
            elif 'Grade' in label or 'CIS' in label or 'Dysplasia' in label or 'Uro' in label or 'Variant' in label or 'Flat' in label or 'uro' in label:
                label = 5
                if label in wsi_counter_dict.keys():
                    if wsi_counter_dict[label] == LIMIT_NUMBER_OF_SAMPLES: continue
                x_uro_400x.append(tile_400x_numpy)
                x_uro_100x.append(tile_100x_numpy)
                x_uro_25x.append(tile_25x_numpy)
                y_uro.append(label)
            else:
                continue

            if label in dataset_counter_dict.keys():
                dataset_counter_dict[label] += 1
            else:
                dataset_counter_dict[label] = 1

            if label in wsi_counter_dict.keys():
                wsi_counter_dict[label] += 1
            else:
                wsi_counter_dict[label] = 1

        print('*******************')
        print('{}/{}: {}'.format(tfrecord_count + 1, len(list_of_tfrecord_files),
                                 tfrecord_filename.split('/')[-1:][0].split('.')[0]))

        for label_id in wsi_counter_dict.keys():
            print('{}: {} tiles '.format(tissue_names_dict[label_id]['display_name'], wsi_counter_dict[label_id]))

        print('-------------------')
        print('Total number of tiles: {} tiles\n'.format(sum(wsi_counter_dict.values())))

    print(f'###################')
    print('Current set size')
    for label_id in dataset_counter_dict.keys():
        print('{}: {} tiles '.format(tissue_names_dict[label_id]['display_name'], dataset_counter_dict[label_id]))
    print('###################\n')

    blood_list = list(zip(x_blood_400x, x_blood_100x, x_blood_25x, y_blood))
    random.shuffle(blood_list)
    x_blood_400x, x_blood_100x, x_blood_25x, y_blood = zip(*blood_list)
    x_blood_400x = np.array(x_blood_400x)
    x_blood_100x = np.array(x_blood_100x)
    x_blood_25x = np.array(x_blood_25x)
    y_blood = np.array(y_blood).reshape((len(y_blood),))
    y_blood = keras.utils.to_categorical(y_blood, num_classes=6)
    del blood_list
    gc.collect()

    damaged_list = list(zip(x_damaged_400x, x_damaged_100x, x_damaged_25x, y_damaged))
    random.shuffle(damaged_list)
    x_damaged_400x, x_damaged_100x, x_damaged_25x, y_damaged = zip(*damaged_list)
    x_damaged_400x = np.array(x_damaged_400x)
    x_damaged_100x = np.array(x_damaged_100x)
    x_damaged_25x = np.array(x_damaged_25x)
    y_damaged = np.array(y_damaged).reshape((len(y_damaged),))
    y_damaged = keras.utils.to_categorical(y_damaged, num_classes=6)
    del damaged_list
    gc.collect()

    muscle_list = list(zip(x_muscle_400x, x_muscle_100x, x_muscle_25x, y_muscle))
    random.shuffle(muscle_list)
    x_muscle_400x, x_muscle_100x, x_muscle_25x, y_muscle = zip(*muscle_list)
    x_muscle_400x = np.array(x_muscle_400x)
    x_muscle_100x = np.array(x_muscle_100x)
    x_muscle_25x = np.array(x_muscle_25x)
    y_muscle = np.array(y_muscle).reshape((len(y_muscle),))
    y_muscle = keras.utils.to_categorical(y_muscle, num_classes=6)
    del muscle_list
    gc.collect()

    stroma_list = list(zip(x_stroma_400x, x_stroma_100x, x_stroma_25x, y_stroma))
    random.shuffle(stroma_list)
    x_stroma_400x, x_stroma_100x, x_stroma_25x, y_stroma = zip(*stroma_list)
    x_stroma_400x = np.array(x_stroma_400x)
    x_stroma_100x = np.array(x_stroma_100x)
    x_stroma_25x = np.array(x_stroma_25x)
    y_stroma = np.array(y_stroma).reshape((len(y_stroma),))
    y_stroma = keras.utils.to_categorical(y_stroma, num_classes=6)
    del stroma_list
    gc.collect()

    uro_list = list(zip(x_uro_400x, x_uro_100x, x_uro_25x, y_uro))
    random.shuffle(uro_list)
    x_uro_400x, x_uro_100x, x_uro_25x, y_uro = zip(*uro_list)
    x_uro_400x = np.array(x_uro_400x)
    x_uro_100x = np.array(x_uro_100x)
    x_uro_25x = np.array(x_uro_25x)
    y_uro = np.array(y_uro).reshape((len(y_uro),))
    y_uro = keras.utils.to_categorical(y_uro, num_classes=6)
    del uro_list
    gc.collect()

    print("Total number of tiles:",
          y_blood.shape[0] + y_damaged.shape[0] + y_muscle.shape[0] + y_stroma.shape[0] + y_uro.shape[0])

    # For the training set, we must define the original training set as well as the pools
    if is_it_training:

        # Creating training and pool splits
        x_400x, x_100x, x_25x, y = (
            tf.concat(
                (
                    x_blood_400x[:set_split],
                    x_damaged_400x[:set_split],
                    x_muscle_400x[:set_split],
                    x_stroma_400x[:set_split],
                    x_uro_400x[:set_split],
                ),
                0,
            ),
            tf.concat(
                (
                    x_blood_100x[:set_split],
                    x_damaged_100x[:set_split],
                    x_muscle_100x[:set_split],
                    x_stroma_100x[:set_split],
                    x_uro_100x[:set_split],
                ),
                0,
            ),
            tf.concat(
                (
                    x_blood_25x[:set_split],
                    x_damaged_25x[:set_split],
                    x_muscle_25x[:set_split],
                    x_stroma_25x[:set_split],
                    x_uro_25x[:set_split],
                ),
                0,
            ),
            tf.concat(
                (
                    y_blood[:set_split],
                    y_damaged[:set_split],
                    y_muscle[:set_split],
                    y_stroma[:set_split],
                    y_uro[:set_split],
                ),
                0,
            ),
        )

        x_pool_blood_400x, x_pool_blood_100x, x_pool_blood_25x, y_pool_blood = (
            x_blood_400x[set_split:],
            x_blood_100x[set_split:],
            x_blood_25x[set_split:],
            y_blood[set_split:],
        )
        x_pool_damaged_400x, x_pool_damaged_100x, x_pool_damaged_25x, y_pool_damaged = (
            x_damaged_400x[set_split:],
            x_damaged_100x[set_split:],
            x_damaged_25x[set_split:],
            y_damaged[set_split:],
        )
        x_pool_muscle_400x, x_pool_muscle_100x, x_pool_muscle_25x, y_pool_muscle = (
            x_muscle_400x[set_split:],
            x_muscle_100x[set_split:],
            x_muscle_25x[set_split:],
            y_muscle[set_split:],
        )
        x_pool_stroma_400x, x_pool_stroma_100x, x_pool_stroma_25x, y_pool_stroma = (
            x_stroma_400x[set_split:],
            x_stroma_100x[set_split:],
            x_stroma_25x[set_split:],
            y_stroma[set_split:],
        )
        x_pool_uro_400x, x_pool_uro_100x, x_pool_uro_25x, y_pool_uro = (
            x_uro_400x[set_split:],
            x_uro_100x[set_split:],
            x_uro_25x[set_split:],
            y_uro[set_split:],
        )

    else:  # Either val or test set

        x_400x, x_100x, x_25x, y = (
            tf.concat(
                (
                    x_blood_400x[:set_split],
                    x_damaged_400x[:set_split],
                    x_muscle_400x[:set_split],
                    x_stroma_400x[:set_split],
                    x_uro_400x[:set_split],
                ),
                0,
            ),
            tf.concat(
                (
                    x_blood_100x[:set_split],
                    x_damaged_100x[:set_split],
                    x_muscle_100x[:set_split],
                    x_stroma_100x[:set_split],
                    x_uro_100x[:set_split],
                ),
                0,
            ),
            tf.concat(
                (
                    x_blood_25x[:set_split],
                    x_damaged_25x[:set_split],
                    x_muscle_25x[:set_split],
                    x_stroma_25x[:set_split],
                    x_uro_25x[:set_split],
                ),
                0,
            ),
            tf.concat(
                (
                    y_blood[:set_split],
                    y_damaged[:set_split],
                    y_muscle[:set_split],
                    y_stroma[:set_split],
                    y_uro[:set_split],
                ),
                0,
            ),
        )

    del x_blood_400x
    del x_blood_100x
    del x_blood_25x
    del y_blood
    del x_damaged_400x
    del x_damaged_100x
    del x_damaged_25x
    del y_damaged
    del x_muscle_400x
    del x_muscle_100x
    del x_muscle_25x
    del y_muscle
    del x_stroma_400x
    del x_stroma_100x
    del x_stroma_25x
    del y_stroma
    del x_uro_400x
    del x_uro_100x
    del x_uro_25x
    del y_uro
    gc.collect()
    print("Set generated")

    # Creating TF Datasets for faster prefetching and parallelization
    current_set = tf.data.Dataset.from_tensor_slices(
        ({"input_400x": x_400x, "input_100x_100x": x_100x, "input_25x_25x": x_25x}, y))

    del x_400x
    del x_100x
    del x_25x
    del y
    gc.collect()

    if is_it_training:
        pool_blood = tf.data.Dataset.from_tensor_slices(
            ({"input_400x": x_pool_blood_400x, "input_100x_100x": x_pool_blood_100x, "input_25x_25x": x_pool_blood_25x},
             y_pool_blood)
        )
        pool_damaged = tf.data.Dataset.from_tensor_slices(
            ({"input_400x": x_pool_damaged_400x, "input_100x_100x": x_pool_damaged_100x,
              "input_25x_25x": x_pool_damaged_25x}, y_pool_damaged)
        )
        pool_muscle = tf.data.Dataset.from_tensor_slices(
            ({"input_400x": x_pool_muscle_400x, "input_100x_100x": x_pool_muscle_100x,
              "input_25x_25x": x_pool_muscle_25x}, y_pool_muscle)
        )
        pool_stroma = tf.data.Dataset.from_tensor_slices(
            ({"input_400x": x_pool_stroma_400x, "input_100x_100x": x_pool_stroma_100x,
              "input_25x_25x": x_pool_stroma_25x}, y_pool_stroma)
        )
        pool_uro = tf.data.Dataset.from_tensor_slices(
            ({"input_400x": x_pool_uro_400x, "input_100x_100x": x_pool_uro_100x, "input_25x_25x": x_pool_uro_25x},
             y_pool_uro)
        )

        del x_pool_blood_400x
        del x_pool_blood_100x
        del x_pool_blood_25x
        del y_pool_blood
        del x_pool_damaged_400x
        del x_pool_damaged_100x
        del x_pool_damaged_25x
        del y_pool_damaged
        del x_pool_muscle_400x
        del x_pool_muscle_100x
        del x_pool_muscle_25x
        del y_pool_muscle
        del x_pool_stroma_400x
        del x_pool_stroma_100x
        del x_pool_stroma_25x
        del y_pool_stroma
        del x_pool_uro_400x
        del x_pool_uro_100x
        del x_pool_uro_25x
        del y_pool_uro
        gc.collect()

        print("Pools of dataset generated")

        return current_set, pool_blood, pool_damaged, pool_muscle, pool_stroma, pool_uro
    else:
        return current_set, None, None, None, None, None


# Helper function for merging new history objects with older ones
def append_history(losses, val_losses, accuracy, val_accuracy, history):
    losses = losses + history.history["loss"]
    val_losses = val_losses + history.history["val_loss"]
    accuracy = accuracy + history.history["accuracy"]
    val_accuracy = val_accuracy + history.history["val_accuracy"]
    return losses, val_losses, accuracy, val_accuracy


# Plotter function
def plot_history(losses, val_losses, accuracies, val_accuracies, prev_epoch_stops, run_count):
    plt.figure(0)
    plt.plot(losses)
    plt.plot(val_losses)
    if len(prev_epoch_stops) > 1:
        for iter_change in prev_epoch_stops:
            plt.axvline(x=iter_change, color="red", linestyle='--', linewidth=1)
    plt.legend(["train_loss", "val_loss"])
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.savefig("al_manuscript_models/loss_" + run_count + ".png")
    plt.close()

    plt.figure(1)
    plt.plot(accuracies)
    plt.plot(val_accuracies)
    if len(prev_epoch_stops) > 1:
        for iter_change in prev_epoch_stops:
            plt.axvline(x=iter_change, color="red", linestyle='--', linewidth=1)
    plt.legend(["train_accuracy", "val_accuracy"])
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig("al_manuscript_models/acc_" + run_count + ".png")
    plt.close()


# Retrieve TRI-CNN model architecture
def get_tri_scale_model(trainable):
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

    # Freeze all convolutional layers
    if trainable is False:
        for layer in base_model_400x.layers:
            layer.trainable = False

        for layer in base_model_100x.layers:
            layer.trainable = False

        for layer in base_model_25x.layers:
            layer.trainable = False

    # Concatenate models
    flatten_layer = keras.layers.concatenate([last_layer_400x, last_layer_100x, last_layer_25x], axis=-1)

    # Define new classifier architecture
    my_dense = Dense(4096, activation='relu', name='my_dense1')(flatten_layer)
    my_dense = Dropout(rate=0.4, noise_shape=None, seed=None)(my_dense)
    my_dense = Dense(4096, activation='relu', name='my_dense2')(my_dense)
    my_dense = Dropout(rate=0.4, noise_shape=None, seed=None)(my_dense)

    # OUTPUT CLASSIFIER LAYER
    my_output = Dense(6, activation='softmax', name='my_output')(my_dense)

    # Define the models
    TL_classifier_model = Model(inputs=[image_input_400x, image_input_100x, image_input_25x], outputs=my_output)

    return TL_classifier_model


# Train model via Active Learning
def train_active_learning_model(
        train_dataset,
        num_train_samples,
        pool_blood,
        pool_damaged,
        pool_muscle,
        pool_stroma,
        pool_uro,
        val_dataset,
        num_val_samples,
        test_dataset,
        num_test_samples,
        num_iterations,
        sampling_size,
        run_count
):
    # Creating lists for storing metrics
    losses, val_losses, accuracies, val_accuracies, prev_epoch_stops = [], [], [], [], []
    acc_loss_epochs = dict()  # Dictionary for preserving info about losses, accuracy and when did the change of iteration happened

    missclasification_per_iteration = dict()  # Store misspredictions throughout the iteration process
    missclasification_per_iteration['num_iterations'] = num_iterations
    missclasification_per_iteration['num_train_samples'] = num_train_samples
    missclasification_per_iteration['num_val_samples'] = num_val_samples
    missclasification_per_iteration['sampling_size'] = sampling_size

    test_results = dict()  # Dictionary with test results
    test_results['num_test_samples'] = num_test_samples

    model = get_tri_scale_model(trainable=True)

    initial_weights = [layer.get_weights() for layer in model.layers]
    model.load_weights(weights_filename, by_name=True)
    for layer, initial in zip(model.layers, initial_weights):
        weights = layer.get_weights()
        if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
            print(f'Checkpoint contained no weights for layer {layer.name}!')

    # Defining checkpoints.
    # The checkpoint callback is reused throughout the training since it only saves the best overall model.
    checkpoint = keras.callbacks.ModelCheckpoint(
        "al_manuscript_models/AL_Model_" + run_count + ".h5", monitor='val_loss', save_best_only=True,
        save_weights_only=True, verbose=1
    )
    # Here, patience is set to 4. This can be set higher if desired.
    early_stopping = keras.callbacks.EarlyStopping(patience=3, verbose=1)

    # Augmentation techniques to be applied for the training set
    trainAug = Sequential([
        preprocessing.RandomFlip("horizontal_and_vertical"),
        preprocessing.RandomRotation(0.3)
    ])

    # We will monitor the false positives and false negatives predicted by our model
    # These will decide the subsequent sampling ratio for every Active Learning loop
    model.compile(
        loss="categorical_crossentropy",
        optimizer=model_optimizer,
        metrics=[
            'accuracy'
        ],
    )

    # Initial fit with a small subset of the training set
    print("-" * 100)
    print("Iteration 0")
    history = model.fit(
        train_dataset.cache().shuffle(500000).batch(BATCH_SIZE),  # .map(lambda x, y: (trainAug(x), y)),
        epochs=EPOCHS,
        validation_data=val_dataset.batch(BATCH_SIZE),
        callbacks=[checkpoint, early_stopping])

    # Appending the history
    losses, val_losses, accuracies, val_accuracies = append_history(
        losses, val_losses, accuracies, val_accuracies, history
    )

    # Plot history
    plot_history(
        losses,
        val_losses,
        accuracies,
        val_accuracies,
        prev_epoch_stops,
        run_count
    )

    # Extract val and test labels for computing metrics
    _, val_labels = tuple(zip(*val_dataset))
    val_labels = tf.math.argmax(val_labels, 1)

    _, test_labels = tuple(zip(*test_dataset))
    test_labels = tf.math.argmax(test_labels, 1)

    # Unfreeze base model for iterations
    # for layer in model.layers: layer.trainable = True

    for iteration in range(num_iterations):

        # Initialize negative counters
        negative_blood, negative_damaged, negative_muscle, negative_stroma, negative_uro = 0, 0, 0, 0, 0

        # Getting predictions from previously trained model
        predictions = model.predict(val_dataset.batch(BATCH_SIZE))

        # Generating labels from the output probabilities
        pred_max = tf.math.argmax(predictions, 1)

        for true_label, predicted_label in zip(val_labels, pred_max):
            is_it_equal = tf.math.equal(true_label, predicted_label)

            if not tf.get_static_value(is_it_equal):
                actual_class = tf.get_static_value(true_label)

                if actual_class == 1:
                    negative_blood += 1
                elif actual_class == 2:
                    negative_damaged += 1
                elif actual_class == 3:
                    negative_muscle += 1
                elif actual_class == 4:
                    negative_stroma += 1
                elif actual_class == 5:
                    negative_uro += 1

        # Evaluating the number of zeros and ones incorrrectly classified
        print("-" * 100)
        print(
            f"Number of blood tiles incorrectly classified: {negative_blood}"
        )
        print(
            f"Number of damaged tiles incorrectly classified: {negative_damaged}"
        )
        print(
            f"Number of muscle tiles incorrectly classified: {negative_muscle}"
        )
        print(
            f"Number of stroma tiles incorrectly classified: {negative_stroma}"
        )
        print(
            f"Number of uro tiles incorrectly classified: {negative_uro}"
        )

        # Save history of missprediction on validation set
        missclasification_per_iteration[iteration] = dict()
        missclasification_per_iteration[iteration]['blood'] = negative_blood
        missclasification_per_iteration[iteration]['damaged'] = negative_damaged
        missclasification_per_iteration[iteration]['muscle'] = negative_muscle
        missclasification_per_iteration[iteration]['stroma'] = negative_stroma
        missclasification_per_iteration[iteration]['uro'] = negative_uro

        # This technique of Active Learning demonstrates ratio based sampling where
        # Number of ones/zeros to sample = Number of ones/zeros incorrectly classified / Total incorrectly classified
        if negative_blood != 0 or negative_damaged != 0 or negative_muscle != 0 or negative_stroma != 0 or negative_uro != 0:
            total = negative_blood + negative_damaged + negative_muscle + negative_stroma + negative_uro
            sample_ratio_blood, sample_ratio_damaged, sample_ratio_muscle, sample_ratio_stroma, sample_ratio_uro = (
                negative_blood / total,
                negative_damaged / total,
                negative_muscle / total,
                negative_stroma / total,
                negative_uro / total,
            )
        # In the case where all samples are correctly predicted, we can sample both classes equally
        else:
            sample_ratio_blood, sample_ratio_damaged, sample_ratio_muscle, sample_ratio_stroma, sample_ratio_uro = 0.2, 0.2, 0.2, 0.2, 0.2

        print(
            f"Sample ratio for blood: {sample_ratio_blood}"
        )
        print(
            f"Sample ratio for damaged: {sample_ratio_damaged}"
        )
        print(
            f"Sample ratio for muscle: {sample_ratio_muscle}"
        )
        print(
            f"Sample ratio for stroma: {sample_ratio_stroma}"
        )
        print(
            f"Sample ratio for uro: {sample_ratio_uro}"
        )

        # Sample the required number of ones and zeros
        sampled_dataset = pool_blood.take(int(sample_ratio_blood * sampling_size)
                                          ).concatenate(pool_damaged.take(int(sample_ratio_damaged * sampling_size))
                                                        ).concatenate(
            pool_muscle.take(int(sample_ratio_muscle * sampling_size))
            ).concatenate(pool_stroma.take(int(sample_ratio_stroma * sampling_size))
                          ).concatenate(pool_uro.take(int(sample_ratio_uro * sampling_size)))

        # Skip the sampled data points to avoid repetition of sample
        pool_blood = pool_blood.skip(int(sample_ratio_blood * sampling_size))
        pool_damaged = pool_damaged.skip(int(sample_ratio_damaged * sampling_size))
        pool_muscle = pool_muscle.skip(int(sample_ratio_muscle * sampling_size))
        pool_stroma = pool_stroma.skip(int(sample_ratio_stroma * sampling_size))
        pool_uro = pool_uro.skip(int(sample_ratio_uro * sampling_size))

        # Concatenating the train_dataset with the sampled_dataset
        train_dataset = train_dataset.concatenate(sampled_dataset).prefetch(
            int(sampling_size * 2)
        )

        print("-" * 100)

        # We recompile the model to reset the optimizer states and retrain the model
        model.compile(
            loss="categorical_crossentropy",
            optimizer=model_optimizer,
            metrics=[
                'accuracy'
            ],
        )

        print("Iteration {}".format(iteration + 1))

        history = model.fit(
            train_dataset.cache().shuffle(500000).batch(BATCH_SIZE),
            validation_data=val_dataset.batch(BATCH_SIZE),
            epochs=EPOCHS,
            callbacks=[checkpoint, early_stopping])

        # Appending the history
        prev_epoch_stops.append(len(losses))
        losses, val_losses, accuracies, val_accuracies = append_history(
            losses, val_losses, accuracies, val_accuracies, history
        )

        # Plot history
        plot_history(
            losses,
            val_losses,
            accuracies,
            val_accuracies,
            prev_epoch_stops,
            run_count
        )

        # Loading the best model from this training loop
        model.load_weights("al_manuscript_models/AL_Model_" + run_count + ".h5", by_name=True)

    # Save info on loss_acc_epochs dictionary
    acc_loss_epochs['losses'] = losses
    acc_loss_epochs['val_losses'] = val_losses
    acc_loss_epochs['accuracies'] = accuracies
    acc_loss_epochs['val_accuracies'] = val_accuracies
    acc_loss_epochs['prev_epoch_stops'] = prev_epoch_stops
    with open('acc_loss_epochs.obj', 'wb') as handle:
        pickle.dump(acc_loss_epochs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save info on missclassifications dictionary
    with open('al_manuscript_models/missclasification_per_iteration_' + run_count + '.obj', 'wb') as handle:
        pickle.dump(missclasification_per_iteration, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Plotting the overall history and evaluating the final model
    print("-" * 100)
    print(
        "Test set evaluation: ",
        model.evaluate(test_dataset.batch(BATCH_SIZE), verbose=0, return_dict=True),
    )
    print("-" * 100)
    # Getting predictions from previously trained model
    predictions = model.predict(test_dataset.batch(BATCH_SIZE))

    # Generating labels from the output probabilities
    pred_max = tf.math.argmax(predictions, 1)

    # Initialize negative counters
    negative_blood, negative_damaged, negative_muscle, negative_stroma, negative_uro = 0, 0, 0, 0, 0

    for true_label, predicted_label in zip(test_labels, pred_max):
        is_it_equal = tf.math.equal(true_label, predicted_label)

        if not tf.get_static_value(is_it_equal):
            actual_class = tf.get_static_value(true_label)

            if actual_class == 1:
                negative_blood += 1
            elif actual_class == 2:
                negative_damaged += 1
            elif actual_class == 3:
                negative_muscle += 1
            elif actual_class == 4:
                negative_stroma += 1
            elif actual_class == 5:
                negative_uro += 1

    # Evaluating the number of zeros and ones incorrrectly classified
    print("-" * 100)
    print(
        f"Number of blood tiles incorrectly classified: {negative_blood}"
    )
    print(
        f"Number of damaged tiles incorrectly classified: {negative_damaged}"
    )
    print(
        f"Number of muscle tiles incorrectly classified: {negative_muscle}"
    )
    print(
        f"Number of stroma tiles incorrectly classified: {negative_stroma}"
    )
    print(
        f"Number of uro tiles incorrectly classified: {negative_uro}"
    )
    print("-" * 100)

    test_results['predictions'] = predictions
    test_results['blood'] = negative_blood
    test_results['damaged'] = negative_damaged
    test_results['muscle'] = negative_muscle
    test_results['stroma'] = negative_stroma
    test_results['uro'] = negative_uro

    with open('al_manuscript_models/test_results_' + run_count + '.obj', 'wb') as handle:
        pickle.dump(test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return model


"""
## MAIN
"""

# Training parameters
TILE_SIZE = 128
BATCH_SIZE = 128
EPOCHS = 10
DEBUG_MODE = False
num_iterations = 4
weights_filename = 'AL_MODEL_WEIGHTS.h5'
model_optimizer = tf.keras.optimizers.SGD(lr=0.000015, decay=1e-6, momentum=0.9, nesterov=True)

# Dataset parameters
val_split = 2500
test_split = 2500
train_split = 25000
LIMIT_NUMBER_OF_SAMPLES = 1000  # It does not apply to blood since it is heavily underrepresented
sampling_size = 20000
dataset_split_train = 0.8
dataset_split_val = 0.1
tissue_names_dict = my_constants.get_tissue_name_and_index_of_classes()
tile_files_path = 'DATASET_DIR/'
list_of_tfrecord_files = [tile_files_path + f for f in os.listdir(tile_files_path)]

if DEBUG_MODE:
    list_of_tfrecord_files = list_of_tfrecord_files[:30]
    val_split = int(val_split * 0.2)
    test_split = int(test_split * 0.2)
    train_split = int(train_split * 0.2)
    num_iterations = int(num_iterations * 0.2)
    sampling_size = int(sampling_size * 0.2)

# WSI-based partition of the dataset
list_of_tfrecord_files_train = list_of_tfrecord_files[:int(len(list_of_tfrecord_files) * dataset_split_train)]
list_of_tfrecord_files_val = list_of_tfrecord_files[int(len(list_of_tfrecord_files) * dataset_split_train):int(
    len(list_of_tfrecord_files) * (dataset_split_train + dataset_split_val))]
list_of_tfrecord_files_test = list_of_tfrecord_files[
                              int(len(list_of_tfrecord_files) * (dataset_split_train + dataset_split_val)):]

print('Number of WSI: {} slides'.format(len(list_of_tfrecord_files)))
print('Training set: {} slides'.format(len(list_of_tfrecord_files_train)))
print('Validation set: {} slides'.format(len(list_of_tfrecord_files_val)))
print('Test set: {} slides'.format(len(list_of_tfrecord_files_test)))
print('-------------------')

# Generate datasets
train_dataset, pool_blood, pool_damaged, pool_muscle, pool_stroma, pool_uro = generate_set(list_of_tfrecord_files_train,
                                                                                           train_split, True)
val_dataset, _, _, _, _, _ = generate_set(list_of_tfrecord_files_val, val_split, False)
test_dataset, _, _, _, _, _ = generate_set(list_of_tfrecord_files_test, test_split, False)

# Iterate over the active learning framework
for run_count in range(num_iterations):
    active_learning_model = train_active_learning_model(
        train_dataset,
        train_split,
        pool_blood,
        pool_damaged,
        pool_muscle,
        pool_stroma,
        pool_uro,
        val_dataset,
        val_split,
        test_dataset,
        test_split,
        num_iterations,
        sampling_size,
        str(run_count)
    )

# Calculate elapse time for current run
elapse_time = time.time() - script_start_time
m, s = divmod(elapse_time, 60)
h, m = divmod(m, 60)
model_time = '%02d:%02d:%02d' % (h, m, s)
print('Finished. Duration: {}'.format(model_time))
