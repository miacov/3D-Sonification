import os
import shutil
import json
import cv2
import numpy as np
import pandas as pd
from keras import layers, models, optimizers
from keras.preprocessing.image import ImageDataGenerator


def create_binary_mask(annotation, frame):
    """
    Creates a binary mask for a frame with value 1 where a bounding box is indicated in annotations, and 0 on all
    the other spots.

    :param annotation: annotation row for frame from annotations dataframe
    :param frame: frame image to get dimensions for mask
    :return: returns the binary mask
    """
    # create empty mask
    mask = np.zeros(frame.shape[:2], dtype=np.uint8)

    # parse json-like label column with bounding boxes
    try:
        bounding_boxes = json.loads(annotation["label"])

        # create rectangles with value 1 on mask for each bounding box, all other spots will be 0 on mask
        for bbox in bounding_boxes:
            image_width = bbox["original_width"]
            image_height = bbox["original_height"]

            # absolute positions
            x = int(bbox["x"]*image_width/100)
            y = int(bbox["y"]*image_height/100)

            # absolute width, height
            width = int(bbox["width"]*image_width/100)
            height = int(bbox["height"]*image_height/100)

            cv2.rectangle(mask, (x, y), (x + width, y + height), 1, -1)
    except TypeError:
        pass

    # print mask
    #cv2.imshow("Mask", mask)
    #cv2.waitKey(0)

    return mask


def process_extracted_video_frames(video_name, output_masks=True):
    """
    Creates training y data for model using extracted video frames. Each produced frame will be according to an
    annotation file with bounding boxes indicating the objects.

    :param video_name: video name to get frames
    :param output_masks: outputs frame mask for each frame if True
    :return: returns frames and masks to use for model training
    """
    frame_folder = os.path.join("frames", video_name.replace(".mp4", ""))

    try:
        annotations = pd.read_csv(os.path.join(frame_folder, "Annotations.csv"))

        frames = []
        masks = []

        frame_folder_plain = os.path.join(frame_folder, "plain")
        frame_folder_classifier = os.path.join(frame_folder, "classifier")

        if os.path.exists(frame_folder_classifier):
            shutil.rmtree(frame_folder_classifier)
        os.makedirs(frame_folder_classifier)

        for idx, annotation in annotations.iterrows():
            if not np.isnan(annotation["id"]):
                frame_name = annotation["image"].rsplit("-", 1)[1]
                frame = cv2.imread(os.path.join(frame_folder_plain, frame_name))

                mask = create_binary_mask(annotation, frame)

                # save the mask as an image
                if output_masks:
                    cv2.imwrite(os.path.join(frame_folder_classifier, frame_name), mask)

                frames.append(frame)
                masks.append(mask)
    except FileNotFoundError:
        return None, None

    return frames, masks


def augment_data(data):
    """
    Augments training data with rotations, flips, etc.

    :param data: x train data
    :return: returns data augmentor fitted on data
    """
    data_augmentor = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    data_augmentor.fit(data)

    return data_augmentor


def create_model(input_shape):
    """
    Create a binary classification model to classify a colored image into a binary mask of 0s and 1s.

    :param input_shape: input image shape (height, width, channels)
    :return: returns model.
    """
    # input layer
    inputs = layers.Input(shape=input_shape)

    # downsampling
    downsample = layers.Conv2D(32, (3, 3), padding="same")(inputs)
    downsample = layers.BatchNormalization()(downsample)
    downsample = layers.Activation("relu")(downsample)
    downsample_pool = layers.MaxPooling2D((2, 2), strides=(2, 2))(downsample)

    # middle layer
    middle = layers.Conv2D(64, (3, 3), padding="same")(downsample_pool)
    middle = layers.BatchNormalization()(middle)
    middle = layers.Activation("relu")(middle)

    # upsampling
    upsample = layers.UpSampling2D((2, 2))(middle)
    upsample = layers.concatenate([downsample, upsample], axis=3)
    upsample = layers.Activation("relu")(upsample)
    upsample = layers.Conv2D(32, (3, 3), padding="same")(upsample)
    upsample = layers.BatchNormalization()(upsample)
    upsample = layers.Activation("relu")(upsample)

    # output sigmoid layer for binary classification
    output = layers.Conv2D(1, (1, 1), activation="sigmoid")(upsample)

    # Create the model
    model = models.Model(inputs=inputs, outputs=output)

    # compile the model
    model.compile(optimizer=optimizers.Adam(learning_rate=1e-4),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])

    return model


def data_generator(all_frames, all_masks, batch_size):
    """
    Generates data pairs in batches to reduce memory load.

    :param all_frames: x data
    :param all_masks: y data
    :param batch_size: batch size
    :return: yields data generator object
    """
    num_samples = len(all_frames)
    while True:
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_indices = indices[start:end]
            batch_frames = [all_frames[i] for i in batch_indices]
            batch_masks = [all_masks[i] for i in batch_indices]
            yield np.array(batch_frames), np.array(batch_masks)


if __name__ == "__main__":
    all_frames = []
    all_masks = []

    # get frames for training from all videos
    """
    videos = pd.read_csv("videos.csv")
    for idx, video in videos.iterrows():
        print("Processing video: " + video["Video Name"])
        frames, masks = process_extracted_video_frames(video["Video Name"])

        if frames is not None:
            all_frames.extend(frames)
            all_masks.extend(masks)
    """

    # get frames for training for selected videos
    frames, masks = process_extracted_video_frames("A_Flight_to_HCG_40.mp4", output_masks=False)

    # annotation for video exists
    if frames is not None:
        all_frames.extend(frames)
        all_masks.extend(masks)

    frames, masks = process_extracted_video_frames("Flight_to_AG_Carinae.mp4", output_masks=False)

    if frames is not None:
        all_frames.extend(frames)
        all_masks.extend(masks)

    # create the model
    dimensions = all_frames[0].shape[:2]  # (height, width)
    input_shape = (dimensions[0], dimensions[1], 3)
    model = create_model(input_shape)

    # generate data pairs to reduce memory usage
    batch_size = 2
    epochs = 10

    train_generator = data_generator(all_frames, all_masks, batch_size)
    steps_per_epoch = len(all_frames) // batch_size

    # train the model on the whole dataset
    model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

    model.save("classifier.h5")

    # test the model on the whole dataset
    test_loss, test_accuracy = model.evaluate(train_generator, steps=steps_per_epoch)
    print("Test Loss: ", test_loss)
    print("Test Accuracy: ", test_accuracy)
