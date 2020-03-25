#!/usr/bin/env python
# -*- coding: utf-8 -*- 

"""
Mask R-CNN
Train on the toy Balloon dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Original Balloon code Written by Waleed Abdulla

This implementation of trash.py is based on the balloon example, and is developed by SIFR.AI

------------------------------------------------------------

"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw

# tingitud errorist: OMP: Error #15: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "weights/mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class TrashConfig(Config):
    """Configuration for training on the toy  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "trash"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + trash

    # Number of training steps per epoch
    # We have tried 5, now let's up the game to 10 
    STEPS_PER_EPOCH = 17

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.95


############################################################
#  Dataset
############################################################

#dataset_dir = '/Project_Trash_Mask_RCNN/Mask-RCNN/Mask_RCNN-master/samples/trash/dataset/train'
class TrashDataset(utils.Dataset):

    def load_trash(self, dataset_dir, subset):
        # Changed from the original, because my json was different
            self.add_class("trash", 1, "trash")

            # Train or validation dataset?
            assert subset in ["train", "val"]
            dataset_dir = os.path.join(dataset_dir, subset)
 
            import glob
            all_jsons = glob.glob("{}/*.json".format(dataset_dir))
            for json_file in all_jsons:
                print(json_file)
                #annotations = json.load(open(os.path.join(dataset_dir, "annotation_project_trash_test.json")))
                annotations = json.load(open(json_file)) 

                annotations = list(annotations.values()) # don't need the dict keys
                for a in annotations:
                    for image_file in a.values():
                        print(image_file)
                        print(image_file.get('filename'))
                        polygons = []
                        for annotation_element in image_file:
                            if annotation_element == 'regions':
                                # need on kõik ühe pildi regioonid
                                for shape in image_file[annotation_element]:
                                    try:
                                        shape = shape.get('shape_attributes')
                                        polygons.append(shape)
                                    except Exception as e:
                                        print(e)
                                        print('annotation inclides other shapes than polygon')
                        try:
                            image_path = os.path.join(dataset_dir, image_file.get('filename'))
                            print(image_path)
                            image = skimage.io.imread(image_path)
                            height, width = image.shape[:2]
                            print(height, width)
    
                            self.add_image("trash", image_id=image_file.get('filename'),  # use file name as a unique image id
                                    path=image_path,
                                    width=width, height=height,
                                    polygons=polygons)
                        except Exception as e: #sellest, et kõiki pilte ei ole train kaustas.. 
                            print(e)   
                        #print('')

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a trash dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "trash":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "trash":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = TrashDataset()
    dataset_train.load_trash(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = TrashDataset()
    dataset_val.load_trash(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                # We had epoch 2, now 10
                epochs=200,
                layers='heads')



############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect trash.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/trash/dataset/",
                        help='Directory of the Trash dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = TrashConfig()
    else:
        class InferenceConfig(TrashConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    else:
        print("'{}' is not recognized. "
              "Use 'train'".format(args.command))
