import cv2
import numpy as np
import sys

#add this to python path, since weights are there
sys.path.append('/trash/Project_Trash_Mask_RCNN/Mask-RCNN/Mask_RCNN-master/')
print(sys.path)

def random_colors(N):
    np.random.seed(1)
    colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
    return colors


def apply_mask(image, mask, color, alpha=0.5):
    """apply mask to image"""
    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask == 1,
            image[:, :, n] * (1 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def display_instances(image, boxes, masks, ids, names, scores):
    """
        take the image and results and apply the mask, box, and Label
    """
    n_instances = boxes.shape[0]
    colors = random_colors(n_instances)

    if not n_instances:
        print('NO INSTANCES TO DISPLAY')
    else:
        assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]

    for i, color in enumerate(colors):
        # we want the colours to only ne in one color: SIFR orange ff5722
        # color = (255, 87, 34)
        if not np.any(boxes[i]):
            continue

        y1, x1, y2, x2 = boxes[i]
        label = names[ids[i]]
        score = scores[i] if scores is not None else None
        caption = '{} {:.2f}'.format(label, score) if score else label
        mask = masks[:, :, i]

        image = apply_mask(image, mask, color)
        image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        image = cv2.putText(
            image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2
        )

    return image


if __name__ == '__main__':
    """
        test everything
    """
    import os
    import sys

    # adding these to path 
    sys.path.append('/trash/Project_Trash_Mask_RCNN/Mask-RCNN/Mask_RCNN-master/')
    sys.path.append('Project_Trash_Mask_RCNN/Mask-RCNN/Mask_RCNN-master')

    #from samples.trash import trash
    import trash
    from mrcnn import model as modellib, utils
    #import utils
    #import model as modellib
    

    batch_size = 1

    ROOT_DIR = os.path.abspath("../../")
    MODEL_DIR = os.path.join(ROOT_DIR, "logs")
    IMAGES_DIR = os.path.join(ROOT_DIR, "predictions")
    VIDEO_DIR = os.path.join(ROOT_DIR, "videos")
    VIDEO_SAVE_DIR = os.path.join(ROOT_DIR, "detected")
    MODEL_PATH = "weights/mask_rcnn_trash_0200_030519_large.h5"

    config = trash.TrashConfig()

    class InferenceConfig(config.__class__):
        # Run detection on one image at a time
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1

    config = InferenceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="inference", model_dir=MODEL_DIR, config=config
    )
    weights_path = os.path.join(ROOT_DIR, MODEL_PATH)
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    class_names = [
        'BG', 'trash'
    ]

    #capture = cv2.VideoCapture(os.path.join(VIDEO_DIR, 'Trash_short.mp4'))

    #try:
    #   if not os.path.exists(VIDEO_SAVE_DIR):
    #       os.makedirs(VIDEO_SAVE_DIR)
    #except OSError:
    #   print ('Error: Creating directory of data')

    images = []
    img_count = 0
    # these 2 lines can be removed if you dont have a 1080p camera.
    #capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    #capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    import glob
    import os
    print(os.getcwd())
    frames = glob.glob("{}/*.jpg".format(IMAGES_DIR))
    img_count = len(frames)
    print('image count: ')
    print(img_count)
    print(frames)
    import pandas as pd
    df = pd.DataFrame()

    import skimage
    i = 0
    for image in frames:
        filename = image.split('/')[-1]
        df.at[i, 'file'] = filename
        try:
            print(image)
            image = skimage.io.imread('{}'.format(image))
            results = model.detect([image], verbose=0)

            # Display results
            #ax = get_ax(1)
            r = results[0]
            df.at[i, 'trash'] = len(r['class_ids'])
            scores = []
            for val in r['scores']:
                scores.append(val)
            df.at[i, 'conficences'] = str(scores)

            # images with trash detected:
            if len(scores) != 0:
                detected_img = display_instances(image, r['rois'], r['masks'], 
                                r['class_ids'], class_names, r['scores'])

                name = os.path.join(ROOT_DIR, os.path.join(VIDEO_SAVE_DIR, filename))
                detected_img = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
                cv2.imwrite(name, detected_img)
                print('img saved')

            i += 1
        except Exception as e:
            print(e)
            i += 1

    df.to_csv('../trash_csv_sample.csv', index=False)

