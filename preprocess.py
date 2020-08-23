from PIL import Image
import numpy as np
import os
import glob
import tqdm
import argparse
from utils.hparams import HParam

def load_image( infilename) :
    img = Image.open( infilename )
    img.load()
    if img.mode == 'P':
        img.convert('RGB')
    data = np.asarray( img, dtype="int32" )
    return data

def start_points(size, split_size, overlap=0):
    points = [0]
    stride = int(split_size * (1-overlap))
    counter = 1
    while True:
        pt = stride * counter
        if pt + split_size >= size:
            points.append(size - split_size)
            break
        else:
            points.append(pt)
        counter += 1
    return points

def crop_image_mask(image_dir, mask_dir, mask_path, X_points, Y_points, split_height=224, split_width=224):
    img_id = os.path.basename(mask_path).split(".")[0]
    mask = load_image(mask_path)
    img = load_image(mask_path.replace("output", "input"))

    count = 0
    num_skipped = 1
    for i in Y_points:
        for j in X_points:
            new_image = img[i:i + split_height, j:j + split_width]
            new_mask = mask[i:i + split_height, j:j + split_width]
            new_mask[new_mask > 1] = 255
            # Skip any Image that is more than 99% empty.
            if np.any(new_mask):
                num_black_pixels, num_white_pixels = np.unique(new_mask, return_counts=True)[1]

                if num_white_pixels / num_black_pixels < 0.01:
                    num_skipped += 1
                    continue

            mask_ = Image.fromarray(new_mask.astype(np.uint8))
            mask_.save("{}/{}_{}.jpg".format(mask_dir, img_id, count), "JPEG")
            im = Image.fromarray(new_image.astype(np.uint8))
            im.save("{}/{}_{}.jpg".format(image_dir, img_id, count), "JPEG")
            count = count + 1

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True,
                        help="yaml file for configuration")
    parser.add_argument('-t', '--train', type=str, required=True,
                        help="Training Folder.")
    parser.add_argument('-v', '--valid', type=str, required=True,
                        help="Validation Folder")
    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, 'r') as f:
        hp_str = ''.join(f.readlines())

    train_dir = args.train
    valid_dir = args.valid
    X_points = start_points(hp.IMAGE_SIZE, hp.CROP_SIZE, 0.14)
    Y_points = start_points(hp.IMAGE_SIZE, hp.CROP_SIZE, 0.14)

    ## Training data
    train_img_dir = os.path.join(train_dir, "input")
    train_mask_dir = os.path.join(train_dir, "output")
    train_img_crop_dir = os.path.join(hp.train, "input_crop")
    os.makedirs(train_img_crop_dir, exist_ok=True)
    train_mask_crop_dir = os.path.join(hp.train, "mask_crop")
    os.makedirs(train_mask_crop_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(train_img_dir, '**', '*.png'), recursive=True)
    mask_files = glob.glob(os.path.join(train_mask_dir, '**', '*.png'), recursive=True)
    print("Length of image :", len(img_files))
    print("Length of mask :", len(mask_files))
    #assert len(img_files) == len(mask_files)



    for mask_path in tqdm.tqdm(mask_files, desc='Cropping Training images'):
        crop_image_mask(train_img_crop_dir, train_mask_crop_dir, mask_path, X_points, Y_points)

    ### Validation data
    valid_img_dir = os.path.join(valid_dir, "input")
    valid_mask_dir = os.path.join(valid_dir, "output")
    valid_img_crop_dir = os.path.join(hp.valid, "input_crop")
    os.makedirs(valid_img_crop_dir, exist_ok=True)
    valid_mask_crop_dir = os.path.join(hp.valid, "mask_crop")
    os.makedirs(valid_mask_crop_dir, exist_ok=True)

    img_files = glob.glob(os.path.join(valid_img_dir, '**', '*.png'), recursive=True)
    mask_files = glob.glob(os.path.join(valid_mask_dir, '**', '*.png'), recursive=True)
    assert len(img_files) == len(mask_files)


    for mask_path in tqdm.tqdm(mask_files, desc='Cropping Validation images'):
        crop_image_mask(valid_img_crop_dir, valid_mask_crop_dir, mask_path, X_points, Y_points)




