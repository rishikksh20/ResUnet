import hparams as hp
from PIL import Image
import shutil
import os
import glob
import tqdm

if __name__ == "__main__":
    train_dir = hp.train
    valid_dir = hp.valid

    train_mask_crop_dir = os.path.join(train_dir, "mask_crop")
    mask_files = glob.glob(
        os.path.join(train_mask_crop_dir, "**", "*.jpg"), recursive=True
    )

    noisy_mask_files = os.path.join(train_dir, "noisy")
    os.makedirs(noisy_mask_files, exist_ok=True)

    count = 0
    print("Total image: ", len(mask_files))
    for f in mask_files:
        img = Image.open(f)
        img.load()
        extrema = img.convert("L").getextrema()
        if extrema == (0, 0):
            count = count + 1
            shutil.copy2(f, f.replace("mask_crop", "noisy"))
            ## If file exists, delete it ##
            if os.path.isfile(f):
                os.remove(f)
            else:  ## Show an error ##
                print("Error: %s file not found" % f)

    print("USeless image: ", count)
