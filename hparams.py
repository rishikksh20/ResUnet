train = 'H:\\Deepsync\\backup\\development\\ResUnet\\data\\road_segmentation_ideal\\training'
valid = "H:\\Deepsync\\backup\\development\\ResUnet\\data\\road_segmentation_ideal\\testing"
log = "logs"
logging_step = 100
validation_interval = 2000 # Save and valid have same interval
checkpoints = "checkpoints"

batch_size = 4
lr = 0.001

IMAGE_SIZE = 1500
CROP_SIZE = 224