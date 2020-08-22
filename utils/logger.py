# adapted from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np

class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, dice_loss, iou, step):
        self.add_scalar('training/dice_loss', dice_loss, step)
        self.add_scalar('training/iou', iou, step)

    def log_validation(self, dice_loss, iou, step):
        self.add_scalar('validation/dice_loss', dice_loss, step)
        self.add_scalar('validation/iou', iou, step)


    def log_images(self, map, target, prediction, step):
        if len(map.shape) > 3:
            map = map.squeeze(0)
        if len(target.shape) > 2:
            target = target.squeeze()
        if len(prediction.shape) > 2:
            prediction = prediction.squeeze()
        self.add_image('map', Image.fromarray(map.astype(np.uint8)), step)
        self.add_image('mask', Image.fromarray(target.astype(np.uint8)), step)
        self.add_image('prediction', Image.fromarray(prediction.astype(np.uint8)), step)
