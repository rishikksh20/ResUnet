# adapted from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
from tensorboardX import SummaryWriter
from PIL import Image
import numpy as np


class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, dice_loss, iou, step):
        self.add_scalar("training/dice_loss", dice_loss, step)
        self.add_scalar("training/iou", iou, step)

    def log_validation(self, dice_loss, iou, step):
        self.add_scalar("validation/dice_loss", dice_loss, step)
        self.add_scalar("validation/iou", iou, step)

    def log_images(self, map, target, prediction, step):
        if len(map.shape) > 3:
            map = map.squeeze(0)
        if len(target.shape) > 2:
            target = target.squeeze()
        if len(prediction.shape) > 2:
            prediction = prediction.squeeze()
        self.add_image("map", map, step)
        self.add_image("mask", target.unsqueeze(0), step)
        self.add_image("prediction", prediction.unsqueeze(0), step)


class LogWriter(SummaryWriter):
    def __init__(self, logdir):
        super(LogWriter, self).__init__(logdir)

    def log_scaler(self, key, value, step, prefix="Training", helper_func=None):
        if helper_func:
            value = helper_func(value)
        self.add_scalar("{}/{}".format(prefix, key), value, step)

    def log_image(self, key, value, step, prefix="Training", helper_func=None):
        if helper_func:
            value = helper_func(value)
        self.add_image("{}/{}".format(prefix, key), value, step)
