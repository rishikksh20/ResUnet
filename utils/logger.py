# adapted from https://github.com/yunjey/pytorch-tutorial/tree/master/tutorials/04-utils/tensorboard
from tensorboardX import SummaryWriter

class MyWriter(SummaryWriter):
    def __init__(self, logdir):
        super(MyWriter, self).__init__(logdir)

    def log_training(self, dice_loss, iou, step):
        self.add_scalar('training/dice_loss', dice_loss, step)
        self.add_scalar('training/iou', iou, step)

    def log_validation(self, dice_loss, iou, map, target, prediction, step):
        self.add_scalar('validation/dice_loss', dice_loss, step)
        self.add_scalar('validation/iou', iou, step)
        self.add_image('map', map, step)
        self.add_image('mask', target, step)
        self.add_image('prediction', prediction, step)
