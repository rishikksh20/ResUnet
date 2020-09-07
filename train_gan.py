import warnings
import numpy as np
warnings.simplefilter("ignore", (UserWarning, FutureWarning))
from utils.hparams import HParam
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from dataset import dataloader
from utils.utils import get_commit_hash
from core.res_unet import ResUnet
from core.res_unet_plus import ResUnetPlusPlus
from core.multiscale import MultiScaleDiscriminator
from utils.logger import LogWriter
import torch
import argparse
import os


def main(hp, num_epochs, resume, name):

    checkpoint_dir = "{}/{}".format(hp.checkpoints, name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    os.makedirs("{}/{}".format(hp.log, name), exist_ok=True)
    writer = LogWriter("{}/{}".format(hp.log, name))
    # get model
    githash = get_commit_hash()
    if hp.RESNET_PLUS_PLUS:
        model_g = ResUnetPlusPlus(3).cuda()
    else:
        model_g = ResUnet(3, 64).cuda()

    model_d = MultiScaleDiscriminator().cuda()

    # set up binary cross entropy and dice loss
    # criterion = metrics.BCEDiceLoss()


    # optimizer
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, nesterov=True)
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    optim_g = torch.optim.Adam(model_g.parameters(),
                               lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))
    optim_d = torch.optim.Adam(model_d.parameters(),
                               lr=hp.train.adam.lr, betas=(hp.train.adam.beta1, hp.train.adam.beta2))



    # starting params
    best_loss = 999
    start_epoch = 0
    step = 0
    # optionally resume from a checkpoint
    if resume:
        if os.path.isfile(resume):

            checkpoint = torch.load(resume)
            model_g.load_state_dict(checkpoint['model_g'])
            model_d.load_state_dict(checkpoint['model_d'])
            optim_g.load_state_dict(checkpoint['optim_g'])
            optim_d.load_state_dict(checkpoint['optim_d'])
            step = checkpoint['step']
            init_epoch = checkpoint['epoch']
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # get data
    mass_dataset_train = dataloader.ImageDataset(
        hp, transform=transforms.Compose([dataloader.ToTensorTarget()])
    )

    mass_dataset_val = dataloader.ImageDataset(
        hp, False, transform=transforms.Compose([dataloader.ToTensorTarget()])
    )

    # creating loaders
    train_dataloader = DataLoader(
        mass_dataset_train, batch_size=hp.batch_size, num_workers=2, shuffle=True
    )
    val_dataloader = DataLoader(
        mass_dataset_val, batch_size=1, num_workers=2, shuffle=False
    )


    model_g.train()
    model_d.train()

    criterion_mse = torch.nn.MSELoss()
    criterion_l1 = torch.nn.L1Loss()

    for epoch in range(start_epoch, num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)

        # iterate over data
        avg_g_loss = []
        avg_d_loss = []
        avg_adv_loss = []
        loader = tqdm(train_dataloader, desc="training")
        for idx, data in enumerate(loader):

            # get the inputs and wrap in Variable
            inputs = data["sat_img"].cuda()
            labels = data["map_img"].cuda()
            start = np.random.randint(0, 512-40)

            # generator
            optim_g.zero_grad()
            fake_mel = model_g(inputs.unsqueeze(1))

            loss_g = 0.0
            loss_g = criterion_l1(fake_mel.squeeze(1), labels)
            adv_loss = 0.0
            if step > hp.train.discriminator_train_start_steps:
                disc_real = model_d(labels.unsqueeze(1),start)
                disc_fake = model_d(fake_mel, start)
                # for multi-scale discriminator

                for score_fake in disc_fake:
                    adv_loss += criterion_mse(score_fake, torch.ones_like(score_fake))
                adv_loss = adv_loss / len(disc_fake)  # len(disc_fake) = 3

            loss_g += hp.model.lambda_adv * adv_loss

            loss_g.backward()
            optim_g.step()


            # discriminator

            loss_d_avg = 0.0
            if step > hp.train.discriminator_train_start_steps:
                start = np.random.randint(0, 512 - 40)
                fake_mel = model_g(inputs.unsqueeze(1))
                fake_mel = fake_mel.detach()
                loss_d_sum = 0.0
                for _ in range(hp.train.rep_discriminator):
                    optim_d.zero_grad()
                    disc_fake = model_d(fake_mel, start)
                    disc_real = model_d(labels.unsqueeze(1), start)
                    loss_d = 0.0
                    loss_d_real = 0.0
                    loss_d_fake = 0.0
                    for score_fake, score_real in zip(disc_fake, disc_real):
                        loss_d_real += criterion_mse(score_real, torch.ones_like(score_real))
                        loss_d_fake += criterion_mse(score_fake, torch.zeros_like(score_fake))
                    loss_d_real = loss_d_real / len(disc_real)  # len(disc_real) = 3
                    loss_d_fake = loss_d_fake / len(disc_fake)  # len(disc_fake) = 3
                    loss_d = loss_d_real + loss_d_fake
                    loss_d.backward()
                    optim_d.step()
                    loss_d_sum += loss_d
                loss_d_avg = loss_d_sum / hp.train.rep_discriminator
                loss_d_avg = loss_d_avg.item()

            step += 1
            # logging
            loss_g = loss_g.item()
            avg_g_loss.append(loss_g)
            avg_d_loss.append(loss_d_avg)
            avg_adv_loss.append(adv_loss)



            # tensorboard logging
            if step % hp.logging_step == 0:
                writer.log_scaler("g_loss",   sum(avg_g_loss) / len(avg_g_loss), step)
                writer.log_scaler("adv_loss", sum(avg_adv_loss) / len(avg_adv_loss), step)
                writer.log_scaler("d_loss",   sum(avg_d_loss) / len(avg_d_loss), step)
                loader.set_description(
                    "Avg : g %.04f d %.04f ad %.04f| step %d" % (sum(avg_g_loss) / len(avg_g_loss),
                                                                                sum(avg_d_loss) / len(avg_d_loss),
                                                                                sum(avg_adv_loss) / len(avg_adv_loss),
                                                                                step)
                )

            # Validatiuon
            if step % hp.validation_interval == 0:
                valid_metrics = validation(
                    val_dataloader, model_g, model_d, criterion_l1, criterion_mse, writer, step
                )
                save_path = os.path.join(checkpoint_dir, '%s_%s_%04d.pt'
                                         % (args.name, githash, epoch))
                torch.save({
                    'model_g': model_g.state_dict(),
                    'model_d': model_d.state_dict(),
                    'optim_g': optim_g.state_dict(),
                    'optim_d': optim_d.state_dict(),
                    'step': step,
                    'epoch': epoch,
                    'hp_str': hp_str,
                    'githash': githash,
                }, save_path)
                print("Saved checkpoint to: %s" % save_path)

            step += 1


def validation(val_dataloader, model_g, model_d, criterion_l1, criterion_mse, writer, step):


    # switch to evaluate mode
    model_g.eval()
    model_d.eval()
    loss_g_sum = 0.0
    loss_d_sum = 0.0
    # Iterate over data.

    for idx, data in enumerate(tqdm(val_dataloader, desc="validation")):

        # get the inputs and wrap in Variable
        inputs = data["sat_img"].cuda()
        labels = data["map_img"].cuda()

        # generator
        start = np.random.randint(0, 512 - 40)
        fake_mel = model_g(inputs.unsqueeze(1))  # B, 1, T' torch.Size([1, 1, 212992])
        if idx < 1:
            writer.log_image("actual", labels.squeeze(), "Validation")
            writer.log_image("input", inputs.squeeze(), "Validation")
            writer.log_image("generated", fake_mel.squeeze(), "Validation")
        disc_fake = model_d(fake_mel, start)  # B, 1, T torch.Size([1, 1, 212893])
        disc_real = model_d(labels, start)

        adv_loss = 0.0
        loss_d_real = 0.0
        loss_d_fake = 0.0
        loss_g = criterion_l1(fake_mel.squeeze(1), labels)


        for score_fake, score_real in zip(disc_fake, disc_real):
            adv_loss += criterion_mse(score_fake, torch.ones_like(score_fake))
            loss_d_real += criterion_mse(score_real, torch.ones_like(score_real))
            loss_d_fake += criterion_mse(score_fake, torch.zeros_like(score_fake))
        adv_loss = adv_loss / len(disc_fake)
        loss_d_real = loss_d_real / len(score_real)
        loss_d_fake = loss_d_fake / len(disc_fake)
        loss_g += hp.model.lambda_adv * adv_loss
        loss_d = loss_d_real + loss_d_fake
        loss_g_sum += loss_g.item()
        loss_d_sum += loss_d.item()

    loss_g_avg = loss_g_sum / len(val_dataloader.dataset)
    loss_d_avg = loss_d_sum / len(val_dataloader.dataset)
    writer.log_scaler("g_loss", loss_g_avg, step, "Validation")
    writer.log_scaler("d_loss", loss_d_avg, step, "Validation")
    print("G Loss: {:.4f} D Loss: {:.4f}".format(loss_g_avg, loss_d_avg))
    model_g.train()
    model_d.train()
    return {"g_loss": loss_g_avg, "d_acc": loss_d_avg}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Road and Building Extraction")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="yaml file for configuration"
    )
    parser.add_argument(
        "--epochs",
        default=75,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument("--name", default="default", type=str, help="Experiment name")

    args = parser.parse_args()

    hp = HParam(args.config)
    with open(args.config, "r") as f:
        hp_str = "".join(f.readlines())

    main(hp, num_epochs=args.epochs, resume=args.resume, name=args.name)
