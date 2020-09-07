import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader


def create_dataloader(hp, train):
    dataset = MelFromDisk(hp, train)

    if train:
        return DataLoader(dataset=dataset, batch_size=hp.train.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True, drop_last=True)
    else:
        return DataLoader(dataset=dataset, batch_size=1, shuffle=False,
                          num_workers=0, pin_memory=False, drop_last=False)


class MelFromDisk(Dataset):
    def __init__(self, hp, train):
        self.hp = hp
        self.train = train
        self.path = hp.data.input_dir if train else hp.data.valid_input_dir
        self.wav_list = glob.glob(os.path.join(self.path, '**', '*.npy'), recursive=True)
        self.mel_segment_length = hp.model.idim
        self.mapping = [i for i in range(len(self.wav_list))]

    def __len__(self):
        return len(self.wav_list)

    def __getitem__(self, idx):
        input_mel = self.wav_list[idx]
        id = os.path.basename(input_mel).split(".")[0]

        input_mel_path = "{}/{}.npy".format(self.hp.data.input_dir, id)
        output_mel_path = "{}/{}.npy".format(self.hp.data.output_dir, id)

        mel_gt = torch.from_numpy(np.load(output_mel_path))
        # mel = torch.load(melpath).squeeze(0) # # [num_mel, T]

        mel_gta = torch.from_numpy(np.load(input_mel_path))

        max_mel_start = mel_gta.size(1) - self.mel_segment_length
        mel_start = random.randint(0, max_mel_start)
        mel_end = mel_start + self.mel_segment_length
        mel_gta = mel_gta[:, mel_start:mel_end]
        mel_gt = mel_gt[:, mel_start:mel_end]

        return mel_gta, mel_gt

    def shuffle_mapping(self):
        random.shuffle(self.mapping)