import sys
sys.path.insert(0,'./nv-wavenet/pytorch')

import os
import json
import argparse
import torch
import importlib
from scipy.io.wavfile import write

#=====START: ADDED FOR DISTRIBUTED======4
#from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
#from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======

from torch.utils.data import DataLoader
from wavenet import WaveNet
from mel2samp_onehot import Mel2SampOnehot
from utils import to_gpu
import nv_wavenet
utils_nv= importlib.import_module("nv-wavenet.pytorch.utils")

class CrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        self.num_classes = wavenet_config["n_out_channels"]

    def forward(self, inputs, targets):
        """
        inputs are batch by num_classes by sample
        targets are batch by sample
        torch CrossEntropyLoss needs
            input = batch * samples by num_classes
            targets = batch * samples
        """
        targets = targets.view(-1)
        inputs = inputs.transpose(1, 2)
        inputs = inputs.contiguous()
        inputs = inputs.view(-1, self.num_classes)
        return torch.nn.CrossEntropyLoss()(inputs, targets)

def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    for key,value in checkpoint_dict.items():
        print(key)
    iteration = checkpoint_dict['iteration']
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    model_for_loading = checkpoint_dict['model']
    model.load_state_dict(model_for_loading.state_dict())
    print("Loaded checkpoint '{}' (iteration {})" .format(
          checkpoint_path, iteration))
    return model, optimizer, iteration

def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
          iteration, filepath))
    model_for_saving = WaveNet(**wavenet_config).cuda()
    model_for_saving.load_state_dict(model.state_dict())
    torch.save({'model': model_for_saving,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)

def train_wav(num_gpus, rank, group_name, output_directory, epochs, learning_rate,
          iters_per_checkpoint, batch_size, seed, checkpoint_path):

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_name, **dist_config)
    # =====END:   ADDED FOR DISTRIBUTED======

    criterion = CrossEntropyLoss()
    model = WaveNet(**wavenet_config).cuda()

    # =====START: ADDED FOR DISTRIBUTED======
    if num_gpus > 1:
        model = apply_gradient_allreduce(model)
    # =====END:   ADDED FOR DISTRIBUTED======

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Load checkpoint if one exists
    iteration = 0
    if checkpoint_path != "":
        model, optimizer, iteration = load_checkpoint(checkpoint_path, model,
                                                      optimizer)
        iteration += 1  # next iteration is iteration + 1

    trainset = Mel2SampOnehot(**data_config)
    # =====START: ADDED FOR DISTRIBUTED======
    train_sampler = DistributedSampler(trainset) if num_gpus > 1 else None
    # =====END:   ADDED FOR DISTRIBUTED======
    train_loader = DataLoader(trainset, num_workers=1, shuffle=False,
                              sampler=train_sampler,
                              batch_size=batch_size,
                              pin_memory=False,
                              drop_last=True)

    # Get shared output_directory ready
    if rank == 0:
        if not os.path.isdir(output_directory):
            os.makedirs(output_directory)
            os.chmod(output_directory, 0o775)
        print("output directory", output_directory)

    model.train()
    epoch_offset = max(0, int(iteration / len(train_loader)))
    # ================ MAIN TRAINNIG LOOP! ===================
    for epoch in range(epoch_offset, epochs):
        print("Epoch: {}".format(epoch))
        for i, batch in enumerate(train_loader):
            model.zero_grad()

            x, y = batch
            x = to_gpu(x).float()
            y = to_gpu(y)
            x = (x, y)  # auto-regressive takes outputs as inputs
            y_pred = model(x)
            loss = criterion(y_pred, y)
            if num_gpus > 1:
                reduced_loss = reduce_tensor(loss.data, num_gpus)[0]
            else:
                reduced_loss = loss.item()
            loss.backward()
            optimizer.step()

            print("{}:\t{:.9f}".format(iteration, reduced_loss))

            if (iteration % iters_per_checkpoint == 0):
                if rank == 0:
                    checkpoint_path = "{}/wavenet_{}".format(
                        output_directory, iteration)
                    save_checkpoint(model, optimizer, learning_rate, iteration,
                                    checkpoint_path)

            iteration += 1

def load_wav_model(checkpoint_path):

    model = torch.load(checkpoint_path)['model']
    wavenet = nv_wavenet.NVWaveNet(**(model.export_weights()))
    return model,wavenet

def chunker(seq, size):
    """
    https://stackoverflow.com/a/434328
    """
    return (seq[pos:pos + size] for pos in range(0, len(seq), size))


def infer_wav(mel_files, model,wavenet, output_dir, batch_size, implementation):

    if implementation == "auto":
        implementation = nv_wavenet.Impl.AUTO
    elif implementation == "single":
        implementation = nv_wavenet.Impl.SINGLE_BLOCK
    elif implementation == "dual":
        implementation = nv_wavenet.Impl.DUAL_BLOCK
    elif implementation == "persistent":
        implementation = nv_wavenet.Impl.PERSISTENT
    else:
        raise ValueError("implementation must be one of auto, single, dual, or persistent")
    mel_files = utils_nv.files_to_list(mel_files)

    for files in chunker(mel_files, batch_size):
        mels = []
        for file_path in files:
            print(file_path)
            mel = torch.load(file_path)
            mel = utils_nv.to_gpu(mel)
            mels.append(torch.unsqueeze(mel, 0))
        cond_input = model.get_cond_input(torch.cat(mels, 0))
        audio_data = wavenet.infer(cond_input, implementation)

        for i, file_path in enumerate(files):
            file_name = os.path.splitext(os.path.basename(file_path))[0]

            audio = utils_nv.mu_law_decode_numpy(audio_data[i, :].cpu().numpy(), wavenet.A)
            audio = utils_nv.MAX_WAV_VALUE * audio
            wavdata = audio.astype('int16')
            write("{}/{}.wav".format(output_dir, file_name),
                  16000, wavdata)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-c','--config', type=str,
                        help='JSON file for nv-wavenet configuration', default='./nv-wavenet/pytorch/config.json')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')
    args = parser.parse_args()


    with open(args.config) as f:
        data = f.read()
    config = json.loads(data)
    train_config = config["train_config"]
    global data_config
    data_config = config["data_config"]
    global dist_config
    dist_config = config["dist_config"]
    global wavenet_config
    wavenet_config = config["wavenet_config"]

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        if args.group_name == '':
            print("WARNING: Multiple GPUs detected but no distributed group set")
            print("Only running 1 GPU.  Use distributed.py for multiple GPUs")
            num_gpus = 1

    if num_gpus == 1 and args.rank != 0:
        raise Exception("Doing single GPU training on rank > 0")

    train_wav(num_gpus, args.rank, args.group_name, **train_config)



