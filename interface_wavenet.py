import sys


print(sys.path)
import os
import json


import argparse
import torch

sys.path.insert(0,'./nv-wavenet/pytorch')
#=====START: ADDED FOR DISTRIBUTED======4
#from distributed import init_distributed, apply_gradient_allreduce, reduce_tensor
#from torch.utils.data.distributed import DistributedSampler
#=====END:   ADDED FOR DISTRIBUTED======
print(os.getcwd())
from torch.utils.data import DataLoader
from wavenet import WaveNet




from mel2samp_onehot import Mel2SampOnehot
from utils import to_gpu




from inference import main as inf_main
import nv_wavenet

#from abc import ABC,abstractmethod





import numpy as np
import pyaudio
import wave

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

def infer_wav(mel_path, checkpoint_path, output_dir, batch_size):

    inf_main(mel_path, checkpoint_path, output_dir, batch_size, implementation=nv_wavenet.Impl.AUTO)

def play_audio(fname):
    wf = wave.open(fname, 'rb')
    p = pyaudio.PyAudio()

    chunk = 1024

    # open stream based on the wave object which has been input.
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    # read data (based on the chunk size)
    data = wf.readframes(chunk)

    # play stream (looping from beginning of file to the end)
    while data != '':
        # writing to the stream is what *actually* plays the sound.
        stream.write(data)
        data = wf.readframes(chunk)

        # cleanup stuff.
    stream.close()
    p.terminate()

    print("Output wave generated")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--train_wav',type=bool, help='Argument to train mel spectogram to audio model',default=False)
    parser.add_argument('--infer', type=bool, help = 'Boolean argument to infer audio from text', default=False)
    parser.add_argument('--config', type=str,
                        help='JSON file for nv-wavenet configuration', default='./nv-wavenet/pytorch/config.json')
    parser.add_argument('-r', '--rank', type=int, default=0,
                        help='rank of process for distributed')
    parser.add_argument('-g', '--group_name', type=str, default='',
                        help='name of group for distributed')

    parser.add_argument('-c', "--checkpoint_path", default='./checkpoints/wavenet_250000')
    parser.add_argument('-o', "--output_dir", default='./output')
    parser.add_argument('-b', "--batch_size", default=1)
    parser.add_argument('-i', "--implementation", type=str, default="persistent",
                        help="""Which implementation of NV-WaveNet to use.
                            Takes values of single, dual, or persistent""")
    parser.add_argument('--play', type=bool, default=True)

    args = parser.parse_args()
    if args.implementation == "auto":
        implementation = nv_wavenet.Impl.AUTO
    elif args.implementation == "single":
        implementation = nv_wavenet.Impl.SINGLE_BLOCK
    elif args.implementation == "dual":
        implementation = nv_wavenet.Impl.DUAL_BLOCK
    elif args.implementation == "persistent":
        implementation = nv_wavenet.Impl.PERSISTENT
    else:
        raise ValueError("implementation must be one of auto, single, dual, or persistent")


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

    if args.train_wav:
        train_wav(num_gpus, args.rank, args.group_name, **train_config)

    if args.infer:
        mel_path = 'mel_files.txt'
        infer_wav(mel_path,args.checkpoint_path, args.output_dir, args.batch_size)

    if args.play:
        play_audio('outdir/text_to_mel.wav')


