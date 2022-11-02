
import configparser
import math
import random
import time
import numpy as np

import torch

#import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion


def has_cuda():
    # is there even cuda available?
    has_cuda = torch.cuda.is_available()
    if (has_cuda):
        # we require cuda version >=3.5
        capabilities = torch.cuda.get_device_capability()
        major_version = capabilities[0]
        minor_version = capabilities[1]
        if major_version < 3 or (major_version == 3 and minor_version < 5):
            has_cuda = False
    return has_cuda


# Output a torch device to use.
def get_torch_device():
    # output the gpu or cpu device
    device = torch.device('cuda' if has_cuda() else 'cpu')
    return device


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent+.00001)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def parseConfigFile(filename='config.ini'):
    args = {}
    config = configparser.ConfigParser()
    config.read('config.ini')

    args['data_folder'] = config['INPUT_DATA']['data_folder']
    args['file_ext'] = config['INPUT_DATA']['file_ext']

    args['data_name'] = config['TRAIN_DATA']['data_name']
    args['patch_size'] = int(config['TRAIN_DATA']['patch_size'])
    args['n_channels'] = int(config['TRAIN_DATA']['n_channels'])
    args['max_samples_per_image'] = int(config['TRAIN_DATA']['max_samples_per_image'])
    if args['max_samples_per_image'] == -1:
        args['max_samples_per_image'] = np.inf

    args['sample_level'] = int(config['TRAIN_DATA']['sample_level'])
    args['train_perc'] = float(config['TRAIN_DATA']['train_perc'])
    args['pytables_folder'] = config['TRAIN_DATA']['pytables_folder']

    args['model_path'] = config['AUTOENCODER']['model_path']
    args['n_classes'] = int(config['AUTOENCODER']['n_classes'])
    args['padding'] = bool(config['AUTOENCODER']['padding'])
    args['depth'] = int(config['AUTOENCODER']['depth'])
    args['wf'] = int(config['AUTOENCODER']['wf'])
    args['up_mode'] = config['AUTOENCODER']['up_mode']
    args['batch_norm'] = bool(config['AUTOENCODER']['batch_norm'])
    args['batch_size'] = int(config['AUTOENCODER']['batch_size'])
    args['num_epochs'] = int(config['AUTOENCODER']['num_epochs'])

    return args


def HeatMap_Sequences(data, filename='Heatmap_Sequences.tif'):
        """
        HeatMap of Sequences
        This method creates a heatmap/clustermap for sequences by latent features
        for the unsupervised deep lerning methods.
        Inputs
        ---------------------------------------
        filename: str
            Name of file to save heatmap.

        sample_num_per_class: int
            Number of events to randomly sample per class for heatmap.
        color_dict: dict
            Optional dictionary to provide specified colors for classes.
        Returns
        ---------------------------------------
        """

        sns.set(font_scale=0.5)
        CM = sns.clustermap(data, standard_scale=1, cmap='bwr')
        ax = CM.ax_heatmap
        ax.set_xticklabels('')
        ax.set_yticklabels('')
        plt.show()
        plt.savefig(filename)


def bw_on_image(im, bw, color):
    for i in range(0, im.shape[2]):
        t = im[:, :, i]
        t[bw] = color[i]
        im[:, :, i] = t


def label_on_image(im, labelIm, M=None, inPlace=True, randSeed=None):
    if not inPlace:
        im2 = im.copy()
        label_on_image(im2, labelIm, M, True)
        return im2
    else:
        max_label = np.max(labelIm)
        if M is None:
            if randSeed is not None:
                random.seed(randSeed)
            M = np.random.randint(0, 255, (max_label+1, 3))

        if max_label == 0:
            return
        elif max_label == 1:
            bw_on_image(im, labelIm == max_label, M[1, :])
        else:
            for r in range(1, im.shape[0]-1):
                for c in range(1, im.shape[1]-1):
                    l = labelIm[r, c]
                    #if l > 0 and (l != labelIm[r,c-1] or l != labelIm[r,c+1] or l != labelIm[r-1,c] or l != labelIm[r+1,c]):
                    if l > 0:
                        im[r,c,0] = M[l, 0]
                        im[r,c,1] = M[l, 1]
                        im[r,c,2] = M[l, 2]


def find_edge(bw, strel=5):
    return np.bitwise_xor(bw, binary_erosion(bw, structure=np.ones((strel, strel))).astype(bw.dtype))
