from unittest import TestCase
import cProfile
import time
from audio_data import *
import numpy as np
import torch.utils.data
from pathlib import Path


class TestWavenetDataset(TestCase):
    def test_dataset_creation(self):
        in_path = Path(__file__).parents[1] / 'train_samples'
        out_path = Path(__file__).parents[1] / 'train_samples/test_dataset.npz'
        dataset = WavenetDataset(dataset_file=str(out_path),
                                 item_length=1000,
                                 target_length=64,
                                 file_location=str(in_path))
        print('Length dataset', len(dataset))

    def test_minibatch_performance(self):
        dataset_file = Path(__file__).parents[1] / 'train_samples/test_dataset.npz'
        assert dataset_file.exists()
        dataset = WavenetDataset(dataset_file=str(dataset_file),
                                 item_length=1000,
                                 target_length=64)
        print('Length dataset', len(dataset))
        dataloader = torch.utils.data.DataLoader(dataset=str(dataset),
                                                 batch_size=8,
                                                 shuffle=True,
                                                 num_workers=8)
        dataloader_iter = iter(dataloader)
        num_batches = 5

        def calc_batches(num=1):
            for i in range(num):
                mb = next(dataloader_iter)
            return mb

        tic = time.time()
        last_minibatch = calc_batches(num_batches)
        toc = time.time()

        print("time it takes to calculate "  + str(num_batches) + " minibatches: " + str(toc-tic) + " s")




class TestListAllAudioFiles(TestCase):
    def test_list_all_audio_files(self):
        path = Path(__file__).parents[1] / 'train_samples'
        files = list_all_audio_files(str(path))
        print(files)
        assert len(files) > 0


class TestQuantizeData(TestCase):
    def test_quantize_data(self):
        data = np.random.rand(32) * 2 - 1
        qd = quantize_data(data, 256)
