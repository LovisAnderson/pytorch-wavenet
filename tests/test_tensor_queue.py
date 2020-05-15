from unittest import TestCase
from scipy.io import wavfile
import torch
import torch.autograd
from torch.nn import ConstantPad2d
import torch.nn.functional as F
from torch.autograd import Variable, gradcheck

from wavenet_training import DilatedQueue, custom_padding
from pathlib import Path


class Test_dilated_queue(TestCase):
    def test_enqueue(self):
        queue = DilatedQueue(max_length=8, num_channels=3)
        e = torch.zeros((3))
        for i in range(11):
            e = e + 1
            queue.enqueue(e)

        data = queue.data[0, :]
        print('data: ', data)
        assert data[0] == 9
        assert data[2] == 11
        assert data[7] == 8

    def test_dequeue(self):
        queue = DilatedQueue(max_length=8, num_channels=1)
        e = torch.zeros((1))
        for i in range(11):
            e = e + 1
            queue.enqueue(e)

        print('data: ', queue.data)

        for i in range(9):
            d = queue.dequeue(num_deq=3, dilation=2)
            print(d)

        assert d[0][0] == 5
        assert d[0][1] == 7
        assert d[0][2] == 9

    def test_combined(self):
        queue = DilatedQueue(max_length=12, num_channels=1)
        e = torch.zeros((1))
        for i in range(30):
            e = e + 1
            queue.enqueue(e)
            d = queue.dequeue(num_deq=3, dilation=4)
            assert d[0][0] == max(i - 7, 0)



class Test_wav_files(TestCase):
    def test_wav_read(self):
        p = Path(__file__).parents[1] / 'train_samples/violin.wav'
        data = wavfile.read(str(p))[1]
        print(data)
        # [0.1, -0.53125...


class Test_padding(TestCase):
    def test_1d(self):
        x = torch.ones((2, 3, 4),  requires_grad=True)

        res = custom_padding(x, 5, dimension=0, pad_start=False)

    
        assert res.size() == (5, 3, 4)
        assert res[-1, 0, 0] == 0


    def test_2d(self):
        pad = ConstantPad2d((5, 0, 0, 0), 0)
        x = Variable(torch.ones((2, 3, 4, 5)))

        res = pad(x)
        print(res.size())
