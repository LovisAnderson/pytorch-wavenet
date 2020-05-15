import torch
from torch.autograd import Variable
from unittest import TestCase
from wavenet_modules import dilate


class Test_Dilation(TestCase):
    def test_dilate(self):
        input = torch.linspace(0, 12, steps=13).view(1, 1, 13)

        dilated = dilate(input, 1)
        assert dilated.size() == (1, 1, 13)
        assert dilated.data[0, 0, 4] == 4
        print(dilated)

        dilated = dilate(input, 2)
        assert dilated.size() == (2, 1, 7)
        assert dilated.data[1, 0, 2] == 4
        print(dilated)

        dilated = dilate(dilated, 4, init_dilation=2)
        assert dilated.size() == (4, 1, 4)
        assert dilated.data[3, 0, 1] == 4
        print(dilated)

        dilated = dilate(dilated, 1, init_dilation=4)
        assert dilated.size() == (1, 1, 16)
        assert dilated.data[0, 0, 7] == 4
        print(dilated)

    def test_dilate_multichannel(self):
        input = torch.linspace(0, 35, steps=36).view(2, 3, 6)

        dilated = dilate(input, 1)
        dilated = dilate(input, 2)
        dilated = dilate(input, 4)

