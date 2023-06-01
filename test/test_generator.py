from unittest import TestCase

import torch

from stylegan3_generator.stylegan3_generator import Generator


class GeneratorTest(TestCase):

    def test_forward(self):
        B = 5
        Z = 64
        C = 3
        R = 128
        network = Generator(Z, 0, 256, R, C)

        input_codes = torch.randn((B, Z))
        generated_images = network(input_codes, None)  # None refers to the condition labels which are not used here

        assert generated_images.shape[0] == B
        assert generated_images.shape[1] == C
        assert generated_images.shape[2] == R
        assert generated_images.shape[3] == R
