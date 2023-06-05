from unittest import TestCase

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm

from stylegan_generators.stylegan2 import Generator


class StyleGAN2Test(TestCase):

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

    def test_overfit(self):
        Z = 64
        C = 3
        R = 128
        D = 2
        B = 16

        dataset = torch.arange(0, R*R*C*D).reshape((D, C, R, R)).cuda() / (R*R*C*D)
        #dataset = torch.randn((D, C, R, R)).cuda()
        network = Generator(Z, 0, 256, R, C).cuda()
        embeddings = nn.Embedding(D, Z).cuda()

        criterion = MSELoss()

        optimizer = Adam(list(network.parameters()) + list(embeddings.parameters()), lr=1e-4)

        progress = tqdm(range(100))
        for iteration in progress:
            idx = torch.randint(0, D, (B,)).cuda()
            batch = dataset[idx]
            codes = embeddings(idx)

            predicted_img = network(codes, None)
            loss = criterion(predicted_img, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress.set_postfix({"loss": loss.item()})

        self.assertLess(loss.item(), 0.05)