import torch
import torch.nn as nn
from einops.layers.torch import EinMix as Mix, Rearrange


def squash(x, dim):
    """ Non-linear activation, that squashes all vectors to have norm < 1 """
    norm_sq = torch.sum(x ** 2, dim, keepdim=True)
    norm = torch.sqrt(norm_sq)
    return (norm_sq / (1.0 + norm_sq)) * (x / norm)


class CapsuleLayerWithRouting(nn.Module):
    def __init__(self, in_caps, in_hid, out_caps, out_hid):
        super().__init__()
        self.input_caps2U = Mix(
            'b in_caps in_hid -> b in_caps out_caps out_hid',
            weight_shape='in_caps in_hid out_caps out_hid',
            in_hid=in_hid, in_caps=in_caps, out_hid=out_hid, out_caps=out_caps,
        )

    def forward(self, input_capsules, routing_iterations):
        U = self.input_caps2U(input_capsules)
        batch, in_caps, out_caps, out_hid = U.shape

        # logsoftmax for connections between capsules
        B = torch.zeros([batch, in_caps, out_caps], device=U.device)

        # routing algorithm (procedure 1 from paper)
        # names of axes: b=batch, i=input capsules, o=output_capsules, h=hidden dim of output capsule
        for _ in range(routing_iterations):
            # "routing softmax" determines connection between capsules in layers
            C = torch.softmax(B, dim=-1)
            S = torch.einsum('bio,bioh->boh', C, U)
            V = squash(S, dim=-1)
            B = B + torch.einsum('bioh,boh->bio', U, V)
        return V


class Encoder(nn.Module):
    def __init__(self, in_h, in_w, in_c,
                 n_primary_caps_groups, primary_caps_dim,
                 n_digit_caps, digit_caps_dim,
                 ):
        super().__init__()
        self.image2primary_capsules = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_primary_caps_groups * primary_caps_dim, kernel_size=9, stride=2),
            # regroup conv output into flat capsules
            Rearrange('b (caps hid) h w -> b (h w caps) hid', caps=n_primary_caps_groups, hid=primary_caps_dim),
        )
        # figure out correct number of capsules by passing a test image through, lazy but simple
        _, n_primary_capsules, _ = self.image2primary_capsules(torch.zeros(1, in_c, in_h, in_w)).shape
        self.primary2digit_capsules = CapsuleLayerWithRouting(
            in_caps=n_primary_capsules, in_hid=primary_caps_dim,
            out_caps=n_digit_caps, out_hid=digit_caps_dim,
        )

    def forward(self, images, routing_iterations=3):
        primary_capsules = self.image2primary_capsules(images) * 0.01  # scaling 0.01 to get norms not too close to 1
        return self.primary2digit_capsules(primary_capsules, routing_iterations)


def Decoder(n_caps, caps_dim, output_h, output_w, output_channels):
    return nn.Sequential(
        Mix('b caps caps_dim -> b hidden', weight_shape='caps caps_dim hidden', caps=n_caps, caps_dim=caps_dim, hidden=512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        Mix('b hidden -> b c h w', weight_shape='hidden c h w', hidden=1024, h=output_h, w=output_w, c=output_channels),
        nn.Sigmoid(),
    )
