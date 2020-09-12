import torch
import torch.nn as nn

from einops import reduce, repeat, rearrange
from einops.layers.torch import WeightedEinsum, Rearrange


def squash(x, dim):
    """
    The non-linear activation used in Capsule.
    It drives the length of a large vector to near 1 and small vector to 0
    This implement equation 1 from the paper.
    """
    norm_sq = torch.sum(x ** 2, dim, keepdim=True)
    norm = torch.sqrt(norm_sq)
    return (norm_sq / (1.0 + norm_sq)) * (x / norm)


class Capsule2Capsule(nn.Module):
    def __init__(self, in_caps, in_hid, out_caps, out_hid, routing_iterations=3):
        super().__init__()
        assert routing_iterations > 0, 'at least one iteration is needed'
        self.routing_iterations = routing_iterations
        self.input_caps2U = WeightedEinsum(
            'b in_caps in_hid -> b in_caps out_caps out_hid',
            weight_shape='in_caps in_hid out_caps out_hid',
            in_hid=in_hid, in_caps=in_caps, out_hid=out_hid, out_caps=out_caps,
        )

    def forward(self, input_capsules):
        U = self.input_caps2U(input_capsules)

        batch, in_caps, out_caps, out_hid = U.shape

        # logsoftmax for connections between capsules
        B = torch.zeros([batch, in_caps, out_caps], device=U.device)

        # routing algorithm
        # names of axes: b=batch, i=input capsules, o=output_capsules, h=hidden dim of output capsule
        for _ in range(self.routing_iterations):
            # "routing softmax" that determines connection between capsules in layers
            C = torch.softmax(B, dim=-1)
            S = torch.einsum('bio,bioh->boh', C, U)
            V = squash(S, dim=-1)
            B = B + torch.einsum('bioh,boh->bio', U, V)
        assert torch.norm(V, dim=2).max() <= 1.001
        return V


class RoutingEncoder(nn.Module):
    def __init__(self, in_h, in_w, in_c,
                 n_primary_caps_groups, primary_caps_dim,
                 n_digit_caps, digit_caps_dim,
                 ):
        super().__init__()
        self.image2primary_capsules = nn.Sequential(
            nn.Conv2d(in_c, 256, kernel_size=9),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, n_primary_caps_groups * primary_caps_dim, kernel_size=9, stride=2),
            Rearrange('b (caps hid) h w -> b (h w caps) hid', caps=n_primary_caps_groups, hid=primary_caps_dim),
        )
        # figure out correct number of capsules by passing a test image through
        _, self.n_primary_capsules, _ = self.image2primary_capsules(torch.zeros(1, in_c, in_h, in_w)).shape
        self.primary2digit_capsules = Capsule2Capsule(
            in_caps=self.n_primary_capsules, in_hid=primary_caps_dim,
            out_caps=n_digit_caps, out_hid=digit_caps_dim,
        )

    def forward(self, images):
        primary_capsules = self.image2primary_capsules(images) * 0.01
        return self.primary2digit_capsules(primary_capsules)


def Decoder(n_caps, caps_dim, output_h, output_w, output_channels):
    return nn.Sequential(
        WeightedEinsum('b caps caps_dim -> b hidden', weight_shape='caps caps_dim hidden',
                       caps=n_caps, caps_dim=caps_dim, hidden=512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        WeightedEinsum('b hidden -> b c h w', weight_shape='hidden c h w',
                       hidden=1024, h=output_h, w=output_w, c=output_channels),
        nn.Sigmoid(),
    )












def test(n_digit_caps=10, digit_caps_dim=11):
    size = 28
    image_channels = 3
    encoder = RoutingEncoder(
        in_c=image_channels, in_h=size, in_w=size,
        n_primary_caps_groups=image_channels,
        primary_caps_dim=12,
        n_digit_caps=n_digit_caps,
        digit_caps_dim=digit_caps_dim,
    )

    batch_size = 2
    images = torch.zeros(batch_size, image_channels, size, size)

    embeddings = encoder(images)

    assert embeddings.shape == (batch_size, n_digit_caps, digit_caps_dim)

    decoder = Decoder(n_digit_caps, caps_dim=digit_caps_dim, output_h=size, output_w=size,
                      output_channels=image_channels)
    recostructed = decoder(embeddings)

    assert recostructed.shape == (batch_size, image_channels, size, size)
    print('ok')


test()
