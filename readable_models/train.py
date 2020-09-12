import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from einops import reduce, asnumpy
import numpy as np

# Preparing dataset


def load_mnist(batch_size, workers=4):
    # Normalize MNIST dataset.
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), fillcolor=0),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    kwargs = dict(num_workers=workers, pin_memory=True)

    training_data_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    testing_data_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=test_transform),
        batch_size=batch_size, shuffle=True, **kwargs)

    return training_data_loader, testing_data_loader


from capsnet import RoutingEncoder, Decoder

device = torch.device('cuda')
encoder = RoutingEncoder(
    in_h=28, in_w=28, in_c=1,
    n_primary_caps_groups=32, primary_caps_dim=8,
    n_digit_caps=10, digit_caps_dim=16
).to(device)

optim = torch.optim.Adam(encoder.parameters())
train_loader, test_loader = load_mnist(batch_size=64, workers=4)


def margin_loss(class_capsules, target_one_hot, m_minus=0.1, m_plus=0.9, loss_lambda=0.5):
    caps_norms = torch.norm(class_capsules, dim=2)
    assert caps_norms.max() <= 1.001
    # Calculate left and right max() terms.
    loss_sig = torch.clamp(m_plus - caps_norms, 0) ** 2
    loss_bkg = torch.clamp(caps_norms - m_minus, 0) ** 2

    t_c = target_one_hot
    # Lc is margin loss for each digit of class c
    loss = t_c * loss_sig + loss_lambda * (1.0 - t_c) * loss_bkg
    return reduce(loss, 'b c -> b', 'sum')


for epoch in range(6):
    for i, (images, labels) in enumerate(train_loader):
        digit_capsules = encoder(images.to(device))
        labels_one_hot = torch.nn.functional.one_hot(labels).to(device)
        loss = margin_loss(digit_capsules, labels_one_hot).mean()
        loss.backward()
        optim.step()
        optim.zero_grad()
        print(epoch, i, loss.item())

    accuracies = []
    for images, labels in test_loader:
        digit_capsules = encoder(images.to(device)).cpu()
        predicted_labels = digit_capsules.norm(dim=2).argmax(dim=1)
        accuracies += asnumpy(predicted_labels == labels).tolist()

    print(f'epoch {epoch} accuracy: {np.mean(accuracies)}')
