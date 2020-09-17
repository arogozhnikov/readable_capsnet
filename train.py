import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from einops import rearrange, reduce, asnumpy
import numpy as np
from capsnet import Encoder, Decoder


def load_mnist(batch_size, workers=4):
    train_transform = transforms.Compose([
        # small random shifts
        transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), fillcolor=0),
        transforms.ToTensor(),
    ])
    test_transform = transforms.ToTensor()

    training_data_loader = DataLoader(
        datasets.MNIST('./data', train=True, download=True, transform=train_transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    testing_data_loader = DataLoader(
        datasets.MNIST('./data', train=False, download=True, transform=test_transform),
        batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)

    image_shape = (28, 28, 1)
    n_classes = 10
    return training_data_loader, testing_data_loader, image_shape, n_classes


device = torch.device('cuda')

train_loader, test_loader, (image_h, image_w, image_c), n_classes = load_mnist(batch_size=64, workers=4)

encoder = Encoder(
    in_h=image_h, in_w=image_w, in_c=image_c,
    n_primary_caps_groups=32, primary_caps_dim=8,
    n_digit_caps=n_classes, digit_caps_dim=16
).to(device)

decoder = Decoder(
    n_caps=n_classes, caps_dim=16,
    output_h=image_h, output_w=image_w, output_channels=image_c,
).to(device)

optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()])


def margin_loss(class_capsules, target_one_hot, m_minus=0.1, m_plus=0.9, loss_lambda=0.5):
    caps_norms = torch.norm(class_capsules, dim=2)
    assert caps_norms.max() <= 1.001, 'capsules outputs should be bound by unit norm'
    # correct capsule is enforced is not penalized if norm > m_plus,
    # while incorrect ones are not penalized if norm < m_minus
    loss_sig = torch.clamp(m_plus - caps_norms, 0) ** 2
    loss_bkg = torch.clamp(caps_norms - m_minus, 0) ** 2

    loss = target_one_hot * loss_sig + loss_lambda * (1.0 - target_one_hot) * loss_bkg
    return reduce(loss, 'b cls -> b', 'sum')


for epoch in range(100):
    for i, (images, labels) in enumerate(train_loader):
        digit_capsules = encoder(images.to(device))
        labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=10).float().to(device)
        loss = margin_loss(digit_capsules, labels_one_hot).mean()
        reconstructed = decoder(digit_capsules * rearrange(labels_one_hot, 'b caps -> b caps 1'))
        reconstruction_loss_mse = (images - reconstructed).pow(2).mean()
        loss += reconstruction_loss_mse * 10.  # pick a weight
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    accuracies = []
    for images, labels in test_loader:
        digit_capsules = encoder(images.to(device)).cpu()
        # predicted capsule is capsule with largest norm
        predicted_labels = digit_capsules.norm(dim=2).argmax(dim=1)
        accuracies += asnumpy(predicted_labels == labels).tolist()

    print(f'epoch {epoch} accuracy: {np.mean(accuracies)}')
