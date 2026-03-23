import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import os

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory
os.makedirs("results/gan_images", exist_ok=True)

# =====================================================
# 📌 PART 1: CNN (CIFAR-10 Classification)
# =====================================================

print("\n--- CNN MODEL TRAINING ---")

transform = T.Compose([T.ToTensor()])

train_data = torchvision.datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)

train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=64, shuffle=True
)

# CNN Model
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


cnn = CNNNet().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(cnn.parameters(), lr=0.001)

# Training CNN
for ep in range(2):
    epoch_loss = 0

    for imgs, lbls in train_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)

        optimizer.zero_grad()
        preds = cnn(imgs)
        loss = loss_fn(preds, lbls)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {ep+1} Loss: {epoch_loss:.4f}")


# =====================================================
# 📌 PART 2: RNN / LSTM / GRU
# =====================================================

print("\n--- SEQUENCE MODELS ---")

vocab = 4000
seq_length = 40

data_x = torch.randint(0, vocab, (400, seq_length)).to(device)
data_y = torch.randint(0, 2, (400,)).float().to(device)

class SequenceModel(nn.Module):
    def __init__(self, mode="RNN"):
        super().__init__()

        self.embed = nn.Embedding(vocab, 50)

        if mode == "LSTM":
            self.layer = nn.LSTM(50, 32, batch_first=True)
        elif mode == "GRU":
            self.layer = nn.GRU(50, 32, batch_first=True)
        else:
            self.layer = nn.RNN(50, 32, batch_first=True)

        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = self.embed(x)
        output, _ = self.layer(x)
        return self.out(output[:, -1]).squeeze()


def run_sequence_model(model_type):
    model = SequenceModel(model_type).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(2):
        optimizer.zero_grad()
        out = model(data_x)
        loss = loss_fn(out, data_y)
        loss.backward()
        optimizer.step()

    print(f"{model_type} Loss: {loss.item():.4f}")


run_sequence_model("RNN")
run_sequence_model("LSTM")
run_sequence_model("GRU")


# =====================================================
# 📌 PART 3: GAN (Fashion-MNIST)
# =====================================================

print("\n--- GAN TRAINING ---")

transform = T.Compose([T.ToTensor()])
fashion_data = torchvision.datasets.FashionMNIST(
    root="./data", train=True, download=True, transform=transform
)

gan_loader = torch.utils.data.DataLoader(
    fashion_data, batch_size=64, shuffle=True
)

# Generator
class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)


# Discriminator
class Disc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


G = Gen().to(device)
D = Disc().to(device)

criterion = nn.BCELoss()
opt_g = optim.Adam(G.parameters(), lr=0.0002)
opt_d = optim.Adam(D.parameters(), lr=0.0002)

from torchvision.utils import save_image

for ep in range(3):
    for real_imgs, _ in gan_loader:
        real_imgs = real_imgs.view(-1, 784).to(device)
        bs = real_imgs.size(0)

        # Labels
        real = torch.ones(bs, 1).to(device)
        fake = torch.zeros(bs, 1).to(device)

        # Train Discriminator
        noise = torch.randn(bs, 100).to(device)
        fake_imgs = G(noise)

        d_loss = criterion(D(real_imgs), real) + \
                 criterion(D(fake_imgs.detach()), fake)

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # Train Generator
        g_loss = criterion(D(fake_imgs), real)

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    print(f"Epoch {ep+1} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

    save_image(fake_imgs.view(-1, 1, 28, 28),
               f"results/gan_images/epoch_{ep}.png")

print("\n✔ Execution Completed")