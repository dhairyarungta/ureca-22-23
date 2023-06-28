import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader

# Define the VoiceDetectionNet model
class VoiceDetectionNet(nn.Module):
    def __init__(self):
        super(VoiceDetectionNet, self).__init__()

        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(32 * 8, 64)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(64, 2)  # Output size 2 for binary classification (Yes/No)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)  # Flatten the input

        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)

        return x

# Custom dataset class for loading and preprocessing the data
class VoiceDataset(Dataset):
    def __init__(self, audio_dir, label_dir):
        self.audio_dir = audio_dir
        self.label_dir = label_dir
        self.audio_files = os.listdir(audio_dir)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        audio_path = os.path.join(self.audio_dir, audio_file)
        waveform, sample_rate = torchaudio.load(audio_path)

        label_file = audio_file.replace('.wav', '.txt')
        label_path = os.path.join(self.label_dir, label_file)
        with open(label_path, 'r') as file:
            label = int(file.read().strip())

        return waveform, label

#Data set importing
audio_dir = ''
label_dir = ''

# Create instances of the dataset
train_dataset = VoiceDataset(audio_dir, label_dir)
test_dataset = VoiceDataset(audio_dir, label_dir)

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Create an instance of the model
model = VoiceDetectionNet()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0

    for waveforms, labels in train_dataloader:
        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(waveforms)

        # Compute the loss
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the weights
        optimizer.step()

        running_loss += loss.item()

    # Print the average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_dataloader)}")

# Save the trained model
torch.save(model.state_dict(), 'voice_detection_model.pt')
