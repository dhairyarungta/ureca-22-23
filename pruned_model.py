import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune

# Define the CNN model
class VoiceClassificationModel(nn.Module):
    def __init__(self):
        super(VoiceClassificationModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(5*5*64, 128)
        self.fc2 = nn.Linear(128, 2)  # Binary classification (2 classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*64)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Instantiate the model
model = VoiceClassificationModel()

# Prune the model using magnitude-based weight pruning
parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight')
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2  # Prune 20% of the weights
)

# Remove the pruned connections
prune.remove(model.conv1, 'weight')
prune.remove(model.conv2, 'weight')
prune.remove(model.fc1, 'weight')
prune.remove(model.fc2, 'weight')

# Save the pruned model
torch.save(model.state_dict(), 'pruned_model.pth')
