import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset_loader import DeepfakeDataset
from model import DeepfakeModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset=DeepfakeDataset("dataset")

loader=DataLoader(dataset,batch_size=8,shuffle=True)

model=DeepfakeModel().to(device)

criterion=nn.BCEWithLogitsLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.0001)

for epoch in range(5):

    total_loss=0

    for images,labels in loader:

        images=images.to(device)
        labels=labels.to(device)
        labels=labels.unsqueeze(1)

        outputs=model(images)

        loss=criterion(outputs,labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss+=loss.item()

    print("Epoch",epoch,"Loss",total_loss)

torch.save(model.state_dict(),"deepfake_model.pth")