import argparse
import os

import torch.nn as nn
import torch.nn.functional as F
import torch
import h5py
import numpy as np
#import pkg_resources
#pkg_resources.require("sklearn==1.0.2")
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

import cerebras.framework.torch as cbtorch
import cerebras.framework.torch.core.cb_model as cm
import cerebras.models.common.pytorch.optim.AdamW as AdamW
from cerebras.framework.torch import amp


# ================================ Define model ================================
class DMSNet(nn.Module):
    """ Define a CNN """

    def __init__(self, device='cpu'):
        """ Initialize network with some hyperparameters """
        super(DMSNet, self).__init__()

        self.device = device
        self.conv1 = nn.Conv2d(3, 8, kernel_size=4)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=4)
        self.pool = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(87584, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.drop = nn.Dropout(p=0.5)

        self.output_dim = 1

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1(x)))).to(self.device)
        x = self.bn2(self.pool(F.relu(self.conv2(x)))).to(self.device)
        x = x.view(x.size(0), -1).to(self.device)
        x = self.drop(F.relu(self.fc1(x))).to(self.device)
        x = self.drop(F.relu(self.fc2(x))).to(self.device)
        x = F.softmax(self.fc3(x), dim=1).to(self.device)
        return x


# ================================ Get the data loader ================================

def get_train_dataloader():
    batch_size = 32
    device = 'cpu'

    if not cm.is_master_ordinal(local=False):
        cm.rendezvous("download_dataset_only_once")

    dataset_path = '/home/ai012/ai012/jounaidr/sciml_bench/datasets/dms_sim/training/data-binary.h5'
    hf = h5py.File(dataset_path, 'r')

    img = hf['train/images'][:]
    img = np.swapaxes(img, 1, 3)
    X_train = torch.from_numpy(np.atleast_3d(img)).to(device)
    lab = np.array(hf['train/labels']).reshape(-1, 1)
    onehot_encoder = OneHotEncoder(sparse=False)
    lab = onehot_encoder.fit_transform(lab).astype(int)
    Y_train = torch.from_numpy(lab).float().to(device)

    img = hf['test/images'][:]
    img = np.swapaxes(img, 1, 3)
    X_test = torch.from_numpy(np.atleast_3d(img)).to(device)
    lab = np.array(hf['test/labels']).reshape(-1, 1)
    lab = onehot_encoder.fit_transform(lab).astype(int)
    Y_test = torch.from_numpy(lab).float().to(device)

    datasets = (X_train, Y_train, X_test, Y_test)
    if cm.is_master_ordinal(local=False):
        cm.rendezvous("download_dataset_only_once")

    train_loader = torch.utils.data.DataLoader(
        datasets,
        batch_size=batch_size,
        sampler=None,
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )
    return train_loader


# ================================ train ================================

def train(X_train, Y_train, batch_size, model, criterion, optimizer):
    model.train()
    train_err = 0
    train_acc = []
    for k in range(batch_size, X_train.shape[0], batch_size):
        preds = model(X_train[k - batch_size:k])
        loss = criterion(preds, Y_train[k - batch_size:k])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_err += loss.item()
        thresholded_results = np.where(
            preds.detach().cpu().numpy() > 0.5, 1, 0)
        train_acc.append(accuracy_score(thresholded_results,
                                        Y_train[k - batch_size:k].detach().cpu().numpy()))
    train_err /= (X_train.shape[0] / (batch_size))
    return train_err, np.mean(np.array(train_acc))


# ================================ run the model! ================================

def run(model, train_dataloader, optimizer, scaler, criterion, save_dir="./"):
    model.train()
    num_epochs = 30
    checkpoint_steps = 10
    steps_per_epoch = 10
    global_step = 0
    max_steps = 10000

    with cbtorch.Session(train_dataloader, mode="train", ) as session:
        for epoch in range(num_epochs):
            cm.master_print(f"Epoch {epoch} train begin")
            for step, batch in enumerate(train_dataloader):
                global_step += 1
                optimizer.zero_grad()
                input, label = batch
                output = model(input)
                loss = criterion(output, label)
                scaler(loss).backward()
                optimizer.step()
                # helper
                if global_step % checkpoint_steps == 0:
                    def closure():
                        file_name = os.path.join(
                            save_dir, f"checkpoint_{epoch}_{step}.mdl"
                        )
                        state = {
                            "model": model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                            "amp": amp.state_dict(),
                        }
                        cm.save(state, file_name, master_only=True)

                    # if we want to access model state in order to save, we must add a step closure
                    session.add_step_closure(closure)
                if global_step >= max_steps or (step + 1) >= steps_per_epoch:
                    session.mark_step()
                    break
                if global_step >= max_steps:
                    break

    cm.master_print("Training completed successfully!")


# ================================ helper methods ================================

def accuracy_score(y_true, y_pred):
    y_pred = np.concatenate(tuple(y_pred))
    y_true = np.concatenate(tuple([[t for t in y] for y in y_true])).reshape(y_pred.shape)
    return (y_true == y_pred).sum() / float(len(y_true))


def main(args):
    learning_rate = 0.0001

    # Initialize Cerebras backend and configure for run
    cbtorch.initialize(cs_ip=args.cs_ip)

    # prepare to move the model and dataloader onto the Cerebras engine
    model = cbtorch.module(DMSNet())
    train_dataloader = cbtorch.dataloader(get_train_dataloader())

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    # Initialize mixed precision to be compatible with Cerebras
    scaler = amp.GradScaler(loss_scale=1.0)

    run(model, train_dataloader, optimizer, scaler, criterion)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cs_ip",
        type=str,
        help="IP address of Cerebras system",
    )
    main(parser.parse_args())
