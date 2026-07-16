"""A single client's local training step."""

import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config


def local_train(model, dataset):
    """Train a copy of `model` on `dataset`; return the updated state_dict.
    
    The model is deep-copied so the caller's weights are untouched: every
    client trains its own local copy from the same starting point. Only the
    weights are returned, since that is all the aggregation step consumes.
    """
    local_model = copy.deepcopy(model).to(config.DEVICE)
    local_model.train()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(local_model.parameters(), lr=config.LEARNING_RATE)

    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    for _ in range(config.LOCAL_EPOCHS):
        for images, labels in loader:
            images = images.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            optimizer.zero_grad()
            logits = local_model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()

    return local_model.state_dict()