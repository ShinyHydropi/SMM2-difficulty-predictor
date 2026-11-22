import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import torch.nn as nn


def train_network(network, loader, validation, device, epochs=60, lr=1e-3):
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    loss_function = nn.BCELoss()
    train_loss_history = []
    val_loss_history = []
    r2_scores = []

    for epoch in range(epochs):
        batch_loss = []
        for batch_idx, batch in enumerate(loader):
            optimizer.zero_grad()
            loss = loss_function(network(batch, device=device), batch["label"].to(device=device))
            loss.backward()
            batch_loss.append(loss.item())
            optimizer.step()
        train_loss_history.append(np.mean(batch_loss))

        losses, r2 = evaluate_model(network, validation, device)
        val_loss_history.append(np.mean(losses))
        r2_scores.append(r2)

        print(
            f"(Epoch {epoch} / {epochs}) training loss: {train_loss_history[-1]:.4f}, validation loss: {val_loss_history[-1]:.4f}, validation R^2: {r2:.4f}")
    print()

    plt.plot(train_loss_history)
    plt.plot(val_loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss history")
    plt.savefig(f"{network.name()}_{epochs}_{lr}.png", dpi=300, bbox_inches="tight")
    plt.show()

    results = {
        "final train loss": train_loss_history[-1],
        "final validation loss": val_loss_history[-1],
        "final validation R^2": r2_scores[-1],
        "train_loss_history": train_loss_history,
        "val_loss_history": val_loss_history,
        "r2_scores": r2_scores
    }

    with open(f"{network.name()}_{epochs}_{lr}.json", "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    return train_loss_history, val_loss_history, r2_scores


def evaluate_model(network, loader, device):
    losses = []
    loss_function = nn.BCELoss()
    preds = torch.tensor([], device=device)
    labs = torch.tensor([], device=device)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            prediction = network(batch, device=device)
            label = batch["label"].to(device=device)

            loss = loss_function(prediction, label)

            losses.append(loss.item())
            preds = torch.cat((preds, prediction), dim=0)
            labs = torch.cat((labs, label), dim=0)

    ss_res = torch.sum((labs - preds) ** 2)
    ss_tot = torch.sum((preds - preds.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    return losses, r2.item()