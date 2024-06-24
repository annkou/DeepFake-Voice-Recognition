from collections import defaultdict
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import torch


def train_model(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs=10,
    save_path="model.pth",
    device="cpu",
):

    best_val_f1 = 0.0
    train_losses = []
    val_losses = []
    val_f1s = []

    for epoch in range(num_epochs):

        # Set the model to training mode.
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            # Clear the gradients from the previous step.
            optimizer.zero_grad()
            outputs = model(images)

            # Ensure labels have the correct shape and type (as float).
            labels = labels.unsqueeze(1).float()

            # Compute the loss between the predictions and the actual labels.
            loss = criterion(outputs, labels)

            # Perform backpropagation to compute the gradients.
            loss.backward()
            optimizer.step()

            # Accumulate the batch loss.
            running_loss += loss.item() * images.size(0)

            # Convert the output logits to binary predictions. (0/1 using the sigmoid)
            preds = torch.round(torch.sigmoid(outputs))

            # Count the number of correct predictions.
            correct += (preds == labels).sum().item()

            # Update the model's weights based on the gradients.
            total += labels.size(0)

        # Calculate the average loss for the epoch.
        epoch_loss = running_loss / len(train_loader.dataset)

        # Calculate the accuracy for the epoch.
        epoch_acc = correct / total

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(
            model, criterion, val_loader, device
        )

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

        # Save the model if it has the best F1 score so far
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                save_path,
            )

            print(f"Model saved at epoch {epoch+1}")

    # Plotting after training
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(val_f1s, label="Val F1 Score", color="orange")
    ax2.set_title("F1 Score per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()

    plt.show()


def train_with_early_stopping(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    num_epochs=10,
    save_path="model.pth",
    patience=2,
    device="cpu",
):

    best_val_f1 = 0.0

    train_losses = []
    val_losses = []
    val_f1s = []

    consec_increases = 0

    for epoch in range(num_epochs):

        # Set the model to training mode.
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Clear the gradients from the previous step.
            optimizer.zero_grad()
            outputs = model(images)

            # Ensure labels have the correct shape and type (as float).
            labels = labels.unsqueeze(1).float()

            # Compute the loss between the predictions and the actual labels.
            loss = criterion(outputs, labels)

            # Perform backpropagation to compute the gradients.
            loss.backward()
            optimizer.step()

            # Accumulate the batch loss.
            running_loss += loss.item() * images.size(0)

            # Convert the output logits to binary predictions. (0/1 using the sigmoid)
            preds = torch.round(torch.sigmoid(outputs))

            # Count the number of correct predictions.
            correct += (preds == labels).sum().item()

            # Update the model's weights based on the gradients.
            total += labels.size(0)

        # Calculate the average loss for the epoch.
        epoch_loss = running_loss / len(train_loader.dataset)

        # Calculate the accuracy for the epoch.
        epoch_acc = correct / total

        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(
            model, criterion, val_loader, device
        )

        train_losses.append(epoch_loss)
        val_losses.append(val_loss)
        val_f1s.append(val_f1)

        print(
            f"Epoch {epoch+1}/{num_epochs}, "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Val Precision: {val_precision:.4f}, Val Recall: {val_recall:.4f}, Val F1: {val_f1:.4f}"
        )

        # Early stopping logic

        if epoch > 0 and val_losses[-1] > val_losses[-2]:
            consec_increases += 1
        else:
            consec_increases = 0

        if consec_increases == patience:
            print(
                f"Stopped early at epoch {epoch + 1} - val loss increased for {consec_increases} consecutive epochs!"
            )
            break

        # Save the model if it has the best F1 score so far

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                save_path,
            )
            print(f"Model saved at epoch {epoch+1}")

    # Plotting after training

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train Loss")
    ax1.plot(val_losses, label="Val Loss")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    ax2.plot(val_f1s, label="Val F1 Score", color="orange")
    ax2.set_title("F1 Score per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1 Score")
    ax2.legend()

    plt.show()


def evaluate_model(model, criterion, val_loader, device="cpu"):
    model.eval()

    val_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    sample_predictions = defaultdict(list)

    with torch.no_grad():

        for images, labels, original_samples in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            labels = labels.unsqueeze(1).float()

            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)

            preds = torch.round(torch.sigmoid(outputs))
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            # Collect predictions for each original sample
            for i, original_sample in enumerate(original_samples):
                sample_predictions[original_sample].append(preds[i].item())

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    val_precision = precision_score(all_labels, all_preds)
    val_recall = recall_score(all_labels, all_preds)
    val_f1 = f1_score(all_labels, all_preds)

    return val_loss, val_acc, val_precision, val_recall, val_f1
