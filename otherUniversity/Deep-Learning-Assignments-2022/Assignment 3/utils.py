'''This file contains utility functions for the assignment.'''
import torch
import seaborn as sns
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt


def train_model(model, train_loader, val_loader, epochs, criterion, optimizer,
                device, writer=None, verbose=True, teacher_model=None,
                alpha=0.5, T=10):
    """
    This function trains the model for the given number of epochs and returns
    loss and accuracy of train and val sets.

    Args:
    model: model to train
    train_loader: train data loader
    val_loader: validation data loader
    epochs: number of epochs to train
    criterion: loss function
    optimizer: optimizer
    device: device to use for training (cpu or gpu)
    tensorboard: if True, add loss and accuracy to tensorboard
    writer: tensorboard writer
    verbose: print loss and accuracy for each epoch or not
    teacher_model: teacher model for distillation
    alpha: weight for distillation loss
    T: temperature for distillation loss

    Returns:
    train_losses: list of train losses for each epoch
    val_losses: list of val losses for each epoch
    train_acc: list of train accuracies for each epoch
    val_acc: list of val accuracies for each epoch

    """
    train_losses = []
    val_losses = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        train_loss = 0
        val_loss = 0
        train_correct = 0
        val_correct = 0
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(images)
            if teacher_model is not None:
                with torch.no_grad():
                    teacher_output = teacher_model(images)
                loss = criterion(output, labels) * alpha + \
                    criterion(F.log_softmax(output / T, dim=1),
                    F.softmax(teacher_output / T, dim=1)) * (1 - alpha) * T ** 2
            else:
                loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(labels.view_as(pred)).sum().item()
        train_losses.append(train_loss / len(train_loader))
        train_acc.append(train_correct / len(train_loader.dataset))
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                if teacher_model is not None:
                    with torch.no_grad():
                        teacher_output = teacher_model(images)
                    loss = criterion(output, labels) * alpha + \
                        criterion(F.log_softmax(output / T, dim=1),
                        F.softmax(teacher_output / T, dim=1)) * (1 - alpha) * T ** 2
                else:
                    loss = criterion(output, labels)
                val_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(labels.view_as(pred)).sum().item()
        val_losses.append(val_loss / len(val_loader))
        val_acc.append(val_correct / len(val_loader.dataset))
        if verbose:
            print("Epoch: {}/{} \t Train loss: {:.3f} \t "
                  "Train accuracy: {:.3f} \t val loss: {:.3f} "
                  "\t val accuracy: {:.3f}"
                  .format(epoch + 1, epochs, train_loss / len(train_loader),
                          train_correct / len(train_loader.dataset),
                          val_loss / len(val_loader),
                          val_correct / len(val_loader.dataset)))
        if writer is not None:
            writer.add_scalar('Loss/train',
                              train_loss / len(train_loader), epoch)
            writer.add_scalar('Loss/val',
                              val_loss / len(val_loader), epoch)
            writer.add_scalar('Accuracy/train',
                              train_correct / len(train_loader.dataset), epoch)
            writer.add_scalar('Accuracy/val',
                              val_correct / len(val_loader.dataset), epoch)
            writer.close()
    return train_losses, val_losses, train_acc, val_acc

def print_metrics(model, test_loader, device, classes):
    """
    This function prints the classification report and confusion matrix for
    the test set.

    Args:
    model: model to evaluate
    test_loader: test data loader
    device: device to use for evaluation (cpu or gpu)
    classes: list of classes
    """
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            pred = output.argmax(dim=1, keepdim=True)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
    print(classification_report(y_true, y_pred, target_names=classes, digits=4))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.

    Args:
        model: model to evaluate

    Returns:
        number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
