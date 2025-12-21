"""
FGSM and PGD attack implementations (PyTorch)
"""
import torch

def fgsm_attack(model, images, labels, epsilon, loss_fn):
    images.requires_grad = True
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    model.zero_grad()
    loss.backward()
    data_grad = images.grad.data
    perturbed = images + epsilon * data_grad.sign()
    perturbed = torch.clamp(perturbed, 0, 1)
    return perturbed

def pgd_attack(model, images, labels, epsilon, alpha, iters, loss_fn):
    ori_images = images.data
    perturbed = images.clone().detach()
    for i in range(iters):
        perturbed.requires_grad = True
        outputs = model(perturbed)
        loss = loss_fn(outputs, labels)
        model.zero_grad()
        loss.backward()
        perturbed = perturbed + alpha * perturbed.grad.data.sign()
        eta = torch.clamp(perturbed - ori_images, min=-epsilon, max=epsilon)
        perturbed = torch.clamp(ori_images + eta, 0, 1).detach()
    return perturbed
