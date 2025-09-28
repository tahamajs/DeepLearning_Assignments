import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def evaluate_domain_adaptation(source_model, target_model, source_loader, target_loader, class_names=None):
    """Evaluate domain adaptation performance."""
    device = next(source_model.parameters()).device
    
    source_model.eval()
    source_correct = 0
    source_total = 0
    source_preds = []
    source_labels = []
    
    with torch.no_grad():
        for images, labels in source_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = source_model(images)
            _, predicted = torch.max(outputs.data, 1)
            source_correct += (predicted == labels).sum().item()
            source_total += labels.size(0)
            source_preds.extend(predicted.cpu().numpy())
            source_labels.extend(labels.cpu().numpy())
    
    source_acc = source_correct / source_total
    
    target_correct = 0
    target_total = 0
    target_preds = []
    target_labels = []
    
    with torch.no_grad():
        for images, labels in target_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = target_model(images)
            _, predicted = torch.max(outputs.data, 1)
            target_correct += (predicted == labels).sum().item()
            target_total += labels.size(0)
            target_preds.extend(predicted.cpu().numpy())
            target_labels.extend(labels.cpu().numpy())
    
    target_acc = target_correct / target_total
    
    print(f"Source Domain Accuracy: {source_acc:.4f}")
    print(f"Target Domain Accuracy: {target_acc:.4f}")
    print(f"Accuracy Drop: {source_acc - target_acc:.4f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    cm_source = confusion_matrix(source_labels, source_preds)
    disp_source = ConfusionMatrixDisplay(confusion_matrix=cm_source, display_labels=class_names)
    disp_source.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Source Domain Confusion Matrix')
    
    cm_target = confusion_matrix(target_labels, target_preds)
    disp_target = ConfusionMatrixDisplay(confusion_matrix=cm_target, display_labels=class_names)
    disp_target.plot(ax=ax2, cmap='Oranges')
    ax2.set_title('Target Domain Confusion Matrix')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'source_accuracy': source_acc,
        'target_accuracy': target_acc,
        'accuracy_drop': source_acc - target_acc
    }

def visualize_domain_features(model, source_loader, target_loader, n_samples=1000):
    """Visualize feature distributions using t-SNE."""
    device = next(model.parameters()).device
    model.eval()
    
    source_features = []
    target_features = []
    source_labels = []
    target_labels = []
    
    with torch.no_grad():
        for images, labels in source_loader:
            images = images.to(device)
            features = model(images)
            source_features.append(features.cpu().numpy())
            source_labels.append(labels.numpy())
            if len(source_features[0]) >= n_samples:
                break
    
    with torch.no_grad():
        for images, labels in target_loader:
            images = images.to(device)
            features = model(images)
            target_features.append(features.cpu().numpy())
            target_labels.append(labels.numpy())
            if len(target_features[0]) >= n_samples:
                break
    
    source_features = np.concatenate(source_features)[:n_samples]
    target_features = np.concatenate(target_features)[:n_samples]
    source_labels = np.concatenate(source_labels)[:n_samples]
    target_labels = np.concatenate(target_labels)[:n_samples]
    
    all_features = np.concatenate([source_features, target_features])
    tsne = TSNE(n_components=2, random_state=42)
    features_2d = tsne.fit_transform(all_features)
    
    source_2d = features_2d[:n_samples]
    target_2d = features_2d[n_samples:]
    
    plt.figure(figsize=(10, 8))
    plt.scatter(source_2d[:, 0], source_2d[:, 1], c=source_labels, cmap='tab10', alpha=0.6, label='Source Domain', marker='o')
    plt.scatter(target_2d[:, 0], target_2d[:, 1], c=target_labels, cmap='tab10', alpha=0.6, label='Target Domain', marker='s')
    plt.colorbar()
    plt.title('Feature Distribution: Source vs Target Domain (t-SNE)')
    plt.legend()
    plt.show()






import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import pandas as pd
import os
import pickle
from torchinfo import summary
import torch.nn as nn
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import precision_recall_fscore_support as score
from collections import defaultdict
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


from google.colab import drive
drive.mount('/content/drive/')
path = '/content/drive/MyDrive/Colab/NNDL/CA6/Part1/'


dataset_path = os.path.join(path,'Dataset/dataset.zip')


with open('mnist.pkl', 'rb') as file:
    mnist = pickle.load(file)
with open('mnistm.pkl', 'rb') as file:
    mnistm = pickle.load(file)

list(mnist.keys()),len(mnist[b'labels']),len(mnistm[b'labels'])


n_sample = 5
indices = np.random.choice(len(mnist[b'labels']),n_sample)

plt.figure(figsize = (10,4))
for i in range(n_sample):
    img = mnist[b'images'][indices[i]]
    label = mnist[b'labels'][indices[i]]
    plt.subplot(2,n_sample,i+1)
    plt.imshow(img,cmap='gray')
    plt.title(label)
    plt.axis('off')

    img = mnistm[b'images'][indices[i]]
    plt.subplot(2,n_sample,i+1+n_sample)
    plt.imshow(img)
    plt.title(label)
    plt.axis('off')


class MNISTDataset(Dataset):
    def __init__(self, data_dict, indices, transform=None):
        self.images = data_dict[b'images'][indices]
        self.labels = data_dict[b'labels'][indices]
        self.transform = transform

    def set_transform(self,transform):
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label


transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((32,32)),
                                transforms.Normalize(mean=0.5, std=0.5)])


num_data = len(mnist[b'labels'])
train_size = int(0.8 * num_data)

indices = np.random.permutation(num_data)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

traindataset = MNISTDataset(data_dict=mnist, indices=train_indices, transform=transform)
testdataset = MNISTDataset(data_dict=mnist, indices=test_indices, transform=transform)

traindataset_m = MNISTDataset(data_dict=mnistm, indices=train_indices, transform=transform)
testdataset_m = MNISTDataset(data_dict=mnistm, indices=test_indices,transform=transform)
len(traindataset),len(testdataset),len(traindataset_m),len(testdataset_m)


batch_size=32
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=True, num_workers=2)

testloader = DataLoader(testdataset, batch_size=batch_size, shuffle=False, num_workers=2)

trainloader_m = DataLoader(traindataset_m, batch_size=batch_size, shuffle=True, num_workers=2)

testloader_m = DataLoader(testdataset_m, batch_size=batch_size, shuffle=False, num_workers=2)


def initialize_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes, initialize_weights=None, shared = None):
        super().__init__()
        self.shared = shared
        self.private = nn.Sequential(
            nn.Conv2d(in_channels,32,5),
            nn.MaxPool2d(2,2),
        )
        self.shared = shared
        if shared == None:
          self.shared = nn.Sequential(
              nn.Conv2d(32,48,5),
              nn.MaxPool2d(2,2),
              nn.Flatten(),
              nn.Linear(1200, 100),
              nn.ReLU(),
              nn.Linear(100, 100),
              nn.ReLU(),
              nn.Linear(100, num_classes),
          )
        if initialize_weights != None:
            self.private.apply(initialize_weights)
            self.shared.apply(initialize_weights)

    def get_shared(self):
        return self.shared

    def get_private(self):
        return self.private

    def forward(self, x):
        x = self.private(x)
        return self.shared(x)

summary(Classifier(1,10,initialize_weights),(1,1,32,32))


def get_predictions(model,data_loader,device):
    y_pred = []
    y_true = []
    model = model.eval().to(device)

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            y_pred.append(outputs.argmax(dim=-1).cpu().numpy())

        y_true.append(labels.cpu().numpy())

    y_true = np.concatenate(y_true, axis=0).flatten()
    y_pred = np.concatenate(y_pred, axis=0).flatten()
    return y_pred, y_true


def evaluate(predictions, actual_values,model_names):
    scores = defaultdict(lambda: defaultdict(float))
    for i, name in enumerate(model_names):
        precision, recall, fscore, support = score(actual_values[i], predictions[i])

        scores[name]["Accuracy"] = accuracy_score(actual_values[i],predictions[i])
        scores[name]["Precision"] = precision_score(actual_values[i],predictions[i],average='weighted')
        scores[name]["Recall"] = recall_score(actual_values[i],predictions[i],average='weighted')
        scores[name]["F1 score"] = f1_score(actual_values[i],predictions[i],average='weighted')

        cm=confusion_matrix(actual_values[i],predictions[i], normalize='true')
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, cmap="viridis")
        plt.title(name)
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.show()

    return pd.DataFrame(scores)


def train_epoch(model, data_loader, criterion, optimizer,scheduler, device):
    model.train()
    num_batches = len(data_loader)
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()
        if scheduler !=None:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / num_batches, correct / total

def validation_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / num_batches, correct / total


def train(model, train_loader, val_loader, criterion, optimizer, epochs, device, scheduler=None, report_val = True):
  hist = {
      "train_loss": [],
      "train_accuracy": [],
      "test_loss": [],
      "test_accuracy": [],
  }

  model = model.to(device)
  for epoch in range(1, epochs + 1):
    train_loss,train_accuracy = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
    hist['train_loss'].append(train_loss)
    hist['train_accuracy'].append(train_accuracy)
    print(f"Epoch [{epoch}] Average Train Loss: {train_loss:.4f} Average Train Accuracy: {train_accuracy:.4f}")

    if report_val:
        val_loss,val_accuracy = validation_epoch(model, val_loader, criterion, device)
        hist['test_loss'].append(val_loss)
        hist['test_accuracy'].append(val_accuracy)
        print(f"Epoch [{epoch}] Average Test Loss: {val_loss:.4f} Average Test Accuracy: {val_accuracy:.4f}")

  return hist


def plot_history(history,metric_name,key,min_ylim=None,max_ylim=None):
    range_epochs = range(1,len(history[f'test_{key}'])+1)
    plt.plot(range_epochs,history[f'train_{key}'])
    plt.plot(range_epochs,history[f'test_{key}'])
    plt.title(f"{metric_name} over epochs")
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    if min_ylim==None:
        min_ylim = min(min(history[f'train_{key}']),min(history[f'test_{key}']))*0.99
    if max_ylim==None:
        max_ylim = max(max(history[f'train_{key}']),max(history[f'test_{key}']))*1.01
    plt.ylim(min_ylim, max_ylim)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


transform_3ch = transforms.Compose([transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3,1,1)),
                                transforms.Resize((32,32)),
                                transforms.Normalize(mean=0.5, std=0.5)])
traindataset.set_transform(transform_3ch)
testdataset.set_transform(transform_3ch)


epochs = 50
lr = 1e-3
weight_decay = 1e-5

model = Classifier(3,10,initialize_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()

lr_lambda = lambda step: 0.95 ** (step // 20000)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

hist = train(model, trainloader, testloader, criterion, optimizer, epochs, device,scheduler=scheduler)

state_dict = model.state_dict()
torch.save(state_dict, 'classifier_mnist.pth')

plot_history(hist,'Source Test Loss','loss',min_ylim=0)
plot_history(hist,'Source Test Accuracy','accuracy',max_ylim=1)


y_pred = []
y_true = []
model_names = ['MNIST Test','MNIST-M Train+Test']

pred, true = get_predictions(model,testloader,device)
y_pred.append(pred)
y_true.append(true)

dataset_m = MNISTDataset(data_dict=mnistm, indices=np.arange(num_data), transform=transform)
loader_m = DataLoader(dataset_m, batch_size=batch_size, shuffle=False, num_workers=2)

pred, true = get_predictions(model,loader_m,device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred,y_true,model_names)
scores


traindataset.set_transform(transform)
testdataset.set_transform(transform)


class Generator(nn.Module):
    class ResidualBlock(nn.Module):
        def __init__(self, channels,initialize_weights=None):
            super().__init__()
            self.block = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.Conv2d(channels, channels, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channels)
            )
            if initialize_weights != None:
                self.block.apply(initialize_weights)

        def forward(self, x):
            return x + self.block(x)

    def __init__(self, noise_dim, input_size,in_channels, base_channels=64,
                 out_channels=3, initialize_weights=None):
        super().__init__()
        self.fc = nn.Linear(noise_dim, input_size * input_size)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels+1, base_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            self.ResidualBlock(base_channels,initialize_weights),
            self.ResidualBlock(base_channels,initialize_weights),
            self.ResidualBlock(base_channels,initialize_weights),
            self.ResidualBlock(base_channels,initialize_weights),
            self.ResidualBlock(base_channels,initialize_weights),
            self.ResidualBlock(base_channels,initialize_weights),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        if initialize_weights != None:
            self.fc.apply(initialize_weights)
            self.model.apply(initialize_weights)

    def forward(self, x_s, z):
        z_proj = self.fc(z).view(z.size(0), 1, x_s.size(2), x_s.size(3))
        x = torch.cat([x_s, z_proj], dim=1)
        x = self.model(x)
        return x

summary(Generator(10,32,1,64,3,initialize_weights), [(1, 1, 32, 32), (1, 10,)])


class Discriminator(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels,out_channels,stride,noise_mean,noise_std,initialize_weights=None):
            super().__init__()
            self.noise_std = noise_std
            self.noise_mean = noise_mean
            self.block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2),
                nn.Dropout(p=0.1),
            )
            if initialize_weights != None:
                self.block.apply(initialize_weights)

        def forward(self, x):
            x = self.block(x)
            noise = 0
            if self.training:
                noise = torch.randn_like(x) * self.noise_std + self.noise_mean
            return x+noise

    def __init__(self, noise_dim, base_channels=64,noise_mean=0,noise_std=0.2, initialize_weights=None):
        super().__init__()
        self.model = nn.Sequential(
            self.ConvBlock(3,64,1,noise_mean,noise_std,initialize_weights),
            self.ConvBlock(64,128,2,noise_mean,noise_std,initialize_weights),
            self.ConvBlock(128,256,2,noise_mean,noise_std,initialize_weights),
            self.ConvBlock(256,512,2,noise_mean,noise_std,initialize_weights),
            nn.Flatten(),
            nn.Linear(8192,1),
            nn.Sigmoid()
        )
        if initialize_weights != None:
            self.model.apply(initialize_weights)

    def forward(self, x):
        return self.model(x)

summary(Discriminator(10,64,0,0.2,initialize_weights),(1,3,32,32))


noise_dim = 10

class ModelStruct:
    model = None
    optimizer = None
    criterion = None
    scheduler = None

    def step(self,is_train,loss):
        if is_train:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler !=None:
                self.scheduler.step()


def one_epoch_gan(D,G,ST,FT,d_w,g_w,t_w,s_loader,t_loader,is_train,device):
    num_batches = len(s_loader)
    d_total_loss = 0
    g_total_loss = 0
    t_total_loss = 0
    s_correct = 0
    t_correct = 0
    f_correct = 0
    total = 0

    for (s_images, s_labels), (t_images, t_labels) in zip(s_loader,t_loader):
        s_images = s_images.to(device)
        t_images = t_images.to(device)
        s_labels = s_labels.to(device)
        t_labels = t_labels.to(device)

        batch_size = s_labels.size(0)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        if is_train:
            D.model.train()
            G.model.train()
            ST.model.train()
            FT.model.train()
            torch.set_grad_enabled(True)
        else:
            D.model.eval()
            G.model.eval()
            ST.model.eval()
            FT.model.eval()
            torch.set_grad_enabled(False)

        noise = torch.rand(batch_size, noise_dim).to(device) * 2 - 1
        f_images = G.model(s_images,noise)

        t_output = D.model(t_images)
        f_output = D.model(f_images.detach())

        d_loss_real = D.criterion(t_output, real_labels)
        d_loss_fake = D.criterion(f_output, fake_labels)
        d_loss = d_loss_real + d_loss_fake
        d_total_loss += d_loss.item()
        d_loss *= d_w
        D.step(is_train,d_loss)

        s_output = ST.model(s_images)
        t_output = FT.model(t_images)
        f_output = FT.model(f_images.detach())

        t_loss_real = ST.criterion(s_output, s_labels)
        t_loss_fake = FT.criterion(f_output, s_labels)
        t_loss = t_loss_real + t_loss_fake
        t_total_loss += t_loss.item()
        ST.step(is_train,t_loss)

        _, s_predicted = torch.max(s_output, 1)
        s_correct += (s_predicted == s_labels).sum().item()
        _, t_predicted = torch.max(t_output, 1)
        t_correct += (t_predicted == t_labels).sum().item()
        _, f_predicted = torch.max(f_output, 1)
        f_correct += (f_predicted == s_labels).sum().item()
        total += real_labels.size(0)

        f_output = D.model(f_images)
        g_loss_d = D.criterion(f_output, real_labels)
        f_output = FT.model(f_images)

        g_loss_t = FT.criterion(f_output, s_labels)
        g_loss = g_loss_d + g_loss_t * t_w
        g_total_loss += g_loss.item()
        g_loss *= g_w
        G.step(is_train,g_loss)

    d_total_loss /= num_batches
    g_total_loss /= num_batches
    t_total_loss /= num_batches
    s_correct /= total
    t_correct /= total
    f_correct /= total
    return d_total_loss,g_total_loss,t_total_loss,s_correct,t_correct,f_correct


def train_gan(D,G,ST,FT,d_w,g_w,t_w, train_loader, test_loader, train_loaderm, test_loaderm, epochs, device):
  hist = {
      "d_train_loss": [],
      "g_train_loss": [],
      "c_train_loss": [],
      "s_train_accuracy": [],
      "t_train_accuracy": [],
      "f_train_accuracy": [],
      "d_test_loss": [],
      "g_test_loss": [],
      "c_test_loss": [],
      "s_test_accuracy": [],
      "t_test_accuracy": [],
      "f_test_accuracy": [],
  }

  D.model = D.model.to(device)
  G.model = G.model.to(device)
  ST.model = ST.model.to(device)
  FT.model = FT.model.to(device)
  for epoch in range(1, epochs + 1):
    d_loss,g_loss,t_loss,s_accuracy,t_accuracy,f_accuracy = one_epoch_gan(
        D,G,ST,FT,d_w,g_w,t_w,train_loader,train_loaderm,True,device)
    hist['d_train_loss'].append(d_loss)
    hist['g_train_loss'].append(g_loss)
    hist['c_train_loss'].append(t_loss)
    hist['s_train_accuracy'].append(s_accuracy)
    hist['t_train_accuracy'].append(t_accuracy)
    hist['f_train_accuracy'].append(f_accuracy)
    total_loss = d_loss*d_w+g_loss*g_w+t_loss*t_w
    print(f"Epoch [{epoch}] Train Loss: Discriminator: {d_loss:.4f} Generator: {g_loss:.4f} Classifier: {t_loss:.4f} Train Accuracy: Source: {s_accuracy:.4f} Target: {t_accuracy:.4f} Fake: {f_accuracy:.4f}")

    d_loss,g_loss,t_loss,s_accuracy,t_accuracy,f_accuracy = one_epoch_gan(
        D,G,ST,FT,d_w,g_w,t_w,test_loader,test_loaderm,False,device)
    hist['d_test_loss'].append(d_loss)
    hist['g_test_loss'].append(g_loss)
    hist['c_test_loss'].append(t_loss)
    hist['s_test_accuracy'].append(s_accuracy)
    hist['t_test_accuracy'].append(t_accuracy)
    hist['f_test_accuracy'].append(f_accuracy)
    total_loss = d_loss*d_w+g_loss*g_w+t_loss*t_w
    print(f"Epoch [{epoch}] Test Loss: Discriminator: {d_loss:.4f} Generator: {g_loss:.4f} Classifier: {t_loss:.4f} Test Accuracy: Source: {s_accuracy:.4f} Target: {t_accuracy:.4f} Fake: {f_accuracy:.4f}")

  return hist


epochs = 50
lr = 1e-3
weight_decay = 1e-5
betas = (0.5, 0.999)
lr_lambda = lambda step: 0.95 ** (step // 20000)

G = ModelStruct()
G.model = Generator(noise_dim,32,1,64,3,initialize_weights).to(device)
G.optimizer = torch.optim.Adam(G.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
G.scheduler = torch.optim.lr_scheduler.LambdaLR(G.optimizer, lr_lambda=lr_lambda)

D = ModelStruct()
D.model = Discriminator(noise_dim,64,0,0.2,initialize_weights).to(device)
D.criterion = nn.BCELoss()
D.optimizer = torch.optim.Adam(D.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
D.scheduler = torch.optim.lr_scheduler.LambdaLR(D.optimizer, lr_lambda=lr_lambda)

ST = ModelStruct()
ST.model = Classifier(1,10,initialize_weights).to(device)
ST.criterion = nn.CrossEntropyLoss()

FT = ModelStruct()
FT.model = Classifier(3,10,initialize_weights,ST.model.get_shared()).to(device)
FT.criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(list(ST.model.parameters()) + list(FT.model.get_private().parameters()),
    lr=lr, betas=betas, weight_decay=weight_decay)
ST.optimizer = optimizer
ST.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
FT.optimizer = optimizer
FT.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

hist = train_gan(D,G,ST,FT,0.013,0.011,0.01, trainloader, testloader,
                 trainloader_m, testloader_m, epochs, device)


def plot_history(history,metric_name,train_key,test_key,min_ylim=None,max_ylim=None):
    range_epochs = range(1,len(history[test_key])+1)
    plt.plot(range_epochs,history[train_key])
    plt.plot(range_epochs,history[test_key])
    plt.title(f"{metric_name} over epochs")
    plt.ylabel(metric_name)
    plt.xlabel('Epoch')
    if min_ylim==None:
        min_ylim = min(min(history[train_key]),min(history[test_key]))*0.99
    if max_ylim==None:
        max_ylim = max(max(history[train_key]),max(history[test_key]))*1.01
    plt.ylim(min_ylim, max_ylim)
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()


for key in hist.keys():
    name = ''
    if key[0] == 'd':
        name = 'Discriminator '
    elif key[0] == 'g':
        name = 'Generator '
    elif key[0] == 'c':
        name = 'Classifier '
    elif key[0] == 's':
        name = 'Source '
    elif key[0] == 't':
        name = 'Target '
    elif key[0] == 'f':
        name = 'Fake '
    if 'train' in key:
      test_key = key.replace('train','test')
    else:
      continue
    if 'loss' in key:
        plot_history(hist,name+'Loss',key,test_key,min_ylim=0,max_ylim=None)
    else:
        plot_history(hist,name+'Accuracy',key,test_key,min_ylim=None,max_ylim=1)


y_pred = []
y_true = []
model_names = ['MNIST Test','MNIST-M Test','Fake MNIST-M Test']

pred, true = get_predictions(ST.model,testloader,device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(FT.model,testloader_m,device)
y_pred.append(pred)
y_true.append(true)

class GenerativeDataLoader(DataLoader):
    def __init__(self, *args, model, noise_dim, device, noise=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model.eval().to(device)
        self.noise_dim = noise_dim
        self.device = device
        self.noise = noise

    def __iter__(self):
        data_iter = super().__iter__()
        for images, labels in data_iter:
            images, labels = images.to(self.device), labels.to(self.device)
            noise = self.noise
            if noise==None:
                noise = torch.rand(images.size(0), self.noise_dim).to(self.device) * 2 - 1
            with torch.no_grad():
              images = self.model(images,noise)
            yield images, labels

testloader_g = GenerativeDataLoader(testdataset, model=G.model, noise_dim=10,
            batch_size=batch_size, shuffle=False, num_workers=2,device=device,)

pred, true = get_predictions(FT.model,testloader_g,device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred,y_true,model_names)
scores


state_dict = G.model.state_dict()
torch.save(state_dict, 'g.pth')
state_dict = D.model.state_dict()
torch.save(state_dict, 'd.pth')
state_dict = ST.model.state_dict()
torch.save(state_dict, 'st.pth')
state_dict = FT.model.state_dict()
torch.save(state_dict, 'ft.pth')





def find_most_similar(data_loader,image):
    min_distance = np.inf
    min_image = None
    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        distances = ((images - image) ** 2).view(images.size(0), -1).sum(dim=1)
        min_val, min_idx = torch.min(distances, dim=0)
        if min_val < min_distance:
            min_distance = min_val.item()
            min_image = images[min_idx]

    return min_image


dataiter = iter(testloader)
dataiter_m = iter(testloader_m)
dataiter_g = iter(testloader_g)

def postprocess_image(image):
    image = image * 0.5 + 0.5
    image = image.cpu().permute(1, 2, 0).numpy()
    return image


n_sample = 6
plt.figure(figsize = (11,8))

images,labels = next(dataiter)
images_m,_ = next(dataiter_m)
images_g,_ = next(dataiter_g)

texts = ['MNIST','MNIST-M','Generated','Most Similar']
for i,text in enumerate(texts):
    plt.subplot(4,n_sample,1+n_sample*i)
    plt.text(0, 0.5, text, fontsize=12, verticalalignment='center')
    plt.axis('off')

for i in range(n_sample-1):
    image = images[i].to(device)
    label = labels[i]
    plt.subplot(4,n_sample,i+2)
    plt.imshow(postprocess_image(image),cmap='gray')
    with torch.no_grad():
        output = ST.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')

    image = images_m[i].to(device)
    plt.subplot(4,n_sample,i+2+n_sample)
    plt.imshow(postprocess_image(image))
    with torch.no_grad():
        output = FT.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')

    image = images_g[i].to(device)
    plt.subplot(4,n_sample,i+2+n_sample*2)
    plt.imshow(postprocess_image(image))
    with torch.no_grad():
        output = FT.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')

    similar_img = find_most_similar(trainloader_m,image)
    plt.subplot(4,n_sample,i+2+n_sample*3)
    plt.imshow(postprocess_image(similar_img))
    with torch.no_grad():
        output = FT.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')


plt.figure(figsize = (10,10))
shown = [False for _ in range(10)]
noises = torch.rand(10, 10).to(device) * 2 - 1
for image,label in testdataset:
    if shown[label]:
        continue
    shown[label] = True
    image = image.unsqueeze(0).to(device)
    for i,noise in enumerate(noises):
        plt.subplot(10,10,10*label+i+1)
        with torch.no_grad():
            image_g = G.model(image,noise.unsqueeze(0)).squeeze(0)
            plt.imshow(postprocess_image(image_g))
            plt.axis('off')
    if sum(shown)==10:
        break


traindataset.set_transform(transform_3ch)
testdataset.set_transform(transform_3ch)


epochs = 20
lr = 1e-3
weight_decay = 1e-5
betas = (0.5, 0.999)
lr_lambda = lambda step: 0.95 ** (step // 20000)

G = ModelStruct()
G.model = Generator(noise_dim,32,3,64,3,initialize_weights).to(device)
G.optimizer = torch.optim.Adam(G.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
G.scheduler = torch.optim.lr_scheduler.LambdaLR(G.optimizer, lr_lambda=lr_lambda)

D = ModelStruct()
D.model = Discriminator(noise_dim,64,0,0.2,initialize_weights).to(device)
D.criterion = nn.BCELoss()
D.optimizer = torch.optim.Adam(D.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
D.scheduler = torch.optim.lr_scheduler.LambdaLR(D.optimizer, lr_lambda=lr_lambda)

T = ModelStruct()
T.model = Classifier(3,10,initialize_weights).to(device)
T.criterion = nn.CrossEntropyLoss()
T.optimizer = torch.optim.Adam(T.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
T.scheduler = torch.optim.lr_scheduler.LambdaLR(T.optimizer, lr_lambda=lr_lambda)

hist = train_gan(D,G,T,T,0.013,0.011,0.01, trainloader, testloader,
                 trainloader_m, testloader_m, epochs, device)


for key in hist.keys():
    name = ''
    if key[0] == 'd':
        name = 'Discriminator '
    elif key[0] == 'g':
        name = 'Generator '
    elif key[0] == 'c':
        name = 'Classifier '
    elif key[0] == 's':
        name = 'Source '
    elif key[0] == 't':
        name = 'Target '
    elif key[0] == 'f':
        name = 'Fake '
    if 'train' in key:
      test_key = key.replace('train','test')
    else:
      continue
    if 'loss' in key:
        plot_history(hist,name+'Loss',key,test_key,min_ylim=0,max_ylim=None)
    else:
        plot_history(hist,name+'Accuracy',key,test_key,min_ylim=None,max_ylim=1)


y_pred = []
y_true = []
model_names = ['MNIST Test','MNIST-M Test','Fake MNIST-M Test']

pred, true = get_predictions(T.model,testloader,device)
y_pred.append(pred)
y_true.append(true)

pred, true = get_predictions(T.model,testloader_m,device)
y_pred.append(pred)
y_true.append(true)

testloader_g = GenerativeDataLoader(testdataset, model=G.model, noise_dim=10,
            batch_size=batch_size, shuffle=False, num_workers=2,device=device,)

pred, true = get_predictions(T.model,testloader_g,device)
y_pred.append(pred)
y_true.append(true)

scores = evaluate(y_pred,y_true,model_names)
scores


state_dict = G.model.state_dict()
torch.save(state_dict, 'g1.pth')
state_dict = D.model.state_dict()
torch.save(state_dict, 'd1.pth')
state_dict = T.model.state_dict()
torch.save(state_dict, 't1.pth')







dataiter = iter(testloader)
dataiter_m = iter(testloader_m)
dataiter_g = iter(testloader_g)


n_sample = 6
plt.figure(figsize = (11,8))

images,labels = next(dataiter)
images_m,_ = next(dataiter_m)
images_g,_ = next(dataiter_g)

texts = ['MNIST','MNIST-M','Generated','Most Similar']
for i,text in enumerate(texts):
    plt.subplot(4,n_sample,1+n_sample*i)
    plt.text(0, 0.5, text, fontsize=12, verticalalignment='center')
    plt.axis('off')

for i in range(n_sample-1):
    image = images[i].to(device)
    label = labels[i]
    plt.subplot(4,n_sample,i+2)
    plt.imshow(postprocess_image(image),cmap='gray')
    with torch.no_grad():
        output = T.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')

    image = images_m[i].to(device)
    plt.subplot(4,n_sample,i+2+n_sample)
    plt.imshow(postprocess_image(image))
    with torch.no_grad():
        output = T.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')

    image = images_g[i].to(device)
    plt.subplot(4,n_sample,i+2+n_sample*2)
    plt.imshow(postprocess_image(image))
    with torch.no_grad():
        output = T.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')

    similar_img = find_most_similar(trainloader_m,image)
    plt.subplot(4,n_sample,i+2+n_sample*3)
    plt.imshow(postprocess_image(similar_img))
    with torch.no_grad():
        output = T.model(image.unsqueeze(0)).argmax(dim=-1)[0]
    plt.title(f'Classified as: {output}')
    plt.axis('off')


plt.figure(figsize = (10,10))
shown = [False for _ in range(10)]
noises = torch.rand(10, 10).to(device) * 2 - 1
for image,label in testdataset:
    if shown[label]:
        continue
    shown[label] = True
    image = image.unsqueeze(0).to(device)
    for i,noise in enumerate(noises):
        plt.subplot(10,10,10*label+i+1)
        with torch.no_grad():
            image_g = G.model(image,noise.unsqueeze(0)).squeeze(0)
            plt.imshow(postprocess_image(image_g))
            plt.axis('off')
    if sum(shown)==10:
        break

