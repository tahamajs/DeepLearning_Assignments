import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from PIL import Image
import torch.nn as nn
import torchvision.models as models
from collections import defaultdict
from torchinfo import summary
import kagglehub
import torch
import os
import re
import json


path = kagglehub.dataset_download("adityajn105/flickr8k")


captions_path = os.path.join(path, "captions.txt")
images_dir = os.path.join(path, "Images/")
file = open(captions_path, "r")
lines = file.readlines()
image_files = os.listdir(images_dir)

nsample = 3
samples = np.random.choice(len(os.listdir(images_dir)), nsample)
plt.figure(figsize=(10 * nsample, 10))
for i, idx in enumerate(samples):
    image_file = image_files[idx]
    img = Image.open(os.path.join(images_dir, image_file))
    caption = None
    for line in lines[1:]:
        line = line.strip().split(",", 1)
        if image_file == line[0]:
            caption = line[1]
            break

    plt.subplot(1, nsample, i + 1)
    plt.imshow(img)
    plt.title(caption, fontsize=20)
    plt.axis("off")
file.close()


len(image_files)


def preprocess_captions(path):
    file = open(path, "r")
    captions = defaultdict(list)

    lines = file.readlines()
    for line in lines[1:]:
        line = line.strip().split(",", 1)
        caption = line[1].lower()
        caption = re.sub("[^a-z ]+", "", caption)[:-1]
        caption = " ".join(caption.split())
        captions[line[0]].append(caption)

    file.close()
    return captions


np.random.seed(42)
captions = preprocess_captions(os.path.join(path, "captions.txt"))
shuffled_keys = np.random.permutation(list(captions.keys()))

test_ratio = 0.1
val_ratio = 0.1
num_val = int(val_ratio * len(captions))
num_test = int(test_ratio * len(captions))
num_train = len(captions) - num_val - num_test

train_captions = dict({key: captions[key] for key in shuffled_keys[:num_train]})
val_captions = dict(
    {key: captions[key] for key in shuffled_keys[num_train : num_val + num_train]}
)
test_captions = dict(
    {key: captions[key] for key in shuffled_keys[num_val + num_train :]}
)


len(train_captions), len(val_captions), len(test_captions)


print("Max number of tokens in train captions:")
train_num_tokens = np.array(
    [caption.count(" ") + 1 for img in train_captions.values() for caption in img]
)
print(train_num_tokens.max())

print("Max number of tokens in validation captions:")
val_num_tokens = np.array(
    [caption.count(" ") + 1 for img in val_captions.values() for caption in img]
)
print(val_num_tokens.max())

print("Max number of tokens in test captions:")
test_num_tokens = np.array(
    [caption.count(" ") + 1 for img in test_captions.values() for caption in img]
)
print(test_num_tokens.max())


class special_tokens:
    Start = "<sos>"
    End = "<eos>"
    Pad = "<pad>"
    Unknown = "<unk>"
    list_of_tokens = [Start, End, Pad, Unknown]


class Tokenizer:
    def __init__(self, captions, unknown_percent=0.1, dictionary=None):
        self.unknown_percent = unknown_percent
        self.unknown_words = []
        self.context_length = (
            max(caption.count(" ") for img in captions.values() for caption in img) + 3
        )
        self.total_tokens = 0
        self.total_unknown_tokens = 0
        if dictionary == None:
            self.dictionary = self._create_dictionary(train_captions)
            with open("tokenizer.json", "w") as f:
                json.dump(self.dictionary, f)
        else:
            self.dictionary = dictionary

    def _create_dictionary(self, targets):
        dictionary = dict(
            {token: i for i, token in enumerate(special_tokens.list_of_tokens)}
        )

        frequencies = defaultdict(int)
        for captions in targets.values():
            caption = " ".join(captions)
            tokens = caption.split(" ")
            for token in tokens:
                frequencies[token] += 1
                self.total_tokens += 1

        frequencies = sorted(list(frequencies.items()), key=lambda x: x[1])
        freq_threshold = frequencies[int(self.unknown_percent * len(frequencies))][1]
        for token, frequency in frequencies:
            if frequency <= freq_threshold:
                self.unknown_words.append((token, frequency))
                self.total_unknown_tokens += frequency
            else:
                dictionary[token] = len(dictionary)

        return dictionary

    def tokenize(self, text: str):
        return text.split(" ")

    def text_to_ids(self, text: str):
        tokens = self.tokenize(text)
        dictionary = self.dictionary
        ids = [dictionary[special_tokens.Pad] for _ in range(self.context_length)]
        ids[0] = dictionary[special_tokens.Start]

        for i in range(1, min(self.context_length - 1, len(tokens) + 1)):
            token = tokens[i - 1]
            if token in dictionary:
                ids[i] = dictionary[token]
            else:
                ids[i] = dictionary[special_tokens.Unknown]

        ids[len(tokens) + 1] = dictionary[special_tokens.End]
        return ids

    def ids_to_text(self, ids: list, ignore_special_tokens=False):
        if not ignore_special_tokens:
            return " ".join([list(self.dictionary.keys())[id] for id in ids])
        else:
            res = ""

            for id in ids:
                if id == special_tokens.End:
                    break
                if id < len(special_tokens.list_of_tokens):
                    continue

                res += list(self.dictionary.keys())[id]
                res += " "
            return res.strip()


tokenizer = Tokenizer(train_captions, unknown_percent=0.1)
len_dictionary = len(tokenizer.dictionary)
len_unk = len(tokenizer.unknown_words)
total_tokens = tokenizer.total_tokens
total_unknown_tokens = tokenizer.total_unknown_tokens
print("Numer of unqiue tokens in captions:", len_dictionary + len_unk)
print("Numer of tokens in dictionary:", len_dictionary)
print("Numer of unknown words in captions:", len_unk)
print(
    "Percent of unknown tokens to all tokens in dictionary:",
    len_unk * 100 / (len_dictionary + len_unk),
)
print("Total number of tokens in captions:", total_tokens)
print(
    "Percent of unknown tokens to all tokens in captions:",
    total_unknown_tokens * 100 / total_tokens,
)
print("List of some unknown words in captions:", tokenizer.unknown_words[:10])


text = list(train_captions.values())[0][0]
print("Text:", text)
ids = tokenizer.text_to_ids(text)
print("IDs:", ids)
output_text = tokenizer.ids_to_text(ids)
print("Output text with special tokens:", output_text)
output_text = tokenizer.ids_to_text(ids, True)
print("Output text without special tokens:", output_text)


class Flickr8kDataset(Dataset):
    def __init__(self, image_dir, captions, tokenizer, transform=None):
        self.images_dir = image_dir
        self.transform = transform
        self.captions = captions
        self.ids = defaultdict(list)
        for img in captions.keys():
            self.ids[img] = np.array(
                [tokenizer.text_to_ids(caption) for caption in captions[img]]
            )
        self.tokenizer = tokenizer
        self.image_files = list(captions.keys())

    def __len__(self):
        return len(self.image_files)

    def get_transform(self):
        return self.transform

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        image_file = self.image_files[index]
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        ids = self.ids[image_file]
        return image, ids


loader = Flickr8kDataset(os.path.join(path, "Images"), train_captions, tokenizer)
image, captions = loader[0]
print(captions)
print(tokenizer.ids_to_text(captions[0]))
image


weights = models.ResNet50_Weights.DEFAULT
transform = weights.transforms()


train_dataset = Flickr8kDataset(
    os.path.join(path, "Images"), train_captions, tokenizer, transform=transform
)
val_dataset = Flickr8kDataset(
    os.path.join(path, "Images"), val_captions, tokenizer, transform=transform
)
test_dataset = Flickr8kDataset(
    os.path.join(path, "Images"), test_captions, tokenizer, transform=transform
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


resnet = models.resnet50(weights=weights)


resnet


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        resnet = models.resnet50(weights=weights)
        modules = list(resnet.children())[:-1]
        encoder = nn.Sequential(*modules)
        for param in encoder.parameters():
            param.requires_grad = False
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x


encoder = Encoder()
dummy_input = torch.randn(1, 3, 224, 224)
features = encoder(dummy_input)
features.shape


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm_cell = nn.LSTMCell(
            input_size=embedding_dim,
            hidden_size=hidden_size,
        )
        self.gru_cell = nn.GRUCell(
            input_size=hidden_size,
            hidden_size=hidden_size,
        )
        self.linear = nn.Linear(in_features=hidden_size, out_features=vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        batch_size = features.size(0)
        lstm_hidden = features
        lstm_cell = features
        gru_hidden = features
        outputs = torch.empty(
            (batch_size, captions.size(1) - 1, self.vocab_size), device=self.device
        )
        embeddings = self.embedding(captions)
        embeddings = self.dropout(embeddings)

        for t in range(1, captions.size(1)):
            lstm_hidden, lstm_cell = self.lstm_cell(
                embeddings[:, t - 1, :], (lstm_hidden, lstm_cell)
            )
            lstm_cell = self.dropout(lstm_cell)
            gru_hidden = self.gru_cell(lstm_cell, gru_hidden)
            gru_hidden = self.dropout(gru_hidden)
            outputs[:, t - 1, :] = self.linear(gru_hidden)
        return outputs

    def generate_greedy(self, features, max_new_tokens):
        batch_size = features.size(0)
        lstm_hidden = features
        lstm_cell = features
        gru_hidden = features
        outputs = torch.zeros(
            (batch_size, max_new_tokens), dtype=torch.int, device=self.device
        )

        for t in range(1, max_new_tokens):
            outs = self.embedding(outputs[:, t - 1])
            lstm_hidden, lstm_cell = self.lstm_cell(outs, (lstm_hidden, lstm_cell))
            gru_hidden = self.gru_cell(lstm_cell, gru_hidden)
            outputs[:, t] = self.linear(gru_hidden).argmax(dim=-1)
        return outputs

    def generate_beam_search(self, features, beam_width, max_new_tokens):
        B = features.size(0)
        K = beam_width
        V = self.vocab_size
        H = self.hidden_size
        T = max_new_tokens

        lstm_hidden = features.unsqueeze(1).repeat(1, K, 1).view(-1, H)
        lstm_cell = features.unsqueeze(1).repeat(1, K, 1).view(-1, H)
        gru_hidden = features.unsqueeze(1).repeat(1, K, 1).view(-1, H)

        outputs = torch.zeros((B * K, T), dtype=torch.int, device=self.device)

        outs = self.embedding(outputs[:, 0])
        lstm_hidden, lstm_cell = self.lstm_cell(outs, (lstm_hidden, lstm_cell))
        gru_hidden = self.gru_cell(lstm_cell, gru_hidden)
        log_probs = torch.log_softmax(self.linear(gru_hidden), dim=-1)

        total_scores = log_probs.view(B, K, V)[:, 0, :]
        top_scores, token_indices = total_scores.topk(K, dim=-1)

        outputs[:, 1] = token_indices.view(-1)

        beam_scores = top_scores.view(-1, 1)

        for t in range(2, T):
            outs = self.embedding(outputs[:, t - 1])
            lstm_hidden, lstm_cell = self.lstm_cell(outs, (lstm_hidden, lstm_cell))
            gru_hidden = self.gru_cell(lstm_cell, gru_hidden)

            log_probs = torch.log_softmax(self.linear(gru_hidden), dim=-1)
            total_scores = beam_scores.view(B, K, 1) + log_probs.view(B, K, V)
            total_scores = total_scores.view(B, -1)

            top_scores, top_indices = total_scores.topk(K, dim=-1)
            beam_indices = top_indices // V
            token_indices = top_indices % V

            flat_old_indices = (
                torch.arange(B, device=self.device).unsqueeze(1) * K + beam_indices
            ).view(-1)
            outputs = outputs[flat_old_indices]
            outputs[:, t] = token_indices.view(-1)

            lstm_hidden = lstm_hidden[flat_old_indices]
            lstm_cell = lstm_cell[flat_old_indices]
            gru_hidden = gru_hidden[flat_old_indices]
            beam_scores = top_scores

        best_beam = beam_scores.argmax(dim=1)
        final_outputs = outputs.view(B, K, T)
        best_outputs = final_outputs[torch.arange(B), best_beam]
        return best_outputs


class HybridModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device):
        super(HybridModel, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(vocab_size, embedding_dim, hidden_size, device)

    def forward(self, images, captions):
        features = self.encoder(images)

        features = features.unsqueeze(1).repeat(1, captions.shape[1], 1)
        features = features.view(-1, features.shape[-1])
        captions = captions.view(-1, captions.shape[-1])

        return self.decoder(features, captions)

    def generate_greedy(self, images, max_new_tokens):
        features = self.encoder(images)
        return self.decoder.generate_greedy(features, max_new_tokens)

    def generate_beam_search(self, images, beam_width, max_new_tokens):
        features = self.encoder(images)
        return self.decoder.generate_beam_search(features, beam_width, max_new_tokens)


vocab_size = len(tokenizer.dictionary)
embedding_dim = 50
hidden_size = 2048
model = HybridModel(vocab_size, embedding_dim, hidden_size, device).to(device)


sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])


dummy_input = torch.randn(2, 3, 224, 224).to(device)
dummy_caption = torch.randint(0, 100, (2, 3, 37)).long().to(device)
dummy_caption.shape


o = model(dummy_input, dummy_caption)
o.shape


texts = model.generate_greedy(dummy_input, 10)
for text in texts:
    print(tokenizer.ids_to_text(text))


texts = model.generate_beam_search(dummy_input, 2, 10)
for text in texts:
    print(tokenizer.ids_to_text(text))


model


summary(model, input_data=(dummy_input, dummy_caption))


def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    num_batches = len(data_loader)

    for batch_idx, (images, caption) in enumerate(data_loader):
        images = images.to(device)
        caption = caption.to(device)
        optimizer.zero_grad()

        predictions = model(images, caption)
        caption = caption[:, :, 1:]
        predictions = predictions.view(-1, predictions.size(-1))
        caption = caption.reshape(-1)

        loss = criterion(predictions, caption)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / num_batches


def validation_epoch(model, data_loader, criterion, device, scheduler=None):
    model.eval()
    total_loss = 0
    num_batches = len(data_loader)

    with torch.no_grad():
        for batch_idx, (images, caption) in enumerate(data_loader):
            images = images.to(device)
            caption = caption.to(device)

            predictions = model(images, caption)
            caption = caption[:, :, 1:]
            predictions = predictions.view(-1, predictions.size(-1))
            caption = caption.reshape(-1)

            loss = criterion(predictions, caption)

            total_loss += loss.item()

    if scheduler != None:
        scheduler.step(total_loss / num_batches)

    return total_loss / num_batches


def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs,
    device,
    scheduler=None,
):
    hist = {
        "train_loss": [],
        "val_loss": [],
    }

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        hist["train_loss"].append(train_loss)
        print(f"Epoch [{epoch}] Average Training Loss: {train_loss:.4f}")

        val_loss = validation_epoch(
            model, val_loader, criterion, optimizer, device, scheduler
        )
        hist["val_loss"].append(val_loss)
        print(f"Epoch [{epoch}] Average Validation Loss: {val_loss:.4f}")

    return hist


batch_size = 32
epochs = 10
learning_rate = 0.001


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dictionary[special_tokens.Pad])


hist = train(model, train_loader, val_loader, criterion, optimizer, epochs, device)


def plot_history(history):
    epochs = len(history["train_loss"])
    plt.plot(range(1, epochs + 1), history["train_loss"])
    plt.plot(range(1, epochs + 1), history["val_loss"])
    plt.title("Loss of model")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Validation/Test"])
    plt.ylim(0, max(max(history["train_loss"]), max(history["val_loss"])))
    plt.show()


plot_history(hist)


def plot_generations(
    model, tokenizer, beam_widths, nsample, val_dataset, max_new_tokens
):
    transform = val_dataset.get_transform()
    val_dataset.set_transform(transform=None)

    fig = plt.figure(figsize=(8, 18))
    fig, axes = plt.subplots(5, 2, figsize=(14, 3.5 * 5))

    for i in range(nsample):
        image, captions = val_dataset[i]
        transformed_image = transform(image)

        lines = []
        ref_text = tokenizer.ids_to_text(captions[0], ignore_special_tokens=True)
        lines.append(f"True: {ref_text}")

        greedy = model.generate_greedy(
            transformed_image.unsqueeze(0).to(device), max_new_tokens
        )
        greedy_text = tokenizer.ids_to_text(greedy[0], ignore_special_tokens=True)
        lines.append(f"Greedy: {greedy_text}")

        for width in beam_widths:
            beam = model.generate_beam_search(
                transformed_image.unsqueeze(0).to(device), width, max_new_tokens
            )
            beam_text = tokenizer.ids_to_text(beam[0], ignore_special_tokens=True)
            lines.append(f"Beam (K={width}): {beam_text}")

        axes[i][0].imshow(image)
        axes[i][0].axis("off")

        axes[i][1].axis("off")
        full_text = "\n\n".join(lines)
        axes[i][1].text(
            0,
            1,
            full_text,
            fontsize=10,
            va="top",
            ha="left",
            wrap=True,
            transform=axes[i][1].transAxes,
        )

    plt.tight_layout()
    val_dataset.set_transform(transform)
    plt.show()


beam_widths = [3, 5, 10]
nsample = 5
max_new_tokens = tokenizer.context_length + 3
plot_generations(model, tokenizer, beam_widths, nsample, val_dataset, max_new_tokens)


def evaluate(model, tokenizer, data_loader, beam_widths, max_new_tokens):
    def accuracy_score(captions, generated):
        generated = generated.tolist()
        best_correct = 0
        best_total = 1e-9

        end_token = tokenizer.dictionary[special_tokens.End]
        len_generated = len(generated)
        if end_token in generated:
            len_generated = generated.index(end_token)

        for caption in captions:
            len_caption = len(caption)
            if end_token in caption:
                len_caption = caption.index(end_token)
            correct = [
                (generated[i] == caption[i])
                for i in range(1, min(len_caption, len_generated))
            ]

            if sum(correct) > best_correct:
                best_correct = sum(correct)
                best_total = len_caption
        return best_correct, best_total

    scores = defaultdict(lambda: defaultdict(float))
    count = 0
    smooth = SmoothingFunction().method1

    for batch_idx, (images, captions) in enumerate(data_loader):
        images = images.to(device)
        captions = captions.tolist()
        new_captions = []
        count += len(captions)

        greedy_ids = model.generate_greedy(images, max_new_tokens)
        greedy = [
            tokenizer.ids_to_text(t, ignore_special_tokens=True) for t in greedy_ids
        ]
        greedy = [tokenizer.tokenize(t) for t in greedy]

        for b in range(len(greedy)):
            caption = captions[b]
            best_correct, best_total = accuracy_score(caption, greedy_ids[b])
            scores["Greedy"]["Micro Accuracy"] += best_correct
            scores["Greedy"]["Total Tokens"] += best_total
            scores["Greedy"]["Macro Accuracy"] += best_correct / best_total

            caption = [
                tokenizer.ids_to_text(t, ignore_special_tokens=True) for t in caption
            ]
            caption = [tokenizer.tokenize(t) for t in caption]
            new_captions.append(caption)

            scores["Greedy"]["BLEU-1"] += sentence_bleu(
                caption, greedy[b], weights=(1, 0, 0, 0), smoothing_function=smooth
            )
            scores["Greedy"]["BLEU-2"] += sentence_bleu(
                caption, greedy[b], weights=(0.5, 0.5, 0, 0), smoothing_function=smooth
            )
            scores["Greedy"]["BLEU-3"] += sentence_bleu(
                caption,
                greedy[b],
                weights=(0.33, 0.33, 0.33, 0),
                smoothing_function=smooth,
            )
            scores["Greedy"]["BLEU-4"] += sentence_bleu(
                caption,
                greedy[b],
                weights=(0.25, 0.25, 0.25, 0.25),
                smoothing_function=smooth,
            )

        for width in beam_widths:
            beam_ids = model.generate_beam_search(images, width, max_new_tokens)
            beam = [
                tokenizer.ids_to_text(t, ignore_special_tokens=True) for t in beam_ids
            ]
            beam = [tokenizer.tokenize(t) for t in beam]

            for b in range(len(greedy)):
                best_correct, best_total = accuracy_score(captions[b], beam_ids[b])
                caption = new_captions[b]
                scores["Beam " + f"K = {width}"]["Micro Accuracy"] += best_correct
                scores["Beam " + f"K = {width}"]["Total Tokens"] += best_total
                scores["Beam " + f"K = {width}"]["Macro Accuracy"] += (
                    best_correct / best_total
                )
                scores["Beam " + f"K = {width}"]["BLEU-1"] += sentence_bleu(
                    caption, beam[b], weights=(1, 0, 0, 0), smoothing_function=smooth
                )
                scores["Beam " + f"K = {width}"]["BLEU-2"] += sentence_bleu(
                    caption,
                    beam[b],
                    weights=(0.5, 0.5, 0, 0),
                    smoothing_function=smooth,
                )
                scores["Beam " + f"K = {width}"]["BLEU-3"] += sentence_bleu(
                    caption,
                    beam[b],
                    weights=(0.33, 0.33, 0.33, 0),
                    smoothing_function=smooth,
                )
                scores["Beam " + f"K = {width}"]["BLEU-4"] += sentence_bleu(
                    caption,
                    beam[b],
                    weights=(0.25, 0.25, 0.25, 0.25),
                    smoothing_function=smooth,
                )

    final = {}
    for method, metrics in scores.items():
        total_tokens = metrics.pop("Total Tokens", 1)
        final[method] = {k: v / count for k, v in metrics.items()}
        final[method]["Micro Accuracy"] = (
            scores[method]["Micro Accuracy"] / total_tokens
        )

    return pd.DataFrame(final).T


evaluate(model, tokenizer, val_loader, beam_widths, max_new_tokens)


vocab_size = len(tokenizer.dictionary)
embedding_dim = 50
hidden_size = 2048
model_with_dropout = HybridModel(vocab_size, embedding_dim, hidden_size, device).to(
    device
)


optimizer = torch.optim.Adam(model_with_dropout.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dictionary[special_tokens.Pad])


epochs = 10
hist_with_dropout = train(
    model_with_dropout, train_loader, val_loader, criterion, optimizer, epochs, device
)
plot_history(hist_with_dropout)


beam_widths = [3, 5, 10]
nsample = 5
max_new_tokens = tokenizer.context_length + 3
plot_generations(
    model_with_dropout, tokenizer, beam_widths, nsample, val_dataset, max_new_tokens
)


evaluate(model_with_dropout, tokenizer, val_loader, beam_widths, max_new_tokens)


class DecoderNoTeacherForcing(Decoder):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device):
        super(DecoderNoTeacherForcing, self).__init__(
            vocab_size, embedding_dim, hidden_size, device
        )

    def forward(self, features, captions):
        batch_size = features.size(0)
        lstm_hidden = features
        lstm_cell = features
        gru_hidden = features
        outputs = torch.empty(
            (batch_size, captions.size(1) - 1, self.vocab_size), device=self.device
        )
        tokens = torch.zeros((batch_size, 1), dtype=torch.int, device=self.device)

        for t in range(1, captions.size(1)):
            embeddings = self.embedding(tokens)[:, 0, :]
            embeddings = self.dropout(embeddings)
            lstm_hidden, lstm_cell = self.lstm_cell(
                embeddings, (lstm_hidden, lstm_cell)
            )
            lstm_cell = self.dropout(lstm_cell)
            gru_hidden = self.gru_cell(lstm_cell, gru_hidden)
            gru_hidden = self.dropout(gru_hidden)
            outputs[:, t - 1, :] = self.linear(gru_hidden)
            tokens = outputs[:, t - 1, :].argmax(dim=-1).unsqueeze(1)
        return outputs


class HybridModelNoTeacherForcing(HybridModel):
    def __init__(self, vocab_size, embedding_dim, hidden_size, device):
        super(HybridModelNoTeacherForcing, self).__init__(
            vocab_size, embedding_dim, hidden_size, device
        )
        self.encoder = Encoder()
        self.decoder = DecoderNoTeacherForcing(
            vocab_size, embedding_dim, hidden_size, device
        )

    def forward(self, images, captions):
        features = self.encoder(images)

        outputs = self.decoder(features, captions[:, 0, :])
        outputs = outputs.unsqueeze(1).repeat(1, captions.shape[1], 1, 1)
        outputs = outputs.view(-1, outputs.shape[-2], outputs.shape[-1])

        return outputs

    def generate_greedy(self, images, max_new_tokens):
        features = self.encoder(images)
        return self.decoder.generate_greedy(features, max_new_tokens)

    def generate_beam_search(self, images, beam_width, max_new_tokens):
        features = self.encoder(images)
        return self.decoder.generate_beam_search(features, beam_width, max_new_tokens)


vocab_size = len(tokenizer.dictionary)
embedding_dim = 50
hidden_size = 2048
model_no_tf = HybridModelNoTeacherForcing(
    vocab_size, embedding_dim, hidden_size, device
).to(device)


optimizer = torch.optim.Adam(model_no_tf.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dictionary[special_tokens.Pad])


epochs = 10
hist_no_tf = train(
    model_no_tf, train_loader, val_loader, criterion, optimizer, epochs, device
)
plot_history(hist_no_tf)


beam_widths = [3, 5, 10]
nsample = 5
max_new_tokens = tokenizer.context_length + 3
plot_generations(
    model_no_tf, tokenizer, beam_widths, nsample, val_dataset, max_new_tokens
)


small_tokenizer = Tokenizer(train_captions, unknown_percent=0.5)
len_dictionary = len(small_tokenizer.dictionary)
len_unk = len(small_tokenizer.unknown_words)
total_tokens = small_tokenizer.total_tokens
total_unknown_tokens = small_tokenizer.total_unknown_tokens
print("Numer of unqiue tokens in captions:", len_dictionary + len_unk)
print("Numer of tokens in dictionary:", len_dictionary)
print("Numer of unknown words in captions:", len_unk)
print(
    "Percent of unknown tokens to all tokens:",
    len_unk * 100 / (len_dictionary + len_unk),
)
print(
    "Percent of unknown tokens to all tokens in captions:",
    total_unknown_tokens * 100 / total_tokens,
)
print("List of some unknown words in captions:", small_tokenizer.unknown_words[-10:])


small_train_dataset = Flickr8kDataset(
    os.path.join(path, "Images"), train_captions, small_tokenizer, transform=transform
)
small_val_dataset = Flickr8kDataset(
    os.path.join(path, "Images"), val_captions, small_tokenizer, transform=transform
)
small_test_dataset = Flickr8kDataset(
    os.path.join(path, "Images"), test_captions, small_tokenizer, transform=transform
)


small_train_loader = DataLoader(
    small_train_dataset, batch_size=batch_size, shuffle=True
)
small_val_loader = DataLoader(small_val_dataset, batch_size=batch_size, shuffle=False)


vocab_size = len(small_tokenizer.dictionary)
embedding_dim = 50
hidden_size = 2048
small_model = HybridModel(vocab_size, embedding_dim, hidden_size, device).to(device)


optimizer = torch.optim.Adam(small_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(
    ignore_index=small_tokenizer.dictionary[special_tokens.Pad]
)


epochs = 10
small_hist = train(
    small_model,
    small_train_loader,
    small_val_loader,
    criterion,
    optimizer,
    epochs,
    device,
)
plot_history(small_hist)


torch.save(model.state_dict(), path + "model")
torch.save(model_with_dropout.state_dict(), path + "model_with_dropout")
torch.save(small_model.state_dict(), path + "small_model")


small_max_new_tokens = small_tokenizer.context_length + 3
plot_generations(
    small_model,
    small_tokenizer,
    beam_widths,
    nsample,
    small_val_dataset,
    small_max_new_tokens,
)


evaluate(small_model, small_tokenizer, small_val_loader, beam_widths, max_new_tokens)


vocab_size = len(tokenizer.dictionary)
embedding_dim = 150
hidden_size = 2048
model1 = HybridModel(vocab_size, embedding_dim, hidden_size, device).to(device)


optimizer = torch.optim.Adam(model1.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dictionary[special_tokens.Pad])


hist1 = train(model1, train_loader, val_loader, criterion, optimizer, epochs, device)
torch.save(model1.state_dict(), path + "model150")
plot_history(hist1)


max_new_tokens = tokenizer.context_length + 3
plot_generations(model1, tokenizer, beam_widths, nsample, val_dataset, max_new_tokens)


evaluate(model1, tokenizer, val_loader, beam_widths, max_new_tokens)


vocab_size = len(tokenizer.dictionary)
embedding_dim = 300
hidden_size = 2048
model2 = HybridModel(vocab_size, embedding_dim, hidden_size, device).to(device)


optimizer = torch.optim.Adam(model2.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dictionary[special_tokens.Pad])


hist2 = train(model2, train_loader, val_loader, criterion, optimizer, epochs, device)
plot_history(hist2)


beam_widths = [3, 5, 10]
nsample = 5
max_new_tokens = tokenizer.context_length + 3
plot_generations(model2, tokenizer, beam_widths, nsample, val_dataset, max_new_tokens)


evaluate(model2, tokenizer, val_loader, beam_widths, max_new_tokens)


vocab_size = len(tokenizer.dictionary)
embedding_dim = 50
hidden_size = 2048
model40_epochs = HybridModel(vocab_size, embedding_dim, hidden_size, device).to(device)


optimizer = torch.optim.Adam(model40_epochs.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dictionary[special_tokens.Pad])


epochs = 40
hist1 = train(
    model40_epochs, train_loader, val_loader, criterion, optimizer, epochs, device
)
plot_history(hist1)


max_new_tokens = tokenizer.context_length + 3
plot_generations(
    model40_epochs, tokenizer, beam_widths, nsample, val_dataset, max_new_tokens
)


evaluate(model40_epochs, tokenizer, val_loader, beam_widths, max_new_tokens)


train_val_dataset = ConcatDataset([train_dataset, val_dataset])
train_val_loader = DataLoader(train_val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


vocab_size = len(tokenizer.dictionary)
embedding_dim = 50
hidden_size = 2048
final_model = HybridModel(vocab_size, embedding_dim, hidden_size, device).to(device)


optimizer = torch.optim.Adam(final_model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.dictionary[special_tokens.Pad])


epochs = 13
final_hist = train(
    final_model, train_val_loader, test_loader, criterion, optimizer, epochs, device
)
plot_history(final_hist)


beam_widths = [3, 5, 10]
nsample = 5
max_new_tokens = tokenizer.context_length + 3
plot_generations(
    final_model, tokenizer, beam_widths, nsample, test_dataset, max_new_tokens
)


evaluate(final_model, tokenizer, test_loader, beam_widths, max_new_tokens)
