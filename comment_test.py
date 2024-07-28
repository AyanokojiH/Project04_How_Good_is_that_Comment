import torch
import torch.nn as nn
import warnings
import pandas as pd
import numpy as np
from gensim.models.word2vec import Word2Vec
from torch.utils.data import DataLoader, Dataset, random_split
import time

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TextDataLoader:
    def __init__(self, data_path):
        self.train_data = pd.read_csv(data_path, encoding='utf-8')

    def data_iterator(self):
        texts = self.train_data['Content'].values[:]
        labels = self.train_data['Score'].values[:]
        for x, y in zip(texts, labels):
            yield x, y

    def preprocess(self):
        x = self.train_data['Content'].values[:]
        y = self.train_data['Score'].values[:]
        return x, y


class Word2VecModel:
    def __init__(self, vector_size=100, min_count=3):
        self.model = Word2Vec(vector_size=vector_size, min_count=min_count)

    def train_model(self, texts):
        self.model.build_vocab(texts)
        self.model.train(texts, total_examples=self.model.corpus_count, epochs=20)

    def average_vector(self, text):
        vec = np.zeros(100).reshape((1, 100))
        for word in text:
            try:
                vec += self.model.wv[word].reshape((1, 100))
            except KeyError:
                continue
        return vec

    def save_model(self, path):
        self.model.save(path)


class CustomTextDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_batch(batch):
    global label_name
    label_list, text_list = [], []

    for text, label in batch:
        label_list.append(label_name.index(label))
        processed_text = torch.tensor(average_vec(text), dtype=torch.float32)
        text_list.append(processed_text)

    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = torch.cat(text_list)

    return text_list.to(device), label_list.to(device)


class TextClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(TextClassificationModel, self).__init__()
        self.fc = nn.Linear(100, num_classes)

    def forward(self, text):
        return self.fc(text)


class Trainer:
    def __init__(self, model, dataloader, criterion, optimizer, scheduler):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler

    def train(self):
        self.model.train()
        total_acc, train_loss, total_count = 0, 0, 0
        log_interval = 50
        start_time = time.time()

        for idx, (text, label) in enumerate(self.dataloader):
            predicted_label = self.model(text)
            self.optimizer.zero_grad()
            loss = self.criterion(predicted_label, label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
            self.optimizer.step()

            total_acc += (predicted_label.argmax(1) == label).sum().item()
            train_loss += loss.item()
            total_count += label.size(0)

            if idx % log_interval == 0 and idx > 0:
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:1d} | {idx:4d}/{len(self.dataloader):4d} batches '
                      f'| train_acc {total_acc / total_count:4.3f} train_loss {train_loss / total_count:4.5f}')
                total_acc, train_loss, total_count = 0, 0, 0
                start_time = time.time()

    def evaluate(self, dataloader):
        self.model.eval()
        total_acc, train_loss, total_count = 0, 0, 0

        with torch.no_grad():
            for idx, (text, label) in enumerate(dataloader):
                predicted_label = self.model(text)
                loss = self.criterion(predicted_label, label)
                total_acc += (predicted_label.argmax(1) == label).sum().item()
                train_loss += loss.item()
                total_count += label.size(0)

        return total_acc / total_count, train_loss / total_count


def main():
    global label_name, total_accu
    total_accu = None
    data_loader = TextDataLoader('data.csv')
    x, y = data_loader.preprocess()

    w2v_model = Word2VecModel()
    w2v_model.train_model(x)
    global average_vec
    average_vec = w2v_model.average_vector
    x_vec = np.concatenate([average_vec(z) for z in x])
    w2v_model.save_model('w2v_model.pkl')

    train_iter = list(data_loader.data_iterator())
    label_name = list(set(y))

    train_size = int(0.8 * len(train_iter))
    valid_size = len(train_iter) - train_size
    split_train_, split_valid_ = random_split(train_iter, [train_size, valid_size])

    train_dataset = CustomTextDataset(split_train_)
    valid_dataset = CustomTextDataset(split_valid_)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
    valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)

    num_classes = len(label_name)
    model = TextClassificationModel(num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=5)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    trainer = Trainer(model, train_dataloader, criterion, optimizer, scheduler)

    for epoch in range(1, 11):
        epoch_start_time = time.time()
        trainer.train()
        val_acc, val_loss = trainer.evaluate(valid_dataloader)

        lr = optimizer.state_dict()['param_groups'][0]['lr']

        if total_accu is not None and total_accu > val_acc:
            scheduler.step()
        else:
            total_accu = val_acc

        print(f'-' * 69)
        print(f'| epoch {epoch:1d} | time: {time.time() - epoch_start_time:4.2f}s | '
              f'valid_acc {val_acc:4.3f} valid_loss {val_loss:4.3f} | lr {lr:4.6f}')
        print(f'-' * 69)

    def predict(text):
        with torch.no_grad():
            text = torch.tensor(average_vec(text), dtype=torch.float32).to(device)
            output = model(text)
            return output.argmax(1).item()

    while True:
        ex_text_str = input()
        model.to("cpu")
        print(f"预测这个评论的得分是：{label_name[predict(ex_text_str)]}")


if __name__ == "__main__":
    main()
