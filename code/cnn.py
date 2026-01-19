import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import random


def set_seed(seed=42):
   torch.manual_seed(seed)
   torch.cuda.manual_seed(seed)
   torch.cuda.manual_seed_all(seed)
   np.random.seed(seed)
   random.seed(seed)
   torch.backends.cudnn.deterministic = True
   torch.backends.cudnn.benchmark = False
   os.environ['PYTHONHASHSEED'] = str(seed)


class KeystrokeDataset(Dataset):
   def __init__(self, data_dir, signal_length=1000):
       self.samples = []
       self.signal_length = signal_length
       subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
       subdirs.sort()
       self.label_to_idx = {label: idx for idx, label in enumerate(subdirs)}
       self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

       for label_name in subdirs:
           folder_path = os.path.join(data_dir, label_name)
           if not os.path.isdir(folder_path):
               continue

           label_idx = self.label_to_idx[label_name]
           csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]


           for file_name in csv_files:
               file_path = os.path.join(folder_path, file_name)
               self.samples.append((file_path, label_idx))


   def __len__(self):
       return len(self.samples)

   def __getitem__(self, idx):
       file_path, label = self.samples[idx]

       try:
           df = pd.read_csv(file_path)

           required_cols = ['VBUS', 'IBUS']
           for col in required_cols:
               if col not in df.columns:
                   return torch.zeros((2, self.signal_length), dtype=torch.float32), torch.tensor(label,
                                                                                                  dtype=torch.long)

           vbus_signal = df['VBUS'].values
           ibus_signal = df['IBUS'].values

           vbus_signal = self._pad_or_truncate(vbus_signal, self.signal_length)
           ibus_signal = self._pad_or_truncate(ibus_signal, self.signal_length)

           signal_tensor = np.stack([vbus_signal, ibus_signal], axis=0)
           signal_tensor = torch.tensor(signal_tensor, dtype=torch.float32)

           if signal_tensor.shape != (2, self.signal_length):
               return torch.zeros((2, self.signal_length), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

           return signal_tensor, torch.tensor(label, dtype=torch.long)

       except Exception as e:
           return torch.zeros((2, self.signal_length), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

   def _pad_or_truncate(self, array, length):
       if len(array) < length:
           padding = np.zeros(length - len(array))
           return np.concatenate([array, padding])
       else:
           return array[:length]


class OptimalThreeLayerCNN(nn.Module):
   def __init__(self, num_classes=10, in_channels=2):
       super(OptimalThreeLayerCNN, self).__init__()

       self.kernel_sizes = [64, 96, 48]
       self.channels = [48, 192, 320]
       self.dropout = 0.8
       self.activation = nn.GELU()

       self.conv1 = nn.Conv1d(in_channels, self.channels[0], self.kernel_sizes[0], padding='same')
       self.bn1 = nn.BatchNorm1d(self.channels[0])
       self.pool1 = nn.MaxPool1d(2)
       self.dropout1 = nn.Dropout(self.dropout * 0.3)

       self.conv2 = nn.Conv1d(self.channels[0], self.channels[1], self.kernel_sizes[1], padding='same')
       self.bn2 = nn.BatchNorm1d(self.channels[1])
       self.pool2 = nn.MaxPool1d(2)
       self.dropout2 = nn.Dropout(self.dropout * 0.4)

       self.conv3 = nn.Conv1d(self.channels[1], self.channels[2], self.kernel_sizes[2], padding='same')
       self.bn3 = nn.BatchNorm1d(self.channels[2])
       self.pool3 = nn.MaxPool1d(2)
       self.dropout3 = nn.Dropout(self.dropout * 0.5)

       self.classifier = nn.Sequential(
           nn.AdaptiveAvgPool1d(1),
           nn.Flatten(),
           nn.Dropout(self.dropout),
           nn.Linear(self.channels[2], num_classes)
       )

       self.param_count = sum(p.numel() for p in self.parameters())

   def forward(self, x):
       x = self.conv1(x)
       x = self.bn1(x)
       x = self.activation(x)
       x = self.pool1(x)
       x = self.dropout1(x)

       x = self.conv2(x)
       x = self.bn2(x)
       x = self.activation(x)
       x = self.pool2(x)
       x = self.dropout2(x)

       x = self.conv3(x)
       x = self.bn3(x)
       x = self.activation(x)
       x = self.pool3(x)
       x = self.dropout3(x)

       x = self.classifier(x)
       return x


class Trainer:
   def __init__(self, data_dir):
       self.data_dir = data_dir
       self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

       self.config = {
           'learning_rate': 0.02,
           'patience': 50,
           'max_epochs': 200,
           'batch_size': 32
       }

   def prepare_data(self):

       full_dataset = KeystrokeDataset(self.data_dir)

       self.label_to_idx = full_dataset.label_to_idx
       self.idx_to_label = full_dataset.idx_to_label

       train_indices, test_indices = train_test_split(
           list(range(len(full_dataset))),
           test_size=0.2,
           random_state=42,
           stratify=[label for _, label in full_dataset.samples]
       )

       train_dataset = KeystrokeDataset(self.data_dir)
       train_dataset.samples = [full_dataset.samples[i] for i in train_indices]
       train_dataset.label_to_idx = full_dataset.label_to_idx
       train_dataset.idx_to_label = full_dataset.idx_to_label

       test_dataset = KeystrokeDataset(self.data_dir)
       test_dataset.samples = [full_dataset.samples[i] for i in test_indices]
       test_dataset.label_to_idx = full_dataset.label_to_idx
       test_dataset.idx_to_label = full_dataset.idx_to_label

       self.train_loader = DataLoader(
           train_dataset,
           batch_size=self.config['batch_size'],
           shuffle=True,
           num_workers=0,
           pin_memory=True if torch.cuda.is_available() else False
       )
       self.test_loader = DataLoader(
           test_dataset,
           batch_size=self.config['batch_size'],
           shuffle=False,
           num_workers=0,
           pin_memory=True if torch.cuda.is_available() else False
       )

   def train_model(self):
       set_seed(42)
       model = OptimalThreeLayerCNN(num_classes=10, in_channels=2).to(self.device)

       optimizer = optim.Adam(model.parameters(), lr=self.config['learning_rate'])
       criterion = nn.CrossEntropyLoss()

       best_accuracy = 0.0
       best_epoch = 0
       no_improve = 0
       train_history = []

       for epoch in range(self.config['max_epochs']):
           model.train()
           train_loss = 0.0
           train_correct = 0
           train_total = 0

           for batch_idx, (signals, labels) in enumerate(self.train_loader):
               if signals.shape[1:] != (2, 1000):
                   continue

               signals = signals.to(self.device)
               labels = labels.to(self.device)

               optimizer.zero_grad()
               outputs = model(signals)
               loss = criterion(outputs, labels)
               loss.backward()
               optimizer.step()

               train_loss += loss.item()
               _, predicted = torch.max(outputs.data, 1)
               train_total += labels.size(0)
               train_correct += (predicted == labels).sum().item()

           train_accuracy = 100 * train_correct / train_total

           model.eval()
           test_correct = 0
           test_total = 0
           test_loss = 0.0

           with torch.no_grad():
               for signals, labels in self.test_loader:
                   signals = signals.to(self.device)
                   labels = labels.to(self.device)
                   outputs = model(signals)
                   loss = criterion(outputs, labels)
                   test_loss += loss.item()

                   _, predicted = torch.max(outputs.data, 1)
                   test_total += labels.size(0)
                   test_correct += (predicted == labels).sum().item()

           test_accuracy = 100 * test_correct / test_total
           test_loss = test_loss / len(self.test_loader)

           train_history.append({
               'epoch': epoch + 1,
               'train_acc': train_accuracy,
               'test_acc': test_accuracy,
               'train_loss': train_loss / len(self.train_loader),
               'test_loss': test_loss
           })

           if test_accuracy > best_accuracy:
               best_accuracy = test_accuracy
               best_epoch = epoch + 1
               no_improve = 0
               torch.save(model.state_dict(), 'best_baseline.pth')
           else:
               no_improve += 1

           if (epoch + 1) % 10 == 0:
               print(
                   f"Epoch {epoch + 1:3d}: Train {train_accuracy:.2f}% | Test {test_accuracy:.2f}% | Best {best_accuracy:.2f}%")

           if no_improve >= self.config['patience']:
               break

       return model, best_accuracy, train_history

   def evaluate_model(self, model_path='best_baseline.pth'):
       model = OptimalThreeLayerCNN(num_classes=10, in_channels=2).to(self.device)
       model.load_state_dict(torch.load(model_path))
       model.eval()

       correct = 0
       total = 0
       class_correct = list(0. for i in range(10))
       class_total = list(0. for i in range(10))

       with torch.no_grad():
           for signals, labels in self.test_loader:
               signals = signals.to(self.device)
               labels = labels.to(self.device)
               outputs = model(signals)
               _, predicted = torch.max(outputs, 1)
               total += labels.size(0)
               correct += (predicted == labels).sum().item()

               c = (predicted == labels).squeeze()
               for i in range(labels.size(0)):
                   label = labels[i]
                   class_correct[label] += c[i].item()
                   class_total[label] += 1

       overall_accuracy = 100 * correct / total

       for i in range(10):
           if class_total[i] > 0:
               accuracy = 100 * class_correct[i] / class_total[i]
               app_name = self.idx_to_label.get(i, f"Unknown_{i}")
               print(f"   {app_name}: {accuracy:.2f}% ({class_correct[i]:.0f}/{class_total[i]:.0f})")

       return overall_accuracy

   def run_experiment(self):
       self.prepare_data()
       model, best_accuracy, history = self.train_model()
       final_accuracy = self.evaluate_model()
       return best_accuracy


def main():
   DATA_DIR = '/input/'

   trainer = Trainer(DATA_DIR)
   best_acc = trainer.run_experiment()

   print(f"\n final result: {best_acc:.2f}%")


if __name__ == '__main__':
   main()
