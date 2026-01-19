import os
import torch, random
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')


def set_seed(seed=42):
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  os.environ['PYTHONHASHSEED'] = str(seed)


def cutmix_data(x, y, alpha=0.15):
  if alpha <= 0:
      return x, y, y, 1.0

  lam = np.random.beta(alpha, alpha)
  batch_size = x.size(0)
  index = torch.randperm(batch_size).to(x.device)

  seq_len = x.shape[2]
  cut_len = int(seq_len * (1 - lam))
  cut_start = random.randint(0, seq_len - cut_len)

  x[:, :, cut_start:cut_start + cut_len] = x[index, :, cut_start:cut_start + cut_len]
  y_a, y_b = y, y[index]

  return x, y_a, y_b, lam


def cutmix_criterion(criterion, pred, y_a, y_b, lam):
  return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class ConservativeAugmentationPipeline:
  def __init__(self, shift_range=20, noise_level=0.003, scale_range=0.03, prob=0.55):
      self.shift_range = shift_range
      self.noise_level = noise_level
      self.scale_range = scale_range
      self.prob = prob

  def time_shift(self, signal):
      c, L = signal.shape
      shift = random.randint(-self.shift_range, self.shift_range)
      if shift == 0:
          return signal
      elif shift > 0:
          return torch.cat([torch.zeros(c, shift, dtype=signal.dtype), signal[:, :-shift]], dim=1)
      else:
          return torch.cat([signal[:, -shift:], torch.zeros(c, -shift, dtype=signal.dtype)], dim=1)

  def add_noise(self, signal):
      gaussian_noise = torch.randn_like(signal) * self.noise_level
      return signal + gaussian_noise

  def scale_amplitude(self, signal):
      scale = 1.0 + random.uniform(-self.scale_range, self.scale_range)
      return signal * scale

  def __call__(self, signal):
      if random.random() >= self.prob:
          return signal
      original_signal = signal
      try:
          if random.random() < 0.7: signal = self.time_shift(signal)
          if random.random() < 0.3: signal = self.add_noise(signal)
          if random.random() < 0.3: signal = self.scale_amplitude(signal)
          return signal
      except Exception:
          return original_signal


class KeystrokeDataset(Dataset):
  def __init__(self, data_dir, signal_length=1000, is_train=False):
      self.samples, self.signal_length, self.is_train = [], signal_length, is_train
      self.augmentation = ConservativeAugmentationPipeline() if self.is_train else None

      for key in range(10):
          key_folder = os.path.join(data_dir, f'Key_{key}')
          if not os.path.isdir(key_folder): continue
          for fname in os.listdir(key_folder):
              if fname.lower().endswith('.csv'):
                  self.samples.append((os.path.join(key_folder, fname), key))

  def __len__(self):
      return len(self.samples)

  def __getitem__(self, idx):
      path, label = self.samples[idx]
      try:
          df = pd.read_csv(path)

          vbus_sig = df['VBUS'].values.astype(np.float32)
          ibus_sig = df['IBUS'].values.astype(np.float32)

          L = len(vbus_sig)

          if L < self.signal_length:
              vbus_sig = np.concatenate([vbus_sig, np.zeros(self.signal_length - L, dtype=np.float32)])
              ibus_sig = np.concatenate([ibus_sig, np.zeros(self.signal_length - L, dtype=np.float32)])
          else:
              vbus_sig = vbus_sig[:self.signal_length]
              ibus_sig = ibus_sig[:self.signal_length]

          tensor = torch.stack([
              torch.tensor(vbus_sig, dtype=torch.float32),
              torch.tensor(ibus_sig, dtype=torch.float32)
          ], dim=0)

          if self.is_train and self.augmentation:
              tensor = self.augmentation(tensor)

          return tensor, torch.tensor(label, dtype=torch.long)

      except Exception as e:
          return torch.zeros((2, self.signal_length)), torch.tensor(label, dtype=torch.long)


class SEBlock(nn.Module):
  def __init__(self, channels, reduction=16):
      super().__init__()
      self.squeeze = nn.AdaptiveAvgPool1d(1)
      self.excitation = nn.Sequential(
          nn.Linear(channels, channels // reduction, bias=False),
          nn.ReLU(inplace=True),
          nn.Linear(channels // reduction, channels, bias=False),
          nn.Sigmoid()
      )

  def forward(self, x):
      b, c, _ = x.size()
      y = self.squeeze(x).view(b, c)
      y = self.excitation(y).view(b, c, 1)
      return x * y.expand_as(x)


class SimpleResidualBlock(nn.Module):
  def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, use_se=True):
      super().__init__()
      self.use_shortcut = (stride != 1 or in_channels != out_channels)
      self.use_se = use_se

      self.conv_path = nn.Sequential(
          nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride,
                    padding=kernel_size // 2, bias=False),
          nn.BatchNorm1d(out_channels),
          nn.ReLU(inplace=True),
          nn.Dropout(0.1),
          nn.Conv1d(out_channels, out_channels, kernel_size,
                    padding=kernel_size // 2, bias=False),
          nn.BatchNorm1d(out_channels)
      )

      if self.use_shortcut:
          self.shortcut = nn.Sequential(
              nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
              nn.BatchNorm1d(out_channels)
          )

      if self.use_se:
          self.se = SEBlock(out_channels, reduction=16)

      self.activation = nn.ReLU(inplace=True)

  def forward(self, x):
      residual = self.shortcut(x) if self.use_shortcut else x
      out = self.conv_path(x)

      if self.use_se:
          out = self.se(out)

      return self.activation(out + residual)


class CutMixTestResNet(nn.Module):
  def __init__(self, num_classes=10, in_channels=2):
      super().__init__()

      channels = [72, 144, 288, 432]
      num_blocks = [2, 3, 3, 2]

      self.stem = nn.Sequential(
          nn.Conv1d(2, channels[0], kernel_size=15, stride=2, padding=7, bias=False),
          nn.BatchNorm1d(channels[0]),
          nn.ReLU(inplace=True),
          nn.Conv1d(channels[0], channels[0], kernel_size=7, stride=1, padding=3, bias=False),
          nn.BatchNorm1d(channels[0]),
          nn.ReLU(inplace=True),
          nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
      )

      layers = []
      current_channels = channels[0]
      for i in range(len(num_blocks)):
          layers.append(SimpleResidualBlock(
              current_channels, channels[i], kernel_size=7, stride=2, use_se=True
          ))
          for _ in range(num_blocks[i] - 1):
              layers.append(SimpleResidualBlock(
                  channels[i], channels[i], kernel_size=7, stride=1, use_se=True
              ))
          current_channels = channels[i]

      self.body = nn.Sequential(*layers)

      self.classifier = nn.Sequential(
          nn.AdaptiveAvgPool1d(1),
          nn.Flatten(),
          nn.Dropout(0.3),
          nn.Linear(channels[-1], 256),
          nn.ReLU(inplace=True),
          nn.Dropout(0.2),
          nn.Linear(256, num_classes)
      )

      param_count = sum(p.numel() for p in self.parameters() if p.requires_grad)

  def forward(self, x):
      x = self.stem(x)
      x = self.body(x)
      return self.classifier(x)


class CutMixTestTrainer:
  def __init__(self, data_dir):
      self.data_dir = data_dir
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
      self.cfg = {
          'lr': 0.0003, 'min_lr': 1e-7, 'patience': 100, 'epochs': 500, 'bs': 64,
          'weight_decay': 1.2e-3, 'warmup_epochs': 10, 'cutmix_alpha': 0.15, 'grad_clip': 1.0
      }

  def prepare_data(self):
      temp_ds = KeystrokeDataset(self.data_dir)
      train_indices, test_indices = train_test_split(
          range(len(temp_ds)), test_size=0.2,
          stratify=[y for _, y in temp_ds.samples], random_state=42
      )

      train_ds = KeystrokeDataset(self.data_dir, is_train=True)
      test_ds = KeystrokeDataset(self.data_dir, is_train=False)

      self.train_loader = DataLoader(Subset(train_ds, train_indices), self.cfg['bs'], shuffle=True, num_workers=0,
                                     pin_memory=True, drop_last=True)
      self.test_loader = DataLoader(Subset(test_ds, test_indices), self.cfg['bs'], shuffle=False, num_workers=0,
                                    pin_memory=True)

  def train(self):
      set_seed(42)
      model = CutMixTestResNet().to(self.device)

      optimizer = optim.AdamW(model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
      warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=self.cfg['warmup_epochs'])
      cosine_scheduler = CosineAnnealingLR(optimizer, T_max=self.cfg['epochs'] - self.cfg['warmup_epochs'],
                                           eta_min=self.cfg['min_lr'])
      scheduler = SequentialLR(optimizer, [warmup_scheduler, cosine_scheduler],
                               milestones=[self.cfg['warmup_epochs']])

      criterion = nn.CrossEntropyLoss()
      best_acc, no_improve = 0, 0

      for epoch in range(1, self.cfg['epochs'] + 1):
          model.train()
          train_loss = 0
          total_samples = 0

          for x, y in self.train_loader:
              x, y = x.to(self.device), y.to(self.device)

              if random.random() < 0.4 and epoch > 15:
                  x, y_a, y_b, lam = cutmix_data(x, y, self.cfg['cutmix_alpha'])
                  optimizer.zero_grad()
                  out = model(x)
                  loss = cutmix_criterion(criterion, out, y_a, y_b, lam)
              else:
                  optimizer.zero_grad()
                  out = model(x)
                  loss = criterion(out, y)

              loss.backward()
              torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg['grad_clip'])
              optimizer.step()

              train_loss += loss.item() * x.size(0)
              total_samples += x.size(0)

          model.eval()
          te_correct = 0
          te_total = 0
          with torch.no_grad():
              for x, y in self.test_loader:
                  x, y = x.to(self.device), y.to(self.device)
                  out = model(x)
                  te_correct += (out.argmax(1) == y).sum().item()
                  te_total += y.size(0)

          te_acc = 100 * te_correct / te_total
          scheduler.step()

          if epoch % 20 == 0 or epoch <= 10:
              print(
                  f"Epoch {epoch:03d} | Loss: {train_loss / total_samples:.4f} | Acc: {te_acc:.2f}%")

          if te_acc > best_acc:
              best_acc, no_improve = te_acc, 0
              torch.save(model.state_dict(), '../cutmix_best.pth')
              print(f"{best_acc:.2f}%")
          else:
              no_improve += 1
              if no_improve >= self.cfg['patience']:
                  break

      return best_acc


def main():
  data_dir = '/input/'
  trainer = CutMixTestTrainer(data_dir)
  trainer.prepare_data()
  trainer.train()


if __name__ == '__main__': main()
