import os

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.utils.data as torch_data
import torchaudio
import torchaudio.transforms as T


class Dataset(torch_data.Dataset):
    def __init__(self, datadir: str, feats: nn.Module):
        self._pathes = []
        self._speakers = []
        self._feats = feats
        speakers = os.listdir(datadir)
        for idx, speaker in enumerate(speakers):
            sp_dir = os.path.join(datadir, speaker)
            for item in os.listdir(sp_dir):
                self._pathes.append(os.path.join(sp_dir, item))
                self._speakers.append(idx)

        # self.time_masking = T.TimeMasking(time_mask_param=30)
        # self.freq_masking = T.FrequencyMasking(freq_mask_param=15)

    def plot_spectrogram(self, index):
        waveform, sample_rate = torchaudio.load(self._pathes[index])
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_mels=128,
            n_fft=1024,
            hop_length=512
        )
        mel_spec = mel_spectrogram(waveform)
        mel_spec_db = T.AmplitudeToDB()(mel_spec)

        plt.figure(figsize=(10, 4))
        plt.imshow(mel_spec_db[0].numpy(), cmap='viridis', aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram')
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show()

    def __getitem__(self, index):
        waveform, sample_rate = torchaudio.load(self._pathes[index])
        assert sample_rate == 16000
        assert waveform.shape[0] == 1

        # waveform = self.time_masking(waveform)
        # waveform = self.freq_masking(waveform)

        feats = self._feats(waveform)[0]
        prev, uttid = os.path.split(self._pathes[index])
        speaker_id = os.path.split(prev)[1]
        return (feats, self._speakers[index], os.path.join(speaker_id, uttid))

    def __len__(self) -> int:
        return len(self._pathes)

    def speakers(self) -> int:
        return self._speakers[-1] + 1


def collate_fn(batch) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    max_length = max(item[0].shape[1] for item in batch)
    X = torch.zeros((len(batch), batch[0][0].shape[0], max_length))
    for idx, item in enumerate(batch):
        X[idx, :, :item[0].shape[1]] = item[0]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long).reshape(len(batch), 1)
    pathes = [item[2] for item in batch]
    return (X, targets, pathes)
