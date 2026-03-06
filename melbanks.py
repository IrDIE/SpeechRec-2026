from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F
import torchaudio

import matplotlib.pyplot as plt
import numpy as np
import torch

def plot_spectrogram(spec, title="Mel spectrogram", sr=16000, hop_length=160, 
                     log=True, eps=1e-6, ax=None, figsize=(10, 4)):
    """
    Plot a mel spectrogram.

    Args:
        spec (torch.Tensor): Spectrogram of shape (..., n_mels, time). 
                             If batched, the first item is used.
        title (str): Title for the plot.
        sr (int): Sample rate of the audio (used for axis labeling).
        hop_length (int): Hop length in samples (used for time axis).
        log (bool): If True, apply 20*log10(spec + eps) before plotting.
        eps (float): Small constant to avoid log(0).
        ax (matplotlib.axes.Axes): Existing axes to plot on. If None, a new figure is created.
        figsize (tuple): Figure size if creating a new figure.
    """
    # Remove batch dimension if present
    if spec.dim() == 3:
        spec = spec[0]  # take first item in batch
    # Ensure it's on CPU and convert to numpy
    spec = spec.detach().cpu().numpy()
    
    # Convert to dB if requested
    if log:
        spec = 20 * np.log10(spec + eps)
    
    # Create time axis (in seconds)
    n_frames = spec.shape[1]
    time = np.arange(n_frames) * hop_length / sr
    
    # Mel frequency axis (just indices, you could also compute mel frequencies)
    mel_bins = np.arange(spec.shape[0])
    
    # Create plot
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    im = ax.imshow(spec, aspect='auto', origin='lower', 
                   extent=[time[0], time[-1], mel_bins[0], mel_bins[-1]],
                   cmap='viridis')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Mel bin')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label='dB' if log else 'Amplitude')
    

    plt.tight_layout()
    plt.savefig('assets/mel_spectrogram.png')

class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        # Do correct initialization of stft params below:
        # hop_length, n_mels, center, return_complex, onesided, normalize_stft, pad_mode, power
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power

        # Do correct initialization of mel fbanks params below:
        # f_min_hz, f_max_hz, norm_mel, mel_scale
        self.f_min_hz = f_min_hz
        self.f_max_hz = samplerate / 2
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale

        # finish parameters initialization
        self.mel_fbanks = self._init_melscale_fbanks()

    def _init_melscale_fbanks(self):
        # To access attributes, use self.<parameter_name>

        return F.melscale_fbanks(
            int(self.n_fft // 2 + 1),  # Nyquist frequency
            f_min=self.f_min_hz,
            f_max=self.f_max_hz,
            n_mels=self.n_mels,
            sample_rate=self.samplerate,
            norm=self.norm_mel,            # pass through
            mel_scale=self.mel_scale 
            # Turns a normal STFT into a mel frequency STFT with triangular filter banks
            # make a full and correct function call
        ) # torch.Size([201, 80])

    def spectrogram(self, x):
        return torch.stft(
            # make a full and correct function call
            x,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            center = self.center,
            pad_mode = self.pad_mode,
            win_length=self.window_length,
            window = self.window.to(x.device),
            normalized=self.normalize_stft,
            onesided=self.onesided,
            return_complex=self.return_complex,
        )

    def forward(self, x, do_plot = False):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        # Return log mel filterbanks matrix
        src_dims = x.dim()
        if src_dims == 3:
            x = x.squeeze(1)

        s = self.spectrogram(x).abs() # magnitude torch.Size([1, 201, 340])
        s = torch.pow(s, self.power) 

        mel_spec = torch.matmul(self.mel_fbanks.T.to(s.device), s)

        res = torch.log(mel_spec + 1e-6 )
        if do_plot:
            plot_spectrogram(s, title="Log spectrogram", sr=self.samplerate, hop_length=self.hop_length, log=True)
        # return dim back
        # if src_dims == 3:
        #     res = res.unsqueeze(1)
        return  res # torch.Size([1, 80, 340]) # power  torch.Size([256, 20, 101])


def main_mel():
    wav_path = 'assets/download.wav'
    signal, sr = torchaudio.load(wav_path) # torch.Size([1, 54272])
    melspec = torchaudio.transforms.MelSpectrogram(hop_length=160, n_mels=80)(signal) # torch.Size([1, 80, 340])
    logmelbanks = LogMelFilterBanks()(signal, do_plot = True)

    assert torch.log(melspec + 1e-6).shape == logmelbanks.shape
    assert torch.allclose(torch.log(melspec + 1e-6), logmelbanks)

    print(f"ALL PASSED")

if __name__ == "__main__":
    main_mel()