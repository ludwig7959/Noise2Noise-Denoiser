import os

import numpy as np
import torch
from dotenv import load_dotenv
from pesq import pesq
from scipy import interpolate

from metrics import AudioMetrics2

load_dotenv()

SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 48000))
N_FFT = int(os.getenv('N_FFT', 3072))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 768))


def resample(original, old_rate, new_rate):
    if old_rate != new_rate:
        duration = original.shape[0] / old_rate
        time_old  = np.linspace(0, duration, original.shape[0])
        time_new  = np.linspace(0, duration, int(original.shape[0] * new_rate / old_rate))
        interpolator = interpolate.interp1d(time_old, original.T)
        new_audio = interpolator(time_new).T
        return new_audio
    else:
        return original


def wsdr_fn(x_, y_pred_, y_true_, eps=1e-8):
    # to time-domain waveform
    y_true_ = torch.squeeze(y_true_, 1)
    y_true = torch.istft(y_true_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=torch.hann_window(window_length=3072).to(y_true_.device))
    x_ = torch.squeeze(x_, 1)
    x = torch.istft(x_, n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=torch.hann_window(window_length=3072).to(x_.device))

    y_pred = y_pred_.flatten(1)
    y_true = y_true.flatten(1)
    x = x.flatten(1)

    def sdr_fn(true, pred, eps=1e-8):
        num = torch.sum(true * pred, dim=1)
        den = torch.norm(true, p=2, dim=1) * torch.norm(pred, p=2, dim=1)
        return -(num / (den + eps))

    # true and estimated noise
    z_true = x - y_true
    z_pred = x - y_pred

    a = torch.sum(y_true**2, dim=1) / (torch.sum(y_true**2, dim=1) + torch.sum(z_true**2, dim=1) + eps)
    wSDR = a * sdr_fn(y_true, y_pred) + (1 - a) * sdr_fn(z_true, z_pred)
    return torch.mean(wSDR)


wonky_samples = []


def get_metricson_loader(loader, net, use_net=True):
    net.eval()
    # Original test metrics
    scale_factor = 32768
    # metric_names = ["CSIG","CBAK","COVL","PESQ","SSNR","STOI","SNR "]
    metric_names = ["PESQ-WB","PESQ-NB","SNR","SSNR","STOI"]
    overall_metrics = [[] for i in range(5)]
    for i, data in enumerate(loader):
        if (i+1)%10==0:
            end_str = "\n"
        else:
            end_str = ","
        #print(i,end=end_str)
        if i in wonky_samples:
            print("Something's up with this sample. Passing...")
        else:
            noisy = data[0]
            clean = data[1]
            if use_net: # Forward of net returns the istft version
                x_est = net(noisy, is_istft=True)
                x_est_np = x_est.view(-1).detach().cpu().numpy()
            else:
                x_est_np = torch.istft(torch.squeeze(noisy, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=torch.hann_window(window_length=3072).to(noisy.device)).view(-1).detach().cpu().numpy()
            x_clean_np = torch.istft(torch.squeeze(clean, 1), n_fft=N_FFT, hop_length=HOP_LENGTH, normalized=True, window=torch.hann_window(window_length=3072).to(clean.device)).view(-1).detach().cpu().numpy()


            metrics = AudioMetrics2(x_clean_np, x_est_np, 48000)

            ref_wb = resample(x_clean_np, 48000, 16000)
            deg_wb = resample(x_est_np, 48000, 16000)
            pesq_wb = pesq(16000, ref_wb, deg_wb, 'wb')

            ref_nb = resample(x_clean_np, 48000, 8000)
            deg_nb = resample(x_est_np, 48000, 8000)
            pesq_nb = pesq(8000, ref_nb, deg_nb, 'nb')

            #print(new_scores)
            #print(metrics.PESQ, metrics.STOI)

            overall_metrics[0].append(pesq_wb)
            overall_metrics[1].append(pesq_nb)
            overall_metrics[2].append(metrics.SNR)
            overall_metrics[3].append(metrics.SSNR)
            overall_metrics[4].append(metrics.STOI)
    print()
    print("Sample metrics computed")
    results = {}
    for i in range(5):
        temp = {}
        temp["Mean"] =  np.mean(overall_metrics[i])
        temp["STD"]  =  np.std(overall_metrics[i])
        temp["Min"]  =  min(overall_metrics[i])
        temp["Max"]  =  max(overall_metrics[i])
        results[metric_names[i]] = temp
    print("Averages computed")
    if use_net:
        addon = "(cleaned by model)"
    else:
        addon = "(pre denoising)"
    print("Metrics on test data",addon)
    for i in range(5):
        print("{} : {:.3f}+/-{:.3f}".format(metric_names[i], np.mean(overall_metrics[i]), np.std(overall_metrics[i])))
    return results