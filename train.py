import gc
import os
from pathlib import Path
import torch
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SoundDataset
from loss import wsdr_fn, get_metricson_loader
from models import DCUnet20


def train(net, train_loader, loss_fn, optimizer, scheduler, epochs):

    train_losses = []

    for e in tqdm(range(epochs)):

        # first evaluating for comparison

        train_loss = train_epoch(net, train_loader, loss_fn, optimizer)
        scheduler.step()
        print(f'Loss: {train_loss}')

        train_losses.append(train_loss)

        torch.save(net.state_dict(), 'Weights/dc20_model_'+str(e+1)+'.pth')
        torch.save(optimizer.state_dict(), 'Weights/dc20_opt_'+str(e+1)+'.pth')

        print("Models saved")

        # clear cache
        torch.cuda.empty_cache()
        gc.collect()

    return train_losses


def train_epoch(net, train_loader, loss_fn, optimizer):
    net.train()
    train_ep_loss = 0.
    counter = 0
    for noisy_chunks, clean_chunks in train_loader:
        for i in range(len(noisy_chunks)):
            noisy_chunk, clean_chunk = noisy_chunks[i].to(DEVICE), clean_chunks[i].to(DEVICE)

            # zero  gradients
            net.zero_grad()

            # get the output from the model
            pred = net(noisy_chunk)

            # calculate loss
            loss = loss_fn(noisy_chunk, pred, clean_chunk)
            loss.backward()
            optimizer.step()

            train_ep_loss += loss.item()
            counter += 1

    train_ep_loss /= counter

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()
    return train_ep_loss


def test_epoch(net, test_loader, loss_fn, use_net=True):
    net.eval()
    test_ep_loss = 0.
    counter = 0.

    testmet = get_metricson_loader(test_loader, net, use_net)

    # clear cache
    gc.collect()
    torch.cuda.empty_cache()

    return test_ep_loss, testmet


def collate_fn(batch):
    noisy_chunks_batch, clean_chunks_batch = zip(*batch)

    noisy_chunks_batch = [torch.stack(chunk_list) for chunk_list in zip(*noisy_chunks_batch)]
    clean_chunks_batch = [torch.stack(chunk_list) for chunk_list in zip(*clean_chunks_batch)]

    return noisy_chunks_batch, clean_chunks_batch


load_dotenv()

SAMPLE_RATE = int(os.getenv('SAMPLE_RATE', 48000))
N_FFT = int(os.getenv('N_FFT', 3072))
HOP_LENGTH = int(os.getenv('HOP_LENGTH', 768))

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

noise_audio_path = Path(input('노이즈가 포함된 경로를 입력하세요: '))
os.makedirs("Weights",exist_ok=True)

train_files = sorted(list(noise_audio_path.rglob('*.wav')))

train_dataset = SoundDataset(train_files, train_files, N_FFT, HOP_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

gc.collect()
torch.cuda.empty_cache()

dcunet20 = DCUnet20(N_FFT, HOP_LENGTH).to(DEVICE)
optimizer = torch.optim.Adam(dcunet20.parameters())

# weights = torch.load("Weights/dc20_model_2.pth", map_location=torch.device(DEVICE))
# optim = torch.load("Weights/dc20_opt_2.pth", map_location=torch.device(DEVICE))
# dcunet20.load_state_dict(weights)
# optimizer.load_state_dict(optim)

loss_fn = wsdr_fn
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

train_losses = train(dcunet20, train_loader, loss_fn, optimizer, scheduler, 50)
