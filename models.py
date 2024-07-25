import torch
from torch import nn
from layers import ComplexConv2D, ComplexBatchNorm2D, ComplexConvTranspose2D


class Encoder(nn.Module):

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45, padding=(0, 0)):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

        self.cconv = ComplexConv2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                   kernel_size=self.filter_size, stride=self.stride_size, padding=self.padding)

        self.cbn = ComplexBatchNorm2D(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        conved = self.cconv(x)
        normed = self.cbn(conved)

        return torch.complex(self.leaky_relu(normed.real), self.leaky_relu(normed.imag))


class Decoder(nn.Module):

    def __init__(self, filter_size=(7, 5), stride_size=(2, 2), in_channels=1, out_channels=45,
                 output_padding=(0, 0), padding=(0, 0), last_layer=False):
        super().__init__()

        self.filter_size = filter_size
        self.stride_size = stride_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output_padding = output_padding
        self.padding = padding

        self.last_layer = last_layer

        self.cconvt = ComplexConvTranspose2D(in_channels=self.in_channels, out_channels=self.out_channels,
                                             kernel_size=self.filter_size, stride=self.stride_size,
                                             output_padding=self.output_padding, padding=self.padding)

        self.cbn = ComplexBatchNorm2D(num_features=self.out_channels)

        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):

        conved = self.cconvt(x)

        if not self.last_layer:
            normed = self.cbn(conved)
            output = torch.complex(self.leaky_relu(normed.real), self.leaky_relu(normed.imag))
        else:
            m_phase = conved / (torch.abs(conved) + 1e-8)
            m_mag = torch.tanh(torch.abs(conved))
            output = m_phase * m_mag

        return output


class DCUnet20(nn.Module):

    def __init__(self, n_fft=64, hop_length=16):
        super().__init__()

        # for istft
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.set_size(model_complexity=32, input_channels=1, model_depth=20)
        self.encoders = []
        self.model_length = 20 // 2

        for i in range(self.model_length):
            module = Encoder(in_channels=self.enc_channels[i], out_channels=self.enc_channels[i + 1],
                             filter_size=self.enc_kernel_sizes[i], stride_size=self.enc_strides[i],
                             padding=self.enc_paddings[i])
            self.add_module("encoder{}".format(i), module)
            self.encoders.append(module)

        self.decoders = []

        for i in range(self.model_length):
            if i != self.model_length - 1:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i],
                                 out_channels=self.dec_channels[i + 1],
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i],
                                 padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i])
            else:
                module = Decoder(in_channels=self.dec_channels[i] + self.enc_channels[self.model_length - i],
                                 out_channels=self.dec_channels[i + 1],
                                 filter_size=self.dec_kernel_sizes[i], stride_size=self.dec_strides[i],
                                 padding=self.dec_paddings[i],
                                 output_padding=self.dec_output_padding[i], last_layer=True)
            self.add_module("decoder{}".format(i), module)
            self.decoders.append(module)

    def forward(self, x, is_istft=True):
        # print('x : ', x.shape)
        orig_x_real = x.real
        orig_x_imag = x.imag
        xs_real = []
        xs_imag = []
        for i, encoder in enumerate(self.encoders):
            xs_real.append(x.real)
            xs_imag.append(x.imag)
            x = encoder(x)
            # print('Encoder : ', x.shape)

        p = x
        for i, decoder in enumerate(self.decoders):
            p = decoder(p)
            if i == self.model_length - 1:
                break
            # print('Decoder : ', p.shape)
            connected_real = xs_real[self.model_length - 1 - i]
            connected_imag = xs_imag[self.model_length - 1 - i]

            p = torch.cat([p, torch.complex(connected_real, connected_imag)], dim=1)

        # u9 - the mask

        mask_real = p.real
        mask_imag = p.imag

        # print('mask : ', mask.shape)

        output_real = mask_real * orig_x_real - mask_imag * orig_x_imag
        output_real = torch.squeeze(output_real, 1)
        output_imag = mask_real * orig_x_imag + mask_imag * orig_x_real
        output_imag = torch.squeeze(output_imag, 1)

        output = torch.complex(output_real, output_imag)

        if is_istft:
            output = torch.istft(output, n_fft=self.n_fft, hop_length=self.hop_length, normalized=True, window=torch.hann_window(window_length=3072).to(output.device))

        return output

    def set_size(self, model_complexity, model_depth=20, input_channels=1):

        if model_depth == 20:
            self.enc_channels = [input_channels,
                                 model_complexity,
                                 model_complexity,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 128]

            self.enc_kernel_sizes = [(7, 1),
                                     (1, 7),
                                     (6, 4),
                                     (7, 5),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3),
                                     (5, 3)]

            self.enc_strides = [(1, 1),
                                (1, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1),
                                (2, 2),
                                (2, 1)]

            self.enc_paddings = [(3, 0),
                                 (0, 3),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0)]

            self.dec_channels = [0,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity * 2,
                                 model_complexity,
                                 model_complexity,
                                 1]

            self.dec_kernel_sizes = [(6, 3),
                                     (6, 3),
                                     (6, 3),
                                     (6, 4),
                                     (6, 3),
                                     (6, 4),
                                     (8, 5),
                                     (7, 5),
                                     (1, 7),
                                     (7, 1)]

            self.dec_strides = [(2, 1),  #
                                (2, 2),  #
                                (2, 1),  #
                                (2, 2),  #
                                (2, 1),  #
                                (2, 2),  #
                                (2, 1),  #
                                (2, 2),  #
                                (1, 1),
                                (1, 1)]

            self.dec_paddings = [(0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 0),
                                 (0, 3),
                                 (3, 0)]

            self.dec_output_padding = [(0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0),
                                       (0, 0)]
        else:
            raise ValueError("Unknown model depth : {}".format(model_depth))
