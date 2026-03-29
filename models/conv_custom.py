import torch
import torch.nn as nn
import torch.nn.functional as F

class OmniAttention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        super(OmniAttention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)
        self.relu = nn.ReLU(inplace=True)

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)
        self.func_channel = self.get_channel_attention

        if in_planes == groups and in_planes == out_planes:  # depth-wise convolution
            self.func_filter = self.skip
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)
            self.func_filter = self.get_filter_attention

        if kernel_size == 1:  # point-wise convolution
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)
            self.func_spatial = self.get_spatial_attention

        if kernel_num == 1:
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)
            self.func_kernel = self.get_kernel_attention

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        self.temperature = temperature

    @staticmethod
    def skip(_):
        return 1.0

    def get_channel_attention(self, x):
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        # if f_att:
        #     return self.func_channel(x), self.func_filter(x)
        return self.func_channel(x) #, self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

class FRBlock(nn.Module):
    """
    Frequency-band Recalibration Block.

    Decomposes the input feature map into hierarchical frequency bands using
    FFT-based low-pass masks, then applies learned spatial attention weights
    to each band independently before summing them back.

    Band decomposition with k_list = [k0, k1, ...] (ascending):
        - high_0  = x       - LPF(x, k0)          ← highest frequency residual
        - high_1  = LPF(x, k0) - LPF(x, k1)       ← mid-high band
        - ...
        - low_last = LPF(x, k_last)                ← lowest frequency residual

    If `global_selection=True`, a channel-wise complex attention is applied to
    the full FFT spectrum before band splitting, gating real and imaginary parts
    independently.
    """
    def __init__(self,
                 in_channels,
                 k_list=[2],
                 lowfreq_att=True,       # if True, also apply learned attention to the residual low-freq band
                 fs_feat='feat',
                 lp_type='freq',         # low-pass type: 'freq' (FFT mask) | 'avgpool' | 'laplacian'
                 act='sigmoid',          # activation for spatial attention weights: 'sigmoid' | 'softmax'
                 spatial='conv',
                 spatial_group=1,        # depthwise group count for attention convolutions
                 spatial_kernel=3,
                 init='zero',            # 'zero' init makes FRBlock an identity at the start of training
                 global_selection=False, # if True, apply complex-domain global attention before band split
                 ):
        super().__init__()

        self.k_list       = k_list
        self.lp_list      = nn.ModuleList()
        self.freq_weight_conv_list = nn.ModuleList()
        self.fs_feat      = fs_feat
        self.lp_type      = lp_type
        self.in_channels  = in_channels
        self.lowfreq_att  = lowfreq_att

        # Cap spatial_group to in_channels if an oversized value is passed
        if spatial_group > 64:
            spatial_group = in_channels
        self.spatial_group = spatial_group

        # Build one attention conv per frequency band:
        # len(k_list) high-freq bands + 1 extra if lowfreq_att is enabled
        if spatial == 'conv':
            _n = len(k_list) + (1 if lowfreq_att else 0)
            for _ in range(_n):
                freq_weight_conv = nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=self.spatial_group,
                    stride=1,
                    kernel_size=spatial_kernel,
                    groups=self.spatial_group,       # depthwise — each group attends to its own channels
                    padding=spatial_kernel // 2,
                    bias=True,
                )
                if init == 'zero':
                    # Zero init → all attention weights start at neutral (sigmoid(0)*2 = 1),
                    # so FRBlock is an identity transformation at the beginning of training
                    freq_weight_conv.weight.data.zero_()
                    freq_weight_conv.bias.data.zero_()
                self.freq_weight_conv_list.append(freq_weight_conv)
        else:
            raise NotImplementedError

        # avgpool-based low-pass filter: replicate-pad then average to suppress border artefacts
        if self.lp_type == 'avgpool':
            for k in k_list:
                self.lp_list.append(nn.Sequential(
                    nn.ReplicationPad2d(padding=k // 2),  # ReplicationPad avoids zero-border ringing
                    nn.AvgPool2d(kernel_size=k, padding=0, stride=1),
                ))
        elif self.lp_type in ('laplacian', 'freq'):
            pass  # handled dynamically in forward via FFT masking
        else:
            raise NotImplementedError

        self.act = act

        # Optional per-channel complex attention applied to the full FFT before band splitting
        self.global_selection = global_selection
        if self.global_selection:
            # Separate 1x1 convolutions for real and imaginary parts —
            # real/imag have different statistical distributions in the frequency domain
            self.global_selection_conv_real = nn.Conv2d(
                in_channels, self.spatial_group, kernel_size=1,
                stride=1, groups=self.spatial_group, padding=0, bias=True,
            )
            self.global_selection_conv_imag = nn.Conv2d(
                in_channels, self.spatial_group, kernel_size=1,
                stride=1, groups=self.spatial_group, padding=0, bias=True,
            )
            if init == 'zero':
                for conv in (self.global_selection_conv_real, self.global_selection_conv_imag):
                    conv.weight.data.zero_()
                    conv.bias.data.zero_()

    def sp_act(self, freq_weight):
        """
        Normalise raw attention logits into positive weights.
        - sigmoid: each channel independently in (0, 2), mean ≈ 1 at init
        - softmax: weights sum to `num_groups` across the group dim, enforcing competition
        """
        if self.act == 'sigmoid':
            return freq_weight.sigmoid() * 2
        elif self.act == 'softmax':
            return freq_weight.softmax(dim=1) * freq_weight.shape[1]
        else:
            raise NotImplementedError

    def forward(self, x, att_feat=None):
        # Use x itself as the attention feature source if no external feature is provided
        if att_feat is None:
            att_feat = x

        b, _, h, w = x.shape
        x_list = []
        pre_x  = x.clone()  # tracks the residual low-frequency content after each band is peeled off

        # Shift DC component to center for symmetric frequency masking
        x_fft = torch.fft.fftshift(torch.fft.fft2(x, norm='ortho'))

        # --- Optional: global complex-domain channel attention ---
        if self.global_selection:
            x_real = x_fft.real
            x_imag = x_fft.imag

            # Compute independent attention maps for real and imaginary components
            global_att_real = self.sp_act(self.global_selection_conv_real(x_real))
            global_att_imag = self.sp_act(self.global_selection_conv_imag(x_imag))

            # Reshape to (b, spatial_group, C//group, H, W) for group-wise multiplication
            global_att_real = global_att_real.reshape(b, self.spatial_group, -1, h, w)
            global_att_imag = global_att_imag.reshape(b, self.spatial_group, -1, h, w)
            x_real = x_real.reshape(b, self.spatial_group, -1, h, w)
            x_imag = x_imag.reshape(b, self.spatial_group, -1, h, w)

            # Reconstruct gated complex spectrum and flatten back to (b, C, H, W)
            x_fft = torch.complex(x_real * global_att_real,
                                  x_imag * global_att_imag).reshape(b, -1, h, w)

        # --- Hierarchical band splitting via FFT rectangular low-pass masks ---
        for idx, freq in enumerate(self.k_list):
            # Build a binary rectangular mask that retains the central (low-freq) region.
            # The mask radius scales as H/(2*freq) and W/(2*freq):
            # larger freq → smaller mask → retains only very low frequencies.
            mask = torch.zeros_like(x[:, 0:1, :, :], device=x.device)
            mask[:, :,
                 round(h/2 - h/(2*freq)) : round(h/2 + h/(2*freq)),
                 round(w/2 - w/(2*freq)) : round(w/2 + w/(2*freq))] = 1.0

            # Reconstruct spatial low-freq content for this band and peel off the high-freq residual
            low_part  = torch.fft.ifft2(torch.fft.ifftshift(x_fft * mask), norm='ortho').real
            high_part = pre_x - low_part  # band-pass residual between consecutive low-pass levels
            pre_x     = low_part          # pass the low-freq remainder to the next iteration

            # Apply learned spatial attention to this high-freq band
            freq_weight = self.sp_act(self.freq_weight_conv_list[idx](att_feat))
            tmp = (freq_weight.reshape(b, self.spatial_group, -1, h, w) *
                   high_part.reshape(b, self.spatial_group, -1, h, w))
            x_list.append(tmp.reshape(b, -1, h, w))

        # --- Handle the residual lowest-frequency band ---
        if self.lowfreq_att:
            # Apply learned attention to the lowest-frequency residual as well
            freq_weight = self.sp_act(self.freq_weight_conv_list[len(x_list)](att_feat))
            tmp = (freq_weight.reshape(b, self.spatial_group, -1, h, w) *
                   pre_x.reshape(b, self.spatial_group, -1, h, w))
            x_list.append(tmp.reshape(b, -1, h, w))
        else:
            # Pass the lowest-frequency band through unweighted
            x_list.append(pre_x)

        # Recombine all attended frequency bands
        return sum(x_list)
    