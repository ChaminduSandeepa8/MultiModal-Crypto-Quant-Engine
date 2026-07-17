import torch
import torch.nn as nn
import logging
from torch.nn.utils.parametrizations import weight_norm

# =====================================================================
# 🎯 TCN QUANT MODEL - The Pattern Sniper (v4 - Defensive init)
# Hardware target: RTX 3050 (Ubuntu, CUDA 12.1, PyTorch 2.4.0)
# =====================================================================

log = logging.getLogger("TCNModel")


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                            stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                  self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

        self._init_weights()

    def _init_weights(self):
        # ⚠️ [FRAGILITY FIX] `conv.parametrizations.weight.original1` is an internal,
        # undocumented attribute of torch's weight_norm parametrization — not public
        # API. It works on the current PyTorch 2.4.0, but a future version could
        # rename/restructure it. Wrapped in try/except so a change there degrades
        # gracefully to PyTorch's default init instead of crashing at model-construction time.
        for conv in [self.conv1, self.conv2]:
            try:
                nn.init.normal_(conv.parametrizations.weight.original1, mean=0.0, std=0.02)
            except AttributeError:
                log.warning("weight_norm internal API (parametrizations.weight.original1) not found — "
                            "skipping custom init, using PyTorch's default Conv1d init instead.")
            if conv.bias is not None:
                nn.init.zeros_(conv.bias)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNQuantModel(nn.Module):
    def __init__(self, input_dim=20, num_channels=None, kernel_size=3,
                 dropout=0.2, n_classes=3, local_window=128):
        super().__init__()

        if num_channels is None:
            num_channels = [64, 128, 128]

        self.local_window = local_window
        self.receptive_field = 1 + 2 * (kernel_size - 1) * (2 ** len(num_channels) - 1)

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                      dilation=dilation_size,
                                      padding=(kernel_size - 1) * dilation_size,
                                      dropout=dropout)]

        self.tcn = nn.Sequential(*layers)

        norm_dim = num_channels[-1]
        if hasattr(nn, "RMSNorm"):
            self.norm = nn.RMSNorm(norm_dim)
        else:
            self.norm = nn.LayerNorm(norm_dim)

        self.head = nn.Sequential(
            nn.Linear(norm_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes)
        )

    def forward(self, x):
        if x.shape[1] > self.local_window:
            x = x[:, -self.local_window:, :]

        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y.transpose(1, 2)

        local_state = y[:, -1, :]
        global_state = y.mean(dim=1)

        normed = torch.cat([self.norm(local_state), self.norm(global_state)], dim=-1)

        logits = self.head(normed)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("🎯 TCN Pattern Sniper (Multi-Scale) මොඩලය පරීක්ෂා කිරීම ආරම්භ වේ...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ Device: {device} | {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    model = TCNQuantModel(input_dim=20, num_channels=[64, 128, 128],
                           n_classes=3, local_window=128).to(device)
    print(f"📦 Total trainable parameters: {model.count_parameters():,}")
    print(f"📐 Receptive field: {model.receptive_field} ticks | Local window: {model.local_window} ticks")

    dummy_input = torch.randn(1, 500, 20, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = torch.softmax(logits, dim=1)

    vram_used_mb = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0

    print("\n" + "=" * 50)
    print(f"🚀 ආදාන Tensor හැඩය (Input):  {dummy_input.shape}")
    print(f"🎯 ප්‍රතිදාන Logits හැඩය (Output): {logits.shape}")
    print(f"💾 VRAM Allocated: {vram_used_mb:.1f} MB")
    print("=" * 50)
    print("📊 TCN AI එකේ අවසන් තීරණ සම්භාවිතාව (Prediction Probabilities):")
    print(f"   🔴 SELL (විකුණන්න):       {probabilities[0, 0].item()*100:.2f}%")
    print(f"   🟡 HOLD (කිසිවක් නොකරන්න): {probabilities[0, 1].item()*100:.2f}%")
    print(f"   🟢 BUY  (මිලදී ගන්න):       {probabilities[0, 2].item()*100:.2f}%")
    print("=" * 50)
    print("🏆 TCN Multi-Scale Architecture එක සාර්ථකව වැඩ කරයි!")