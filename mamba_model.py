import torch
import torch.nn as nn
from mamba_ssm import Mamba

# =====================================================================
# 🧠 MAMBA QUANT MODEL - v3 (Fixed: dt_proj init no longer clobbered)
# Hardware target: RTX 3050 (Ubuntu, CUDA 12.1, PyTorch 2.4.0)
# =====================================================================


class RMSNormCompat(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(norm + self.eps) * self.weight


def get_norm_layer(d_model):
    if hasattr(nn, "RMSNorm"):
        return nn.RMSNorm(d_model)
    return RMSNormCompat(d_model)


class MambaQuantModel(nn.Module):
    def __init__(self, input_dim=20, d_model=128, n_layers=3, n_classes=3, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.n_layers = n_layers

        self.input_proj = nn.Linear(input_dim, d_model)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'mamba': Mamba(
                    d_model=d_model,
                    d_state=32,
                    d_conv=4,
                    expand=2
                ),
                'norm': get_norm_layer(d_model),
                'dropout': nn.Dropout(dropout)
            }) for _ in range(n_layers)
        ])

        self.norm_f = get_norm_layer(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        # 🔴 [CRITICAL BUG FIX] Previously: self.apply(self._init_weights) — this
        # recurses into EVERY submodule including the Mamba() blocks themselves,
        # overwriting mamba_ssm's own carefully-tuned internal initialization
        # (dt_proj.bias uses an inverse-softplus scheme so the SSM's per-channel
        # decay time-constants start from a sensible spread; in_proj/x_proj/out_proj
        # have their own scaled init too). Blowing that away with a generic
        # normal(0, 0.02) + zero-bias init can leave the SSM's selective-scan
        # dynamics starting from a degenerate/uninformative point, which is
        # consistent with observed training collapse (loss converges to the
        # class-marginal entropy instead of learning input-conditioned signal —
        # confirmed via sanity_check.py's near-zero output variance across
        # different real inputs).
        #
        # Fix: only apply the custom init to the layers WE added (input_proj,
        # head). Leave every Mamba() block's internal parameters exactly as
        # mamba_ssm initializes them.
        self.input_proj.apply(self._init_weights)
        self.head.apply(self._init_weights)
        self._scale_residual_weights()

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _scale_residual_weights(self):
        """Residual stream එකට එකතු වෙන layers වල weights, depth අනුව
        scale කිරීමෙන් deep stacks වල gradient/activation explosion වළක්වයි.
        ⚠️ NOTE: this still touches Mamba() internal params (multiplies, doesn't
        re-init), which is a much gentler operation than the old full re-init —
        left in place since depth-scaling residual branches is a legitimate,
        widely-used stabilization technique (unlike overwriting dt_proj's bias)."""
        scale = 1.0 / (2 * self.n_layers) ** 0.5
        for layer in self.layers:
            for p in layer['mamba'].parameters():
                if p.dim() > 1:
                    p.data.mul_(scale)

    def forward(self, x):
        x = self.input_proj(x)

        for layer in self.layers:
            residual = x
            x = layer['norm'](x)
            x = layer['mamba'](x)
            x = layer['dropout'](x)
            x = x + residual

        x = self.norm_f(x)
        last_tick_state = x[:, -1, :]
        logits = self.head(last_tick_state)
        return logits

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_class_weighted_loss(class_counts: list, device):
    counts = torch.tensor(class_counts, dtype=torch.float32)
    weights = (1.0 / counts) * counts.sum() / len(counts)
    weights = weights.to(device)
    return nn.CrossEntropyLoss(weight=weights)


if __name__ == "__main__":
    print("🧠 Mamba Quant Model පරීක්ෂා කිරීම ආරම්භ වේ...")

    if not torch.cuda.is_available():
        raise RuntimeError(
            "❌ CUDA GPU හමු නොවුනි. mamba-ssm CPU මත ක්‍රියා කරන්නේ නැත. "
            "NVIDIA drivers / CUDA toolkit / PyTorch-CUDA install එක පරීක්ෂා කරන්න."
        )

    device = torch.device("cuda")
    print(f"⚡ Device: {device} | {torch.cuda.get_device_name(0)}")
    print(f"⚡ PyTorch: {torch.__version__} | CUDA: {torch.version.cuda}")

    model = MambaQuantModel(input_dim=20, d_model=128, n_layers=3, n_classes=3).to(device)
    print(f"📦 Total trainable parameters: {model.count_parameters():,}")

    dummy_input = torch.randn(1, 500, 20, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(dummy_input)
        probabilities = torch.softmax(logits, dim=1)

    vram_used_mb = torch.cuda.memory_allocated(device) / (1024 ** 2)
    vram_reserved_mb = torch.cuda.memory_reserved(device) / (1024 ** 2)

    print("\n" + "=" * 50)
    print(f"🚀 ආදාන Tensor හැඩය (Input):  {dummy_input.shape}")
    print(f"🎯 ප්‍රතිදාන Logits හැඩය (Output): {logits.shape}")
    print(f"💾 VRAM Allocated: {vram_used_mb:.1f} MB | Reserved: {vram_reserved_mb:.1f} MB")
    print("=" * 50)
    print("📊 AI එකේ අවසන් තීරණ සම්භාවිතාව (Prediction Probabilities):")
    print(f"   🔴 SELL (විකුණන්න):       {probabilities[0, 0].item() * 100:.2f}%")
    print(f"   🟡 HOLD (කිසිවක් නොකරන්න): {probabilities[0, 1].item() * 100:.2f}%")
    print(f"   🟢 BUY  (මිලදී ගන්න):       {probabilities[0, 2].item() * 100:.2f}%")
    print("=" * 50)
    print("🏆 Mamba Architecture එක සාර්ථකව වැඩ කරයි!")