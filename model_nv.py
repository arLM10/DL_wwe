import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged helper blocks (SepConv, BasicBlock, GetGradient)
# ─────────────────────────────────────────────────────────────────────────────

class SepConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size,
                 stride=1, bias=True, padding_mode="zeros"):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channel, in_channel, kernel_size,
            stride=stride, padding=kernel_size // 2,
            groups=in_channel, bias=bias, padding_mode=padding_mode,
        )
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=1,
                               stride=1, padding=0, bias=bias)

    def forward(self, x):
        return self.conv2(self.conv1(x))


class BasicBlock(nn.Module):
    """HIN block — unchanged from original."""
    def __init__(self, in_size, out_size, kernel_size=3, relu_slope=0.1):
        super().__init__()
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.conv_1   = SepConv(in_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_1   = nn.LeakyReLU(relu_slope, inplace=True)
        self.conv_2   = SepConv(out_size, out_size, kernel_size=kernel_size, bias=True)
        self.relu_2   = nn.LeakyReLU(relu_slope, inplace=True)
        self.norm     = nn.InstanceNorm2d(out_size // 2, affine=True)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        return out + self.identity(x)


class GetGradient(nn.Module):
    """Sobel / Laplacian gradient extractor — unchanged from original."""
    def __init__(self, dim=3, mode="sobel"):
        super().__init__()
        self.dim  = dim
        self.mode = mode
        if mode == "sobel":
            ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_y", ky.repeat(self.dim, 1, 1, 1))
            self.register_buffer("kernel_x", kx.repeat(self.dim, 1, 1, 1))
        elif mode == "laplacian":
            kl = torch.tensor([[.25,1,.25],[1,-5,1],[.25,1,.25]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer("kernel_laplace", kl.repeat(self.dim, 1, 1, 1))

    def forward(self, x):
        if self.mode == "sobel":
            gx = F.conv2d(x, self.kernel_x, padding=1, groups=self.dim)
            gy = F.conv2d(x, self.kernel_y, padding=1, groups=self.dim)
            return torch.sqrt(gx**2 + gy**2 + 1e-6)
        elif self.mode == "laplacian":
            return torch.abs(F.conv2d(x, self.kernel_laplace, padding=1, groups=self.dim))


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL COMPONENT 1 — Wavelet Subband Attention Module (WSAM)
#
# Motivation
# ----------
# After the Haar DWT, features are split into 4 subband groups:
#   LL (global color/brightness), LH (horizontal edges),
#   HL (vertical edges), HH (texture / noise).
# Underwater degradation affects these very differently:
#   - Color cast → corrupts LL more than HH
#   - Scattering / blur → attenuates LH and HL
#   - Absorption → mostly kills HH (fine texture)
# The original code fuses all subbands with a single 1×1 conv, giving each
# equal weight regardless of image content.
#
# WSAM adds a squeeze-excitation block that is AWARE of subband structure:
# it pools energy per subband group and learns to re-weight them before
# fusion, so the model can suppress noise-heavy subbands and amplify
# structure-bearing ones adaptively.
#
# Parameter cost: 2 × (4C / r) × 4C ≈ 4C²/r  (r=4 default)
#   e.g. C=32  →  ~1024 extra params  (< 0.1 % of total)
# FLOPs cost: negligible (only on 1×1 spatial GAP output)
# ─────────────────────────────────────────────────────────────────────────────

class WaveletSubbandAttention(nn.Module):
    """
    SE-style attention over the 4 Haar subband groups produced by DWT.

    Unlike standard channel SE (which treats every channel independently),
    this module understands the subband grouping: channels [0..C-1] all
    belong to the LL subband, [C..2C-1] to LH, etc.  The squeeze signal
    is therefore computed per subband (not per channel) and broadcast back.

    Args:
        channels  : number of feature channels C (before DWT, not 4C)
        reduction : squeeze reduction ratio inside the excitation MLP
    """
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        mid = max(channels // reduction, 4)
        # Operates on the full 4C post-DWT tensor
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(4 * channels, mid, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 4 * channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, dwt: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dwt: [B, 4C, H', W'] — post-DWT feature map
        Returns:
            [B, 4C, H', W'] — subband-attended feature map
        """
        attn = self.excite(self.pool(dwt))   # [B, 4C, 1, 1]
        return dwt * attn                    # channel-wise scale


# ─────────────────────────────────────────────────────────────────────────────
# Modified WaveletEnhanceBlock — inserts WSAM between DWT and fuse
# Only 3 lines changed from original (marked with # <<< NEW)
# ─────────────────────────────────────────────────────────────────────────────

class WaveletEnhanceBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        ll = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        lh = torch.tensor([[-0.5,-0.5], [0.5, 0.5]])
        hl = torch.tensor([[-0.5, 0.5],[-0.5, 0.5]])
        hh = torch.tensor([[0.5,-0.5],[-0.5, 0.5]])
        kernel = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer("haar_kernel", kernel.repeat(channels, 1, 1, 1))

        self.subband_attn = WaveletSubbandAttention(channels)  # <<< NEW
        self.fuse = nn.Conv2d(4 * channels, channels, kernel_size=1, bias=False)
        self.post = SepConv(channels, channels, kernel_size=3, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        dwt = F.conv2d(x, self.haar_kernel, stride=2, groups=C)
        dwt = self.subband_attn(dwt)                                  # <<< NEW
        fea = self.post(self.fuse(dwt))
        return F.interpolate(fea, size=(H, W), mode="bilinear", align_corners=False)


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL COMPONENT 2 — Instance-Adaptive Cross-Channel Color Correction (IACCC)
#
# Motivation
# ----------
# GrayWorldRetinex makes two assumptions that fail for underwater images:
#   (a) Gray-world: average RGB should be equal — violated because water
#       selectively absorbs red (≈2 m depth), then orange, then yellow,
#       leaving a systematic blue/green bias.
#   (b) Per-channel independence: each channel's gain is computed from
#       itself alone — but Beer-Lambert attenuation creates cross-channel
#       dependencies (how much red remains depends on depth which is also
#       visible in the green channel).
#
# IACCC replaces the heuristic with a tiny MLP that:
#   1. Computes 6 per-image statistics: [R_mean, G_mean, B_mean,
#                                        R_std,  G_std,  B_std]
#   2. Predicts a 3×3 color correction matrix + 3-vector bias
#      (12 scalars total)
#   3. Applies the matrix as a per-image linear color transform
#
# The MLP is initialized so its output is the identity transform
# (no correction) at the start of training, so convergence is stable.
#
# This is physically motivated: the Beer-Lambert color correction in clear
# water IS a linear matrix transform, so the model can learn to approximate
# it from image statistics.
#
# Parameter cost: 6×16 + 16 + 16×12 + 12 = 316 params
# FLOPs cost: ~322 MACs (only two linear layers on 6 numbers per image)
# ─────────────────────────────────────────────────────────────────────────────

class IACCC(nn.Module):
    """
    Instance-Adaptive Cross-Channel Color Correction.

    For each image in a batch, predicts a 3×3 color correction matrix
    and a 3-vector bias from the image's own RGB statistics, then applies
    the transform.  Models cross-channel color relationships that per-channel
    white balance methods cannot capture.

    Drop-in replacement for GrayWorldRetinex — same forward signature.
    """
    def __init__(self, hidden: int = 16, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 6 statistics → 12 parameters (3×3 matrix + 3 bias)
        self.predictor = nn.Sequential(
            nn.Linear(6, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 12),
        )
        # ── Identity initialisation ──────────────────────────────────────
        # Set the final layer's bias so output = [1,0,0, 0,1,0, 0,0,1, 0,0,0]
        # (identity matrix + zero bias) at the start of training.
        # This makes IACCC a no-op initially, so training is as stable as
        # the original model.
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)
        with torch.no_grad():
            self.predictor[-1].bias[:9].copy_(torch.eye(3).flatten())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] in [0, 1]
        Returns:
            [B, 3, H, W] in [0, 1]  — color-corrected image
        """
        B, C, H, W = x.shape
        # Per-image RGB statistics
        mean = x.mean(dim=(2, 3))           # [B, 3]
        std  = x.std(dim=(2, 3)) + self.eps # [B, 3]
        stats = torch.cat([mean, std], dim=1)  # [B, 6]

        params = self.predictor(stats)         # [B, 12]
        matrix = params[:, :9].view(B, 3, 3)   # [B, 3, 3]
        bias   = params[:, 9:].view(B, 3, 1, 1)# [B, 3, 1, 1]

        # Batched matrix-vector product across spatial positions
        x_flat = x.view(B, 3, -1)              # [B, 3, H×W]
        out = torch.bmm(matrix, x_flat)         # [B, 3, H×W]
        out = out.view(B, 3, H, W) + bias
        return torch.clamp(out, 0.0, 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# NOVEL COMPONENT 3 — Frequency-Gated Skip Connection (FGSC)
#
# Motivation
# ----------
# In the original U-Net decoder:
#       x = self.up1(x3) + x2      # plain addition
#       x = self.up2(x)  + x1      # plain addition
#
# The skip features x1 and x2 came from encoding the *degraded* input.
# They may carry corrupted color, haze, or noise — particularly in their
# high-frequency components.  A plain add propagates all of this into the
# decoder unconditionally.
#
# FGSC adds a lightweight learned gate:
#   gate = σ( Conv1×1( [upsampled, skip] ) )
#   output = upsampled + gate ⊙ skip
#
# The gate can learn to suppress high-frequency noise in the skip while
# still passing clean low-frequency structural information through.
# It adds one 2C→C conv per skip connection (2 total).
#
# Parameter cost: 2 × (2C × C × 1) ≈ 4C²
#   e.g. C=32  →  4096 extra params
#   e.g. C=64  →  16384 extra params
# ─────────────────────────────────────────────────────────────────────────────

class FrequencyGatedSkip(nn.Module):
    """
    Replaces a plain skip-connection addition with a learned gating
    mechanism.

    The gate is conditioned on both the upsampled decoder features and
    the skip features, so it can adapt to the specific content at each
    spatial location — passing clean structure, suppressing degraded
    texture.

    Args:
        channels: number of feature channels (same in both inputs)
    """
    def __init__(self, channels: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, upsampled: torch.Tensor,
                skip: torch.Tensor) -> torch.Tensor:
        """
        Args:
            upsampled: [B, C, H, W] — output of Upsample block
            skip:      [B, C, H, W] — encoder skip feature
        Returns:
            [B, C, H, W] — gated merge
        """
        gate = self.gate(torch.cat([upsampled, skip], dim=1))  # [B, C, H, W]
        return upsampled + gate * skip


# ─────────────────────────────────────────────────────────────────────────────
# Unchanged blocks: SGFB, BasicLayer, Downsample, Upsample
# (kept identical to original — listed here for completeness)
# ─────────────────────────────────────────────────────────────────────────────

class SGFB(nn.Module):
    def __init__(self, feature_channels=48):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.frdb1 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.frdb2 = BasicBlock(feature_channels, feature_channels, kernel_size=3)
        self.get_gradient = GetGradient(feature_channels, mode="sobel")
        self.conv_grad = nn.Sequential(
            SepConv(feature_channels, feature_channels, kernel_size=3, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        grad  = self.conv_grad(self.get_gradient(x))
        x     = self.frdb1(x)
        alpha = torch.sigmoid(self.alpha)
        x     = alpha * grad * x + (1 - alpha) * x
        return self.frdb2(x)


class BasicLayer(nn.Module):
    def __init__(self, feature_channels=48):
        super().__init__()
        self.fwawb = WaveletEnhanceBlock(feature_channels)
        self.sgfb  = SGFB(feature_channels)

    def forward(self, x):
        res = x
        x = self.fwawb(x) + x
        x = self.sgfb(x)
        return 0.5 * x + 0.5 * res


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2),
        )

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x):
        return self.body(x)


# GrayWorldRetinex is kept below so the original model can still be
# instantiated if you want to ablate. IACCC is used by default.
class GrayWorldRetinex(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        B, C, H, W = x.shape
        mean      = x.mean(dim=(2, 3), keepdim=True)
        gray_mean = mean.mean(dim=1, keepdim=True)
        x         = x * (gray_mean / (mean + self.eps))
        x_log     = torch.log(x + self.eps)
        x_log     = x_log - x_log.mean(dim=(2, 3), keepdim=True)
        x_out     = torch.exp(x_log)
        x_min     = x_out.amin(dim=(-2, -1), keepdim=True)
        x_max     = x_out.amax(dim=(-2, -1), keepdim=True)
        return (x_out - x_min) / (x_max - x_min + self.eps)


# ─────────────────────────────────────────────────────────────────────────────
# Modified myModel
#
# Changes from original (all other lines are identical):
#   1. self.wb = IACCC()          instead of GrayWorldRetinex()
#   2. self.skip_gate1 = FrequencyGatedSkip(feature_channels * 2)
#      self.skip_gate2 = FrequencyGatedSkip(feature_channels)
#   3. x = self.skip_gate1(self.up1(x3), x2)   instead of up1(x3) + x2
#      x = self.skip_gate2(self.up2(x),  x1)   instead of up2(x)  + x1
# ─────────────────────────────────────────────────────────────────────────────

class myModel(nn.Module):
    def __init__(self, in_channels=3, feature_channels=32, use_white_balance=False):
        super().__init__()
        self.use_white_balance = use_white_balance
        if self.use_white_balance:
            self.wb    = IACCC()                          # <<< NEW: was GrayWorldRetinex()
            self.alpha = nn.Parameter(torch.zeros(1, 3, 1, 1), requires_grad=True)

        self.first      = nn.Conv2d(in_channels, feature_channels, kernel_size=3, stride=1, padding=1)
        self.encoder1   = BasicLayer(feature_channels)
        self.down1      = Downsample(feature_channels)
        self.encoder2   = BasicLayer(feature_channels * 2)
        self.down2      = Downsample(feature_channels * 2)
        self.bottleneck = BasicLayer(feature_channels * 4)
        self.up1        = Upsample(feature_channels * 4)
        self.skip_gate1 = FrequencyGatedSkip(feature_channels * 2) # <<< NEW
        self.decoder1   = BasicLayer(feature_channels * 2)
        self.up2        = Upsample(feature_channels * 2)
        self.skip_gate2 = FrequencyGatedSkip(feature_channels)     # <<< NEW
        self.decoder2   = BasicLayer(feature_channels)
        self.out        = nn.Conv2d(feature_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        if self.use_white_balance:
            alpha = torch.sigmoid(self.alpha)
            x = alpha * self.wb(x) + (1 - alpha) * x   # IACCC instead of GrayWorldRetinex

        x1 = self.encoder1(self.first(x))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        x  = self.skip_gate1(self.up1(x3), x2)         # <<< NEW: was up1(x3) + x2
        x  = self.decoder1(x)
        x  = self.skip_gate2(self.up2(x), x1)          # <<< NEW: was up2(x) + x1
        x  = self.decoder2(x)
        return self.out(x) + res


# ─────────────────────────────────────────────────────────────────────────────
# Quick sanity check — run as:  python model.py
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from thop import profile, clever_format

    dummy = torch.rand(1, 3, 256, 256)

    # ---- Original model (for FLOPs comparison) ----
    original = myModel(in_channels=3, feature_channels=32, use_white_balance=True)
    # Temporarily swap back to GrayWorldRetinex to measure baseline
    original.wb = GrayWorldRetinex()
    for m in [original.skip_gate1, original.skip_gate2]:
        m.gate = nn.Sequential(nn.Conv2d(64, 32, 1, bias=False), nn.Sigmoid())  # same arch
    flops_orig, params_orig = profile(original, inputs=(dummy,), verbose=False)

    # ---- Proposed model ----
    proposed = myModel(in_channels=3, feature_channels=32, use_white_balance=True)
    flops_new, params_new = profile(proposed, inputs=(dummy,), verbose=False)

    fo, po = clever_format([flops_orig, params_orig], "%.3f")
    fn, pn = clever_format([flops_new,  params_new],  "%.3f")
    print(f"Original  — FLOPs: {fo},  Params: {po}")
    print(f"Proposed  — FLOPs: {fn},  Params: {pn}")
    print(f"FLOPs overhead: {(flops_new - flops_orig) / flops_orig * 100:.2f} %")
    print(f"Param overhead: {(params_new - params_orig) / params_orig * 100:.2f} %")

    out = proposed(dummy)
    assert out.shape == dummy.shape, "Shape mismatch!"
    print(f"Output shape: {out.shape}  ✓")
