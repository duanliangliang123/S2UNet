import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from thop import profile
except ImportError:
    profile = None


class SpatialSpectralFusionBlock(nn.Module):
    def __init__(self, in_dim, out_dim, reduction=4, dilations=(2, 4, 6)):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        # Project input features if channel dimensions do not match
        if in_dim != out_dim:
            self.proj = nn.Conv2d(in_dim, out_dim, 1)
        else:
            self.proj = None

        # Helper to determine valid GroupNorm groups
        def _valid_groups(ch):
            g = 8
            while g > 1 and ch % g != 0:
                g //= 2
            if ch % g != 0:
                g = 1
            return g

        gn_groups = _valid_groups(out_dim)

        # Local branch: Captures fine-grained spatial details using standard convolutions
        self.local = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # Regional branch: Uses dilated convolutions to capture meso-scale context without increasing parameters
        self.reg_meso_depthwise = nn.ModuleList()
        for d in dilations:
            self.reg_meso_depthwise.append(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=d, dilation=d,
                          groups=1, bias=False)
            )

        # fuse regional features
        self.fusion_pr = nn.Sequential(
            nn.Conv2d(out_dim * len(dilations), out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # Global branch: FFT-based processing to capture global spectral dependencies
        self.fft_dim = out_dim
        self.spectral_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim * 2, 1, bias=False)
        )
        self.spectral_norm = nn.GroupNorm(gn_groups, out_dim)

        # Fusion layer to combine local, regional, and global features
        total_branches = 3

        self.fusion_pw = nn.Sequential(
            nn.Conv2d(out_dim * total_branches, out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # Channel attention mechanism to re-weight feature channels
        self.att_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // reduction, out_dim, 1, bias=False),
            nn.Sigmoid()
        )

        self.final_smooth = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)

        B, C, H, W = x.shape

        # Process through Local Branch
        feat_local = self.local(x)

        # Process through Regional Branch (Dilated Convs)
        feat_mesos = []
        for dw in self.reg_meso_depthwise:
            feat_mesos.append(F.silu(dw(x)))

        feat_regional = self.fusion_pr(torch.cat(feat_mesos, dim=1))

        # Process through Global Spectral Branch (FFT)
        x_float = x.to(dtype=torch.float32)
        x_fft = torch.fft.rfft2(x_float, norm='ortho')

        # Concatenate real and imaginary parts for processing
        x_fft_cat = torch.cat([x_fft.real, x_fft.imag], dim=1)

        # Apply spectral gating
        x_fft_filtered = self.spectral_gate(x_fft_cat)

        # Residual connection
        x_fft_filtered = x_fft_filtered + x_fft_cat

        # Reconstruct complex tensor
        real, imag = torch.chunk(x_fft_filtered, 2, dim=1)
        x_fft_rec = torch.complex(real, imag)

        # Transform back to spatial domain
        feat_global = torch.fft.irfft2(x_fft_rec, s=(H, W), norm='ortho')
        feat_global = self.spectral_norm(feat_global)

        # Concatenate and fuse all branches
        all_feats = [feat_local] + [feat_regional] + [feat_global]
        cat_feats = torch.cat(all_feats, dim=1)

        out = self.fusion_pw(cat_feats)

        # Apply attention and residual connection
        att = self.att_fc(out)
        out = out * att

        out = self.final_smooth(out) + x

        return out


class StructureAwareRectificationModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Laplacian kernel for edge extraction
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer('laplacian_kernel', kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))

        input_dim = dim * 4
        hidden_dim = max(dim, 16)
        groups = 8 if dim >= 8 and dim % 8 == 0 else 1

        self.context_norm = nn.GroupNorm(groups, input_dim)

        # Networks to predict affine parameters (gamma, beta) for rectification
        self.rectify_net_b_on_a = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim * 2, 1)
        )

        self.rectify_net_a_on_b = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim * 2, 1)
        )

        self.smooth_a = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.SiLU())
        self.smooth_b = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.SiLU())

        # Initialize last layers to zero for identity mapping at the start
        nn.init.zeros_(self.rectify_net_b_on_a[-1].weight)
        nn.init.zeros_(self.rectify_net_b_on_a[-1].bias)
        nn.init.zeros_(self.rectify_net_a_on_b[-1].weight)
        nn.init.zeros_(self.rectify_net_a_on_b[-1].bias)

    def get_structure_map(self, x):
        # Extract structural edge information
        return F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.dim)

    def forward(self, a, b):
        edge_a = self.get_structure_map(a)
        edge_b = self.get_structure_map(b)

        # Concatenate features and their edge maps
        combined_context = torch.cat([a, edge_a, b, edge_b], dim=1)

        combined_context = self.context_norm(combined_context)

        # Predict parameters to rectify A
        params_a = self.rectify_net_b_on_a(combined_context)
        gamma_a, beta_a = torch.chunk(params_a, 2, dim=1)
        gamma_a = torch.sigmoid(gamma_a) * 2.0

        a_rectified = a * gamma_a + beta_a

        # Predict parameters to rectify B
        params_b = self.rectify_net_a_on_b(combined_context)
        gamma_b, beta_b = torch.chunk(params_b, 2, dim=1)
        gamma_b = torch.sigmoid(gamma_b) * 2.0

        b_rectified = b * gamma_b + beta_b

        # Apply smoothing and residual connection
        a_out = self.smooth_a(a_rectified) + a
        b_out = self.smooth_b(b_rectified) + b

        return a_out, b_out


class FeatureProcessingBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        groups = 8 if dim >= 8 else 1
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x) + x


class StructureAwareUncertaintyFusion(nn.Module):
    def __init__(self, dim):
        super().__init__()

        # Rectification modules for aligning NIR features with RGB
        self.sarm_nir1 = StructureAwareRectificationModule(dim)
        self.sarm_nir2 = StructureAwareRectificationModule(dim)

        # Uncertainty estimation network to weigh modalities
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 3, padding=1)
        )

        # Initialize to output low uncertainty initially
        nn.init.constant_(self.uncertainty_estimator[-1].weight, 0)
        nn.init.constant_(self.uncertainty_estimator[-1].bias, 0)

        self.refine = SpatialSpectralFusionBlock(dim, dim)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, x1, x2, x3):
        # x1: RGB, x2: NIR1, x3: NIR2

        # Rectify/Align NIR features based on RGB structural information
        x1_calib1, x2_calib = self.sarm_nir1(x1, x2)
        x1_calib2, x3_calib = self.sarm_nir2(x1, x3)

        # Combine calibrated RGB features
        x1_final = (x1_calib1 + x1_calib2) / 2.0

        stack = torch.stack([x1_final, x2_calib, x3_calib], dim=1)
        B, N, C, H, W = stack.shape
        flat_feats = stack.view(B * N, C, H, W)

        # Estimate pixel-wise uncertainty for each modality
        uncertainty_maps = self.uncertainty_estimator(flat_feats).view(B, N, 1, H, W)

        # Clamp values to ensure numerical stability during Softmax
        uncertainty_maps = torch.clamp(uncertainty_maps, min=-10.0, max=10.0)

        # Calculate fusion weights (inverse variance)
        weights = F.softmax(-uncertainty_maps, dim=1)

        # Calculate channel-wise attention weights
        global_context = self.global_pool(flat_feats).view(B * N, C)
        channel_attn = self.global_fc(global_context).view(B, N, C, 1, 1)

        # Apply weights and fuse
        weighted_stack = stack * weights * channel_attn
        fused_feat = torch.sum(weighted_stack, dim=1)

        # Refine fused features
        out = self.refine(fused_feat)

        return out


class Encoder(nn.Module):
    def __init__(self, dim, dim_mults, num_blocks=1):
        super().__init__()
        self.dims = [dim * m for m in dim_mults]
        in_out = list(zip([dim] + self.dims[:-1], self.dims))

        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * len(in_out)

        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        # Build encoder stages with processing blocks and downsampling
        for (in_dim, out_dim), n in zip(in_out, num_blocks):
            layers = [SpatialSpectralFusionBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            self.downsamples.append(nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1))

        mid_dim = self.dims[-1]
        self.bottleneck = SpatialSpectralFusionBlock(mid_dim, mid_dim)

    def forward(self, x):
        skips = []
        for block, downsample in zip(self.blocks, self.downsamples):
            x = block(x)
            skips.append(x)
            x = downsample(x)
        x = self.bottleneck(x)
        return skips, x


class Decoder(nn.Module):
    def __init__(self, dim, dim_mults, num_blocks=1):
        super().__init__()
        encoder_dims = [dim * m for m in dim_mults]
        reversed_dims = list(reversed(encoder_dims))
        reversed_dims.append(dim)
        in_out = list(zip(reversed_dims, reversed_dims[1:]))
        if isinstance(num_blocks, int):
            num_blocks = [num_blocks] * len(in_out)

        self.upsamples = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.feature_processing_blocks = nn.ModuleList()

        # Build decoder stages with upsampling and feature fusion
        for idx, ((in_dim, out_dim), n) in enumerate(zip(in_out, num_blocks)):
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_dim, in_dim, 3, padding=1)
            ))
            layers = [SpatialSpectralFusionBlock(in_dim + in_dim if i == 0 else out_dim, out_dim) for i in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            self.feature_processing_blocks.append(FeatureProcessingBlock(out_dim))

    def forward(self, x, skips):
        for upsample, block, feature_block, skip in zip(self.upsamples, self.blocks, self.feature_processing_blocks,
                                                        reversed(skips)):
            x = upsample(x)

            # Handle dimension mismatch if necessary
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')

            # Concatenate skip connection
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = feature_block(x)
        return x


class S2UNet(nn.Module):
    def __init__(self, dim=32, dim_mults=(1, 2, 4, 8), num_blocks_encoder=1, num_blocks_decoder=1):
        super().__init__()
        self.init_conv_rgb = nn.Conv2d(3, dim, 7, padding=3)
        self.init_conv_nir = nn.Conv2d(1, dim, 7, padding=3)

        # Independent encoders for each modality
        self.encoder_rgb = Encoder(dim, dim_mults, num_blocks=num_blocks_encoder)
        self.encoder_nir1 = Encoder(dim, dim_mults, num_blocks=num_blocks_encoder)
        self.encoder_nir2 = Encoder(dim, dim_mults, num_blocks=num_blocks_encoder)

        encoder_dims = [dim * m for m in dim_mults]
        mid_dim = encoder_dims[-1]

        # Uncertainty-aware fusion at the bottleneck
        self.bottleneck_fuse = StructureAwareUncertaintyFusion(mid_dim)

        # Uncertainty-aware fusion for skip connections
        self.skip_fuses = nn.ModuleList([StructureAwareUncertaintyFusion(d) for d in encoder_dims])

        self.decoder = Decoder(dim, dim_mults, num_blocks=num_blocks_decoder)
        self.final_conv = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, rgb, nir1, nir2):
        # Encode RGB modality
        x_rgb = self.init_conv_rgb(rgb)
        skips_rgb, x_rgb = self.encoder_rgb(x_rgb)

        # Encode NIR1 modality
        x_nir1 = self.init_conv_nir(nir1)
        skips_nir1, x_nir1 = self.encoder_nir1(x_nir1)

        # Encode NIR2 modality
        x_nir2 = self.init_conv_nir(nir2)
        skips_nir2, x_nir2 = self.encoder_nir2(x_nir2)

        # Fuse deep bottleneck features
        x = self.bottleneck_fuse(x_rgb, x_nir1, x_nir2)

        # Fuse skip connections from all modalities
        fused_skips = []
        for s_rgb, s_nir1, s_nir2, fuse_block in zip(skips_rgb, skips_nir1, skips_nir2, self.skip_fuses):
            fused = fuse_block(s_rgb, s_nir1, s_nir2)
            fused_skips.append(fused)

        # Decode to generate restoration result
        x = self.decoder(x, fused_skips)
        out = self.final_conv(x)
        return out


def calculate_flops_params():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize model with reduced parameters for testing
    model = S2UNet(dim=16, dim_mults=(1, 2, 4, 5),
                             num_blocks_encoder=1, num_blocks_decoder=1).to(device)

    rgb_input = torch.randn(1, 3, 256, 256).to(device)
    nir1_input = torch.randn(1, 1, 256, 256).to(device)
    nir2_input = torch.randn(1, 1, 256, 256).to(device)

    model.eval()
    try:
        with torch.no_grad():
            start = time.time()
            result = model(rgb_input, nir1_input, nir2_input)
            end = time.time()
            print(f"Forward pass successful. Output: {tuple(result.shape)}")
            print(f"Time: {(end - start) * 1000:.2f} ms")

            # Check outputs for numerical validity
            if torch.isnan(result).any():
                print("!!! WARNING: NaN detected in output during inference !!!")
            else:
                print("Output numerical check: PASSED (No NaNs)")

        if profile:
            model.to('cpu')
            rgb_input = rgb_input.to('cpu')
            nir1_input = nir1_input.to('cpu')
            nir2_input = nir2_input.to('cpu')

            flops, params = profile(model, inputs=(rgb_input, nir1_input, nir2_input), verbose=False)
            print(f"FLOPs: {flops / 1e9:.2f} G")
            print(f"Params: {params / 1e6:.2f} M")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    calculate_flops_params()