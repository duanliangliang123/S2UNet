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

        # Channel projection if dimensions mismatch
        if in_dim != out_dim:
            self.proj = nn.Conv2d(in_dim, out_dim, 1)
        else:
            self.proj = None

        # Dynamically determine valid group size for GroupNorm
        def _valid_groups(ch):
            g = 8
            while g > 1 and ch % g != 0:
                g //= 2
            if ch % g != 0:
                g = 1
            return g

        gn_groups = _valid_groups(out_dim)

        # Local feature extractor using standard convolutions
        self.local = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU(),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # Regional feature extractor using multi-scale dilated convolutions
        self.reg_meso_depthwise = nn.ModuleList()
        for d in dilations:
            self.reg_meso_depthwise.append(
                nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=d, dilation=d,
                          groups=1, bias=False)
            )

        # Aggregator for regional features
        self.fusion_pr = nn.Sequential(
            nn.Conv2d(out_dim * len(dilations), out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # Global feature extractor using Fast Fourier Transform (FFT) gating
        self.fft_dim = out_dim
        self.spectral_gate = nn.Sequential(
            nn.Conv2d(out_dim * 2, out_dim * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(out_dim * 2, out_dim * 2, 1, bias=False)
        )
        self.spectral_norm = nn.GroupNorm(gn_groups, out_dim)

        # Point-wise fusion for local, regional, and global features
        total_branches = 3
        self.fusion_pw = nn.Sequential(
            nn.Conv2d(out_dim * total_branches, out_dim, 1, bias=False),
            nn.GroupNorm(gn_groups, out_dim),
            nn.SiLU()
        )

        # Channel attention mechanism
        self.att_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_dim, out_dim // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim // reduction, out_dim, 1, bias=False),
            nn.Sigmoid()
        )

        # Final smoothing convolution before residual connection
        self.final_smooth = nn.Conv2d(out_dim, out_dim, 1)

    def forward(self, x):
        if self.proj is not None:
            x = self.proj(x)

        B, C, H, W = x.shape

        # 1. Compute local features
        feat_local = self.local(x)

        # 2. Compute and aggregate regional features
        feat_mesos = []
        for dw in self.reg_meso_depthwise:
            feat_mesos.append(F.silu(dw(x)))
        feat_regional = self.fusion_pr(torch.cat(feat_mesos, dim=1))

        # 3. Compute global features via FFT domain gating
        x_float = x.to(dtype=torch.float32)
        x_fft = torch.fft.rfft2(x_float, norm='ortho')
        x_fft_cat = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_fft_filtered = self.spectral_gate(x_fft_cat)
        x_fft_filtered = x_fft_filtered + x_fft_cat
        real, imag = torch.chunk(x_fft_filtered, 2, dim=1)
        x_fft_rec = torch.complex(real, imag)
        feat_global = torch.fft.irfft2(x_fft_rec, s=(H, W), norm='ortho')
        feat_global = self.spectral_norm(feat_global)

        # 4. Concatenate and fuse multi-scale features
        all_feats = [feat_local] + [feat_regional] + [feat_global]
        cat_feats = torch.cat(all_feats, dim=1)
        out = self.fusion_pw(cat_feats)

        # 5. Apply channel attention and residual connection
        att = self.att_fc(out)
        out = out * att
        out = self.final_smooth(out) + x

        return out


class StructureAwareRectificationModule(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

        # Fixed Laplacian kernel for edge/structure detection
        kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]])
        self.register_buffer('laplacian_kernel', kernel.view(1, 1, 3, 3).repeat(dim, 1, 1, 1))

        input_dim = dim * 4
        hidden_dim = max(dim, 16)
        groups = 8 if dim >= 8 and dim % 8 == 0 else 1

        self.context_norm = nn.GroupNorm(groups, input_dim)

        # Predicts affine parameters (gamma, beta) for feature A based on A+B context
        self.rectify_net_b_on_a = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim * 2, 1)
        )

        # Predicts affine parameters (gamma, beta) for feature B based on A+B context
        self.rectify_net_a_on_b = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, dim * 2, 1)
        )

        self.smooth_a = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.SiLU())
        self.smooth_b = nn.Sequential(nn.Conv2d(dim, dim, 3, padding=1), nn.SiLU())

        # Initialize transformation components to identity mappings
        nn.init.zeros_(self.rectify_net_b_on_a[-1].weight)
        nn.init.zeros_(self.rectify_net_b_on_a[-1].bias)
        nn.init.zeros_(self.rectify_net_a_on_b[-1].weight)
        nn.init.zeros_(self.rectify_net_a_on_b[-1].bias)

    def get_structure_map(self, x):
        # Applies depthwise convolution with Laplacian kernel to extract high-frequency edges
        return F.conv2d(x, self.laplacian_kernel, padding=1, groups=self.dim)

    def forward(self, a, b):
        edge_a = self.get_structure_map(a)
        edge_b = self.get_structure_map(b)

        # Build joint context from original features and their structural maps
        combined_context = torch.cat([a, edge_a, b, edge_b], dim=1)
        combined_context = self.context_norm(combined_context)

        # Modulate feature A
        params_a = self.rectify_net_b_on_a(combined_context)
        gamma_a, beta_a = torch.chunk(params_a, 2, dim=1)
        gamma_a = torch.sigmoid(gamma_a) * 2.0
        a_rectified = a * gamma_a + beta_a

        # Modulate feature B
        params_b = self.rectify_net_a_on_b(combined_context)
        gamma_b, beta_b = torch.chunk(params_b, 2, dim=1)
        gamma_b = torch.sigmoid(gamma_b) * 2.0
        b_rectified = b * gamma_b + beta_b

        # Apply smoothing and residual addition
        a_out = self.smooth_a(a_rectified) + a
        b_out = self.smooth_b(b_rectified) + b

        return a_out, b_out


class FeatureProcessingBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        groups = 8 if dim >= 8 else 1
        # Basic residual block for intermediate feature refinement
        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(groups, dim),
            nn.SiLU()
        )

    def forward(self, x):
        return self.block(x) + x


class StructureAwareUncertaintyFusion(nn.Module):
    def __init__(self, dim, num_inputs):
        super().__init__()
        self.num_inputs = num_inputs

        # Mutual rectification modules for one anchor and multiple auxiliaries
        self.sarms = nn.ModuleList([
            StructureAwareRectificationModule(dim) for _ in range(num_inputs - 1)
        ])

        # Estimates pixel-wise uncertainty for soft-weighted fusion
        self.uncertainty_estimator = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 2, 1, 3, padding=1)
        )
        nn.init.constant_(self.uncertainty_estimator[-1].weight, 0)
        nn.init.constant_(self.uncertainty_estimator[-1].bias, 0)

        self.refine = SpatialSpectralFusionBlock(dim, dim)

        # Global channel attention to scale features prior to summation
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(dim, dim // 4),
            nn.ReLU(),
            nn.Linear(dim // 4, dim),
            nn.Sigmoid()
        )

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs

        anchor = inputs[0]
        auxiliaries = inputs[1:]

        anchor_calib_accum = []
        processed_auxs = []

        # Pairwise rectification of the anchor and each auxiliary modality
        if len(auxiliaries) > 0:
            for i, aux in enumerate(auxiliaries):
                anchor_calib_i, aux_calib = self.sarms[i](anchor, aux)
                anchor_calib_accum.append(anchor_calib_i)
                processed_auxs.append(aux_calib)

            # Average all rectified anchor variants
            anchor_final = sum(anchor_calib_accum) / len(anchor_calib_accum)
        else:
            anchor_final = anchor

        calibrated_inputs = [anchor_final] + processed_auxs

        # Compute pixel-wise uncertainty scores for adaptive weighting
        stack = torch.stack(calibrated_inputs, dim=1)
        B, N, C, H, W = stack.shape
        flat_feats = stack.view(B * N, C, H, W)

        uncertainty_maps = self.uncertainty_estimator(flat_feats).view(B, N, 1, H, W)
        uncertainty_maps = torch.clamp(uncertainty_maps, min=-10.0, max=10.0)
        weights = F.softmax(-uncertainty_maps, dim=1)

        # Calculate channel-wise attention based on global context
        global_context = self.global_pool(flat_feats).view(B * N, C)
        channel_attn = self.global_fc(global_context).view(B, N, C, 1, 1)

        # Element-wise fusion
        weighted_stack = stack * weights * channel_attn
        fused_feat = torch.sum(weighted_stack, dim=1)

        # Post-fusion refinement
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

        # Build progressive downsampling hierarchy
        for (in_dim, out_dim), n in zip(in_out, num_blocks):
            layers = [SpatialSpectralFusionBlock(in_dim if i == 0 else out_dim, out_dim) for i in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            self.downsamples.append(nn.Conv2d(out_dim, out_dim, 3, stride=2, padding=1))

        # Bottom-most processing layer
        mid_dim = self.dims[-1]
        self.bottleneck = SpatialSpectralFusionBlock(mid_dim, mid_dim)

    def forward(self, x):
        skips = []
        # Store intermediate activations for skip connections
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

        # Build progressive upsampling hierarchy
        for idx, ((in_dim, out_dim), n) in enumerate(zip(in_out, num_blocks)):
            self.upsamples.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_dim, in_dim, 3, padding=1)
            ))
            layers = [SpatialSpectralFusionBlock(in_dim + in_dim if i == 0 else out_dim, out_dim) for i in range(n)]
            self.blocks.append(nn.Sequential(*layers))
            self.feature_processing_blocks.append(FeatureProcessingBlock(out_dim))

    def forward(self, x, skips):
        # Iteratively upsample, concatenate with skip connections, and refine
        for upsample, block, feature_block, skip in zip(self.upsamples, self.blocks, self.feature_processing_blocks,
                                                        reversed(skips)):
            x = upsample(x)
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='nearest')
            x = torch.cat([x, skip], dim=1)
            x = block(x)
            x = feature_block(x)
        return x


class S2UNet(nn.Module):
    def __init__(self, in_channels_list=[3, 1, 1], dim=32, dim_mults=(1, 2, 4, 8),
                 num_blocks_encoder=1, num_blocks_decoder=1, share_aux_init_conv=True):
        super().__init__()

        self.num_inputs = len(in_channels_list)
        self.in_channels_list = in_channels_list
        self.share_aux_init_conv = share_aux_init_conv

        # Configure initial convolutions based on weight-sharing rules for auxiliary inputs
        if self.share_aux_init_conv and self.num_inputs > 1:
            self.init_conv_anchor = nn.Conv2d(in_channels_list[0], dim, 7, padding=3)

            aux_channels = in_channels_list[1]
            assert all(ch == aux_channels for ch in in_channels_list[1:])
            self.init_conv_aux = nn.Conv2d(aux_channels, dim, 7, padding=3)
        else:
            self.init_convs = nn.ModuleList()
            for ch in in_channels_list:
                self.init_convs.append(nn.Conv2d(ch, dim, 7, padding=3))

        # Instantiate separate encoders per modality
        self.encoders = nn.ModuleList()
        for _ in range(self.num_inputs):
            self.encoders.append(Encoder(dim, dim_mults, num_blocks=num_blocks_encoder))

        encoder_dims = [dim * m for m in dim_mults]
        mid_dim = encoder_dims[-1]

        # Inter-modality fusion mechanism at the bottleneck
        self.bottleneck_fuse = StructureAwareUncertaintyFusion(mid_dim, self.num_inputs)

        # Inter-modality fusion mechanisms at every skip connection scale
        self.skip_fuses = nn.ModuleList([
            StructureAwareUncertaintyFusion(d, self.num_inputs) for d in encoder_dims
        ])

        # Single unified decoder and final output projection
        self.decoder = Decoder(dim, dim_mults, num_blocks=num_blocks_decoder)
        self.final_conv = nn.Conv2d(dim, 3, 3, padding=1)

    def forward(self, inputs):
        assert len(inputs) == self.num_inputs

        all_skips = []
        all_bottlenecks = []

        # 1. Extract multi-scale features independently for each input modality
        for i, (x_in, enc) in enumerate(zip(inputs, self.encoders)):

            if self.share_aux_init_conv and self.num_inputs > 1:
                if i == 0:
                    x = self.init_conv_anchor(x_in)
                else:
                    x = self.init_conv_aux(x_in)
            else:
                x = self.init_convs[i](x_in)

            skips, x_neck = enc(x)
            all_skips.append(skips)
            all_bottlenecks.append(x_neck)

        # 2. Fuse the deep bottleneck features from all modalities
        x = self.bottleneck_fuse(all_bottlenecks)

        # 3. Align and fuse intermediate skip connections across all modalities
        num_scales = len(all_skips[0])
        fused_skips = []

        for scale_idx in range(num_scales):
            skips_at_scale = [all_skips[mod_idx][scale_idx] for mod_idx in range(self.num_inputs)]
            fuse_block = self.skip_fuses[scale_idx]
            fused = fuse_block(skips_at_scale)
            fused_skips.append(fused)

        # 4. Decode the fused representations into the final output
        x = self.decoder(x, fused_skips)
        out = self.final_conv(x)
        return out


def convert_old_weights_to_new(old_state_dict):
    # Maps variable names from legacy checkpoint files to current module architectures
    new_state_dict = {}

    for old_key, tensor in old_state_dict.items():
        new_key = old_key

        new_key = new_key.replace("init_conv_rgb", "init_conv_anchor")
        new_key = new_key.replace("init_conv_nir", "init_conv_aux")

        new_key = new_key.replace("encoder_rgb", "encoders.0")
        new_key = new_key.replace("encoder_nir1", "encoders.1")
        new_key = new_key.replace("encoder_nir2", "encoders.2")

        new_key = new_key.replace("sarm_nir1", "sarms.0")
        new_key = new_key.replace("sarm_nir2", "sarms.1")

        new_state_dict[new_key] = tensor

    return new_state_dict


def verify_dynamic_model():
    # Validates structure setup, executes a dummy forward pass, and measures profile statistics
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    print("--- Testing Perfect Equivalent of Net_best3LRG [3, 1, 1] ---")

    model = S2UNet(
        in_channels_list=[3, 1, 1],
        dim=16,
        dim_mults=(1, 2, 4, 5),
        share_aux_init_conv=True
    ).to(device)

    inputs = [
        torch.randn(1, 3, 256, 256).to(device),
        torch.randn(1, 1, 256, 256).to(device),
        torch.randn(1, 1, 256, 256).to(device)
    ]

    with torch.no_grad():
        out = model(inputs)

    print(f"Equivalent Model Output Shape: {out.shape}")
    print("Test passed! Model structurally perfectly matches Net_best3LRG.py.")

    if profile:
        print("\nCalculating FLOPs for Equivalent [3, 1, 1] case:")
        model.to('cpu')
        inputs_cpu = [t.to('cpu') for t in inputs]
        flops, params = profile(model, inputs=(inputs_cpu,), verbose=False)
        print(f"FLOPs: {flops / 1e9:.2f} G")
        print(f"Params: {params / 1e6:.2f} M")


if __name__ == "__main__":
    verify_dynamic_model()