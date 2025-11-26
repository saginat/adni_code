import torch
import torch.nn as nn
import einops
import math


class PatchMerger(nn.Module):
    def __init__(self, dim, num_tokens_out):
        super().__init__()
        self.scale = dim**-0.5
        self.norm = nn.LayerNorm(dim)
        self.queries = nn.Parameter(torch.randn(num_tokens_out, dim))

    def forward(self, x):
        x = self.norm(x)
        sim = torch.matmul(self.queries, x.transpose(-1, -2)) * self.scale
        attn = sim.softmax(dim=-1)
        return torch.matmul(attn, x)


class AtlasDecoder(nn.Module):
    def __init__(self, input_dim, output_dims, p_dropout=0.0, first_pass=True):
        super().__init__()

        self.atlas_n, self.t = output_dims
        self.first_pass = first_pass

        self.input_layer = nn.Sequential(
            nn.LazyLinear(input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
        )

        self.intermediate_layer = nn.Sequential(
            nn.Linear(input_dim, input_dim // 3),
            nn.LayerNorm(input_dim // 3),
            nn.GELU(),
            nn.Dropout(p_dropout),
        )

        self.projection_layer = nn.Linear(input_dim // 3, self.atlas_n * self.t)
        self.unflatten = nn.Unflatten(1, (self.atlas_n, self.t))

    def forward(self, x):
        x = einops.rearrange(x, "b tokens transformer_d -> b (tokens transformer_d)")
        x = self.input_layer(x)
        x = self.intermediate_layer(x)
        x = self.projection_layer(x)
        x = self.unflatten(x)
        self.first_pass = False
        return x


class PatchSelection(nn.Module):
    def __init__(self, reduced_patches):
        super().__init__()
        self.linear = nn.LazyLinear(reduced_patches)
        self.layer_norm = nn.LayerNorm(reduced_patches)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.linear(x)
        x = self.layer_norm(x)
        x = self.gelu(x)
        return x


class TransformerAutoEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        patch_size,
        embedding_dim,
        num_heads,
        num_layers,
        output_dims,
        num_of_spatial_patches,
        num_cls_tokens=0,
        reduced_patches_factor_percent=1,
        reduce_time_factor_percent=1,
        p_dropout=0.0,
        custom_decoder=None,
        merge_patches=10,
        use_patch_merger=True,
        first_pass=True,
    ):
        super().__init__()
        self.h, self.w, self.d, self.time = input_size
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        self.p_dropout = p_dropout
        self.num_of_spatial_patches = num_of_spatial_patches
        self.first_pass = first_pass
        self.use_temporal_selection = reduce_time_factor_percent < 1
        self.reduce_time_factor_percent = reduce_time_factor_percent
        self.use_patch_selection = reduced_patches_factor_percent < 1
        self.use_patch_merger = use_patch_merger

        if self.use_patch_selection:
            self.reduced_patches = int(
                num_of_spatial_patches * reduced_patches_factor_percent
            )
            self.patch_selection = PatchSelection(self.reduced_patches)
        else:
            self.reduced_patches = num_of_spatial_patches

        if self.use_temporal_selection:
            self.reduced_time_patches = int(self.time * reduce_time_factor_percent)
            self.temporal_selection = nn.LazyLinear(self.reduced_time_patches)
        else:
            self.reduced_time_patches = self.time

        self.patch_embed = nn.LazyLinear(embedding_dim)
        self.num_cls_tokens = num_cls_tokens
        if num_cls_tokens > 0:
            self.cls_tokens = nn.Parameter(
                torch.randn(1, num_cls_tokens, embedding_dim)
            )

        self.num_of_tokens = (
            self.reduced_patches * self.reduced_time_patches + self.num_cls_tokens
        )

        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, num_of_spatial_patches, embedding_dim // 2)
        )
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, self.time, embedding_dim // 2)
        )

        # Optional: learnable CLS token positional embedding
        if self.num_cls_tokens > 0:
            self.cls_pos_embed = nn.Parameter(
                torch.randn(1, num_cls_tokens, embedding_dim)
            )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, nhead=num_heads, batch_first=True
            ),
            num_layers=num_layers,
        )

        if self.use_patch_merger:
            self.merge_patches = PatchMerger(embedding_dim, merge_patches)

        decoder_input_dim = (
            embedding_dim * merge_patches
            if self.use_patch_merger
            else embedding_dim * self.reduced_patches * self.reduced_time_patches
        )

        self.custom_recon_decoder = AtlasDecoder(
            input_dim=decoder_input_dim, output_dims=(200, self.time)
        )

    def create_2d_positional_embeddings(self, patches_shape):
        B, T, n_spatial_patches, _ = patches_shape

        spatial_pos = self.spatial_pos_embed[:, :n_spatial_patches]
        temporal_pos = self.temporal_pos_embed[:, :T]

        spatial_pos = spatial_pos.unsqueeze(1)
        temporal_pos = temporal_pos.unsqueeze(2)

        spatial_expanded = spatial_pos.expand(1, T, n_spatial_patches, -1)
        temporal_expanded = temporal_pos.expand(1, T, n_spatial_patches, -1)

        pos_embed_2d = torch.cat([spatial_expanded, temporal_expanded], dim=-1)

        pos_embed_2d = einops.rearrange(pos_embed_2d, "1 t p emb -> 1 (t p) emb")

        return pos_embed_2d.expand(B, -1, -1)

    def reconstruct(self, decoded_patches, B, H, W, D, pad_h, pad_w, pad_d):
        total_patches = decoded_patches.shape[1]
        num_patches_h = (H + pad_h) // self.patch_size[0]
        num_patches_w = (W + pad_w) // self.patch_size[1]
        num_patches_d = total_patches // (num_patches_h * num_patches_w)

        decoded = decoded_patches.view(
            B,
            num_patches_h,
            num_patches_w,
            num_patches_d,
            self.patch_size[0],
            self.patch_size[1],
            self.patch_size[2],
        )
        decoded = decoded.permute(0, 4, 5, 6, 1, 2, 3).contiguous()
        decoded = decoded.view(B, -1, H + pad_h, W + pad_w, D + pad_d)
        return einops.rearrange(decoded[:, :, :H, :W, :D], "b t h w d -> b h w d t")

    def patchify(self, x):
        """
        Convert input tensor into patches.

        Args:
            x (torch.Tensor): Input tensor with shape (B, T, H, W, D)

        Returns:
            torch.Tensor: Patches with shape (B, T, n_spatial_patches, patch_dim)
            tuple: Padding dimensions (pad_h, pad_w, pad_d)
        """
        x = einops.rearrange(x, "b h w d t -> b t h w d")
        B, T, H, W, D = x.shape

        pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
        pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
        pad_d = (self.patch_size[2] - D % self.patch_size[2]) % self.patch_size[2]

        x = nn.functional.pad(x, (0, pad_d, 0, pad_w, 0, pad_h))

        x = x.unfold(2, self.patch_size[0], self.patch_size[0])  # Patches along H
        x = x.unfold(3, self.patch_size[1], self.patch_size[1])  # Patches along W
        x = x.unfold(4, self.patch_size[2], self.patch_size[2])  # Patches along D

        self.n_of_spatial_patches = x.size(2) * x.size(3) * x.size(4)
        patch_dim = self.patch_size[0] * self.patch_size[1] * self.patch_size[2]
        patches = x.reshape(B, T, self.n_of_spatial_patches, patch_dim)

        return patches, (pad_h, pad_w, pad_d), (H, W, D, T)

    def patch_embedding(self, patches):
        """
        Embed patches and add 2D positional encoding.

        Args:
            patches (torch.Tensor): Patches tensor

        Returns:
            torch.Tensor: Embedded and positionally encoded tokens
        """
        B = patches.shape[0]
        patches_reshaped = einops.rearrange(patches, "b t p p_dim -> b (t p) p_dim")
        tokens = self.patch_embed(patches_reshaped)

        # Add 2D positional embeddings
        pos_embed_2d = self.create_2d_positional_embeddings(patches.shape)
        tokens += pos_embed_2d

        # Add CLS tokens if configured
        if hasattr(self, "num_cls_tokens") and self.num_cls_tokens > 0:
            cls_tokens = self.cls_tokens.expand(B, -1, -1)
            if hasattr(self, "cls_pos_embed"):
                cls_tokens += self.cls_pos_embed.expand(B, -1, -1)
            print(
                f"Adding {self.num_cls_tokens} cls tokens"
            ) if self.first_pass else None
            tokens = torch.cat([cls_tokens, tokens], dim=1)

        return tokens

    def spatial_temporal_reduction(self, encoded):
        """
        Perform spatial and temporal reduction if configured.
        """
        if hasattr(self, "num_cls_tokens") and self.num_cls_tokens > 0:
            cls_tokens = encoded[:, : self.num_cls_tokens]
            other_tokens = encoded[:, self.num_cls_tokens :]
        else:
            cls_tokens = None
            other_tokens = encoded
            del encoded

        other_tokens = einops.rearrange(
            other_tokens,
            "b (t p) transformer_d -> b t p transformer_d",
            t=self.time,
            p=self.n_of_spatial_patches,
        )

        if self.use_patch_selection:
            other_tokens = einops.rearrange(
                other_tokens, "b t p transformer_d -> b t transformer_d p"
            )
            other_tokens = self.patch_selection(other_tokens)
            other_tokens = einops.rearrange(
                other_tokens, "b t transformer_d p -> b t p transformer_d"
            )

        if self.use_temporal_selection:
            other_tokens = einops.rearrange(
                other_tokens, "b t p transformer_d -> b p transformer_d t"
            )
            other_tokens = self.temporal_selection(other_tokens)
            other_tokens = einops.rearrange(
                other_tokens, "b p transformer_d t -> b (t p) transformer_d"
            )
        else:
            other_tokens = einops.rearrange(other_tokens, "b t p d -> b (t p) d")

        if self.use_patch_merger:
            other_tokens = self.merge_patches(other_tokens)

        return cls_tokens, other_tokens

    def encode(self, x):
        # Reshape to patches
        patches, (pad_h, pad_w, pad_d), (H, W, D, T) = self.patchify(x)
        # Embed patches with 2D positional embeddings
        tokens = self.patch_embedding(patches)
        # Transformer Encoding
        encoded = self.encoder(tokens)

        cls_tokens_encoded, other_tokens_encoded = self.spatial_temporal_reduction(
            encoded
        )

        return cls_tokens_encoded, other_tokens_encoded

    def forward(self, x, custom_recon=None):
        cls_tokens, other_tokens = self.encode(x)
        decoded = self.custom_recon_decoder(other_tokens)

        task_predictions = {}
        task_predictions["Reconstruction"] = decoded
        self.first_pass = False

        return task_predictions
