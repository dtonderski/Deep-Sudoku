import einops
import torch
from einops.layers.torch import Rearrange
from torch import nn

from deepsudoku.utils.network_utils import to_categorical


class Embedding(nn.Module):
    def __init__(self, latent_vector_size, input_channels=9, patches=82):
        super().__init__()
        self.project = nn.Conv2d(
            input_channels, latent_vector_size, kernel_size=1
        )
        self.flatten = Rearrange("b c h w -> b (h w) c")
        self.cls = nn.Parameter(torch.randn(1, 1, latent_vector_size))
        self.pos = nn.Parameter(torch.randn(1, patches, latent_vector_size))

    def forward(self, x):
        x = self.flatten(self.project(x))
        cls = einops.repeat(self.cls, "() n e -> b n e", b=x.shape[0])
        x = torch.cat([cls, x], dim=1)
        return x + self.pos


class MultiHeadedAttention(nn.Module):
    def __init__(self, latent_vector_size, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.scale = (latent_vector_size / n_heads) ** (-0.5)
        self.latent_vector_size = torch.tensor(latent_vector_size)
        self.qkv = nn.Linear(latent_vector_size, latent_vector_size * 3)
        self.proj = nn.Linear(latent_vector_size, latent_vector_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """

        :param x: (1, 82, latent_vector_size)
        :return:
        """
        # b - batches, p - patches, h - heads, d - head_embedding
        qkv = einops.rearrange(
            self.qkv(x), "b p (h d qkv) -> qkv b h p d", h=self.n_heads, qkv=3
        )

        queries, keys, values = qkv
        transposed_keys = einops.rearrange(keys, "b h p d -> b h d p")
        dot = torch.matmul(queries, transposed_keys)
        scaled_dot = dot * self.scale
        softmaxed_dot = torch.softmax(scaled_dot, dim=-1)

        attention = torch.matmul(softmaxed_dot, values)
        attention = einops.rearrange(attention, "b h p d -> b p (h d)")

        output = self.proj(attention)
        output = self.dropout(output)

        return output


class MHABlock(nn.Module):
    def __init__(self, latent_vector_size, n_heads, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(latent_vector_size)
        self.mha = MultiHeadedAttention(latent_vector_size, n_heads, dropout)

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.mha(x)
        return x + skip


class MLPBlock(nn.Module):
    def __init__(self, latent_vector_size, n_hidden, dropout):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.LayerNorm(latent_vector_size),
            nn.Linear(latent_vector_size, n_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(n_hidden, latent_vector_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        skip = x
        x = self.mlp(x)
        return x + skip


class EncoderBlock(nn.Sequential):
    def __init__(self, latent_vector_size, n_heads, n_hidden, dropout):
        super().__init__(
            MHABlock(latent_vector_size, n_heads, dropout),
            MLPBlock(latent_vector_size, n_hidden, dropout),
        )


class Decoder(nn.Sequential):
    def __init__(self, latent_vector_size):
        super().__init__()
        self.norm = nn.Sequential(nn.LayerNorm(latent_vector_size))

        self.policy_head = nn.Linear(latent_vector_size, 9)
        self.value_head = nn.Linear(latent_vector_size, 1)

    def forward(self, x):
        x = self.norm(x)
        p = self.policy_head(x[:, 1:])
        p_permuted = p.permute(0, 2, 1)

        p_reshaped = torch.reshape(p_permuted, (-1, 9, 9, 9))
        v = self.value_head(x[:, :1])[:, :, 0]
        return p_reshaped, v


class Transformer(nn.Module):
    def __init__(self, depth, latent_vector_size, n_heads, n_hidden, dropout):
        super().__init__()
        self.embedding = Embedding(latent_vector_size)
        self.encoder = nn.Sequential(*[
            EncoderBlock(latent_vector_size, n_heads, n_hidden, dropout)
            for _ in range(depth)
        ])
        self.decoder = Decoder(latent_vector_size)

    def forward(self, x):
        if x.shape[1] == 1:
            # Must one-hot encode!
            x = to_categorical(x)
        x = self.embedding(x)
        x = self.encoder(x)
        p, v = self.decoder(x)
        return p, v


def main():
    sudoku = torch.rand((5, 1, 9, 9))
    transformer = Transformer(6, 256, 8, 1024, 0)
    x = transformer(sudoku)
    print(x[0].shape, x[1].shape)


if __name__ == "__main__":
    main()
