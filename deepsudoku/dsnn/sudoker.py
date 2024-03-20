import torch
from torch import nn
import einops
from deepsudoku.utils.network_utils import to_categorical
from deepsudoku.dsnn.transformer import Embedding, MLPBlock, Decoder


class MultiHeadedAttention(nn.Module):
    def __init__(self, latent_vector_size, n_heads, dropout, mode="regular"):
        super().__init__()
        assert mode in ["regular", "row", "block", "col"]
        self.n_heads = n_heads
        self.scale = (latent_vector_size / n_heads) ** (-0.5)
        self.latent_vector_size = torch.tensor(latent_vector_size)
        self.qkv = nn.Linear(latent_vector_size, latent_vector_size * 3)
        self.proj = nn.Linear(latent_vector_size, latent_vector_size)
        self.dropout = nn.Dropout(dropout)
        self.mode = mode

    def forward(self, x):
        """

        :param x: (b, 82, latent_vector_size)
        :return:
        """
        if self.mode != "regular":
            cls, x = torch.tensor_split(x, [1], dim=1)
            # cls: (b,1,latent_vector_size), x:(b,81,latent_vector_size)
            if self.mode == "block":
                x = einops.rearrange(
                    x,
                    "b (h_blocks block_h w_blocks block_w) d "
                    "-> (b h_blocks w_blocks) (block_h block_w) d",
                    h_blocks=3,
                    w_blocks=3,
                    block_h=3,
                    block_w=3,
                )
            elif self.mode == "row":
                x = einops.rearrange(x, "b (h w) d -> (b h) w d", h=9)
            else:
                # col
                x = einops.rearrange(x, "b (h w) d -> (b w) h d", w=9)
            # x: (b*9, 9, latent_vector_size)

            cls = einops.repeat(cls, "b () d -> (b p) () d", p=9)
            # cls: (b*9, 1, latent_vector_size)

            x = torch.cat([cls, x], dim=1)
            # x: (b*9, 10, latent_vector_size)

        # b - batches, p - patches, h - heads, d - head_embedding
        qkv = einops.rearrange(
            self.qkv(x), "b p (h d qkv) -> qkv b h p d", h=self.n_heads, qkv=3
        )
        # qkv: (3, b*9, h, 10, d)

        queries, keys, values = qkv
        transposed_keys = einops.rearrange(keys, "b h p d -> b h d p")
        # k: (b*9, h, d, 10)
        dot = torch.matmul(queries, transposed_keys)
        # dot: (b*9, h, 10, 10)
        scaled_dot = dot * self.scale
        softmaxed_dot = torch.softmax(scaled_dot, dim=-1)
        # softmaxed_dot: (b*9, h, 10,10)

        attention = torch.matmul(softmaxed_dot, values)
        # attention: (b*9, h, 10, d)
        attention = einops.rearrange(attention, "b h p d -> b p (h d)")
        # attention: (b*9, 10, latent_vector_size)

        output = self.proj(attention)
        output = self.dropout(output)
        # projected_attention: (b*9, 10, latent_vector_size)

        if self.mode != "regular":
            cls, x = torch.tensor_split(output, [1], dim=1)
            # cls: (b*9, 1, latent_vector_size),
            # x: (b*9, 9, latent_vector_size)
            cls = einops.rearrange(cls, "(b p) () d -> b p () d", p=9)
            # cls: (b, 9, 1, latent_vector_size)
            cls = torch.mean(cls, dim=1)
            # cls: (b, 1, latent_vector_size)
            if self.mode == "block":
                x = einops.rearrange(
                    x,
                    "(b h_blocks w_blocks) (block_h block_w) d -> "
                    "b (h_blocks block_h w_blocks block_w) d",
                    h_blocks=3,
                    w_blocks=3,
                    block_h=3,
                    block_w=3,
                )
            elif self.mode == "row":
                x = einops.rearrange(x, "(b h) w d -> b (h w) d", h=9)
            else:
                # col
                x = einops.rearrange(x, "(b w) h d -> b (h w) d", w=9)

            # x: (b, 81, latent_vector_size)
            output = torch.cat([cls, x], dim=1)
            # output: (b, 82, latent_vector_size)

        return output


class MHABlock(nn.Module):
    def __init__(self, latent_vector_size, n_heads, dropout, mode="regular"):
        super().__init__()
        self.norm = nn.LayerNorm(latent_vector_size)
        self.mha = MultiHeadedAttention(
            latent_vector_size, n_heads, dropout, mode
        )

    def forward(self, x):
        skip = x
        x = self.norm(x)
        x = self.mha(x)
        return x + skip


class EncoderBlock(nn.Sequential):
    def __init__(
        self, latent_vector_size, n_heads, n_hidden, dropout, mode="regular"
    ):
        super().__init__(
            MHABlock(latent_vector_size, n_heads, dropout, mode),
            MLPBlock(latent_vector_size, n_hidden, dropout),
        )


class Sudoker(nn.Module):
    def __init__(self, depth, latent_vector_size, n_heads, n_hidden, dropout):
        super().__init__()
        assert depth % 3 == 0
        self.embedding = Embedding(latent_vector_size)
        self.encoder = nn.Sequential(*[
            nn.Sequential(
                EncoderBlock(
                    latent_vector_size, n_heads, n_hidden, dropout, mode="row"
                ),
                EncoderBlock(
                    latent_vector_size, n_heads, n_hidden, dropout, mode="col"
                ),
                EncoderBlock(
                    latent_vector_size,
                    n_heads,
                    n_hidden,
                    dropout,
                    mode="block",
                ),
            )
            for _ in range(depth // 3)
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
    pass
    # Below could be useful for writing tests!!

    # # 1, 81, 768 -> 9, 9, 768
    # x = torch.zeros((2, 1, 9, 9))
    # cls = torch.zeros(2, 1, 1)
    # for b in range(2):
    #     for i in range(9):
    #         for j in range(9):
    #             x[b, :, i, j] = b * 100 + i * 10 + j + 1
    #     cls[b, :, :] = b / 2
    #
    # x = einops.rearrange(x, 'b d h w -> b (h w) d')
    #
    # x = torch.cat([cls, x], dim=1)
    # cls, x = torch.tensor_split(x, [1], dim=1)
    # # print(einops.rearrange(x, 'b (h w) d -> (b h) w d', h = 4))
    #
    # x = einops.rearrange(x,
    #                      'b (h_blocks block_h w_blocks block_w) d
    #                      -> (b h_blocks w_blocks) (block_h block_w) d',
    #                      h_blocks=3, w_blocks=3, block_h=3, block_w=3)
    # cls = einops.repeat(cls, 'b () d -> (b p) () d', p=9)
    #
    # x = torch.cat([cls, x], dim=1)
    # print(x.shape)
    # cls, x = torch.tensor_split(x, [1], dim=1)
    # cls = einops.rearrange(cls, '(b p) () d -> b p () d', p=9)
    # print(cls)
    # cls = torch.mean(cls, dim=1)
    # print(cls.shape)
    #
    # x = einops.rearrange(x,
    #                      '(b h_blocks w_blocks) (block_h block_w) d ->
    #                      b (h_blocks block_h w_blocks block_w) d',
    #                      h_blocks=3, w_blocks=3, block_h=3, block_w=3)
    # x = torch.cat([cls, x], dim=1)
    # print(x)


if __name__ == "__main__":
    main()
