import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Reduce


class Embedding(nn.Module):
    def __init__(self,
                 user_num : int = 100,
                 item_num : int = 100, 
                 emb_size : int = 256, 
                 factor_num : int = 128):
        super(Embedding, self).__init__()

        self.embed_user = [nn.Embedding(user_num, factor_num) for _ in range(emb_size)]
        self.embed_item = [nn.Embedding(item_num, factor_num) for _ in range(emb_size)]

        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_size))
        self.positions = nn.Parameter(torch.randn(2 * factor_num + 1, emb_size))
    
    def forward(self, user, item):
        b, _ = user.size()

        embed_user = [layer(user) for layer in self.embed_user]
        embed_item = [layer(item) for layer in self.embed_item]

        embed_user = torch.cat(embed_user, dim = 1)
        embed_item = torch.cat(embed_item, dim = 1)

        embed_user = rearrange(embed_user, 'b n c -> b c n')
        embed_item = rearrange(embed_item, 'b n c -> b c n')

        x = torch.cat([embed_user, embed_item], dim = 1) # b, factor_num * 2, 2*emb_size

        # pretend the cls tokens to the input
        cls_tokens = repeat(self.cls_token, '() n e -> b n e', b=b)
        x = torch.cat([cls_tokens, x], dim=1) # b , factor_num + 1, 2 * emb_size

        # add position embedding
        x += self.positions

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 emb_size : int = 256,
                 num_heads : int = 8, 
                 dropout : float = 0.):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads

        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)
        
    def forward(self, x , mask = None):
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h = self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h = self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h = self.num_heads)

        # sum up over the last axis
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys) # batch, num_heads, query_len, key_len
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
            
        scaling = self.emb_size ** (1/2)
        att = F.softmax(energy, dim=-1) / scaling
        att = self.att_drop(att)

        # sum up over the third axis
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class FeedForwardBlock(nn.Sequential):
    def __init__(self, 
                 emb_size : int, 
                 expansion : int = 4, 
                 drop_p : float = 0.):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size)
        )


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self, 
                 emb_size : int = 256, 
                 drop_p : float = 0., 
                 forward_expansion : int = 4,
                 forward_drop_p : float = 0.,
                 ** kwargs):
        super().__init__(
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    MultiHeadAttention(emb_size, **kwargs),
                    nn.Dropout(drop_p)
                )
             ),
            ResidualAdd(
                nn.Sequential(
                    nn.LayerNorm(emb_size),
                    FeedForwardBlock(emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                    nn.Dropout(drop_p)
                )
            )
        )


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth : int = 12, **kwargs):
        super().__init__(*[TransformerEncoderBlock(**kwargs) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size : int = 256):
        super().__init__(
            Reduce('b n e -> b e', reduction = 'mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 1)
        )


class ViT(nn.Module):
    def __init__(self,user_num : int = 100,
                 item_num : int = 100,
                 emb_size : int = 256,
                 factor_num : int = 128,
                 depth : int = 12,
                 **kwargs):
        """ NCF Framework Using Transformer Structure

        Args:
            user_num (int, optional): number of users. Defaults to 100.
            item_num (int, optional): number of items. Defaults to 100.
            emb_size (int, optional): number of embedding layers. Defaults to 256.
            factor_num (int, optional): number of predictive factors. Defaults to 128.
            depth (int, optional): number of EncoderBlock. Defaults to 12.
        """
        super().__init__()
        self.emb = Embedding(user_num, item_num, emb_size, factor_num)
        self.enc = TransformerEncoder(depth, **kwargs)
        self.cls = ClassificationHead()
    
    def forward(self, user, item):
        x = self.emb(user, item)
        x = self.enc(x)
        x = self.cls(x)
        return x
