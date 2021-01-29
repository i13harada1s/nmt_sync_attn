from typing import Dict, List, Optional

import torch
from torch import Tensor
from fairseq.models.transformer import TransformerEncoder

from .transformer_sync_encoder_layer import TransformerSyncEncoderLayer


class TransformerSyncEncoder(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """
    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        self.sync_encoder_self_attention_idx  = args.sync_encoder_self_attention_idx

    def build_encoder_layer(self, args):
        layer = TransformerSyncEncoderLayer(args)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def forward(
        self,
        src_tokens,
        src_lengths,
        return_all_hiddens: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            src_lengths (torch.LongTensor): lengths of each source sentence of
                shape `(batch)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
            return_all_attn (bool, optional): also return all of the
                intermediate layers' attention weights (default: False).
            token_embeddings (torch.Tensor, optional): precomputed embeddings
                default `None` will recompute embeddings
        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
                - **encoder_attn** (List[Tensor]): all intermediate
                  layers' attention weights of shape `(num_heads, batch, src_len, src_len)`.
                  Only populated if *return_all_attn* is True.
        """
        x, encoder_embedding = self.forward_embedding(src_tokens, token_embeddings)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        encoder_states = []
        encoder_attn = []

        # encoder layers
        for idx, layer in enumerate(self.layers):
            x, attn = layer(
                x, 
                encoder_padding_mask,
                need_attn=(idx in self.sync_encoder_self_attention_idx),
            )
            if return_all_hiddens:
                assert encoder_states is not None
                encoder_states.append(x)
            if attn is not None:
                encoder_attn.append(attn)
        
        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "encoder_attn": encoder_attn, # List[B x T x T]
            "src_tokens": [],
            "src_lengths": [],
        }