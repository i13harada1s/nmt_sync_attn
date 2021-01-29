from typing import Dict, List, Optional

import torch
from torch import Tensor
from fairseq.models.transformer import TransformerDecoder

from .transformer_sync_decoder_layer import TransformerSyncDecoderLayer


class TransformerSyncDecoder(TransformerDecoder):
    """
    Transformer decoder consisting of *args.decoder_layers* layers. Each layer
    is a :class:`TransformerDecoderLayer`.
    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): decoding dictionary
        embed_tokens (torch.nn.Embedding): output embedding
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(self, args, dictionary, embed_tokens, no_encoder_attn=False):
        super().__init__(args, dictionary, embed_tokens, no_encoder_attn=False)
        self.sync_decoder_self_attention_idx  = args.sync_decoder_self_attention_idx
        self.sync_cross_attention_idx = args.sync_cross_attention_idx

    def build_decoder_layer(self, args, no_encoder_attn=False):
        layer = TransformerSyncDecoderLayer(args, no_encoder_attn)
        if getattr(args, "checkpoint_activations", False):
            layer = checkpoint_wrapper(layer)
        return layer

    def extract_features_scriptable(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]],
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        """
        Similar to *forward* but only return features.
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).
        Args:
            full_context_alignment (bool, optional): don't apply
                auto-regressive mask to self-attention (default: False).
            alignment_layer (int, optional): return mean alignment over
                heads at this layer (default: last layer).
            alignment_heads (int, optional): only average alignment over
                this many heads (default: all heads).
        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - a dictionary with any model-specific outputs
        """
        if alignment_layer is None:
            alignment_layer = self.num_layers - 1

        # embed positions
        positions = (
            self.embed_positions(
                prev_output_tokens, incremental_state=incremental_state
            )
            if self.embed_positions is not None
            else None
        )

        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]
            if positions is not None:
                positions = positions[:, -1:]

        # embed tokens and positions
        x = self.embed_scale * self.embed_tokens(prev_output_tokens)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.project_in_dim is not None:
            x = self.project_in_dim(x)

        if positions is not None:
            x += positions

        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        self_attn_padding_mask: Optional[Tensor] = None
        if self.cross_self_attention or prev_output_tokens.eq(self.padding_idx).any():
            self_attn_padding_mask = prev_output_tokens.eq(self.padding_idx)

        self_attn: List[Optional[Tensor]] = []
        cross_qk_attn: List[Optional[Tensor]] = []
        cross_kq_attn: List[Optional[Tensor]] = []

        # decoder layers
        attn: Optional[Tensor] = None
        inner_states: List[Optional[Tensor]] = [x]
        for idx, layer in enumerate(self.layers):
            if incremental_state is None and not full_context_alignment:
                self_attn_mask = self.buffered_future_mask(x)
            else:
                self_attn_mask = None
                
            need_attn = (
                bool((idx in self.sync_decoder_self_attention_idx)) or \
                bool((idx in self.sync_cross_attention_idx)) or \
                bool((idx == alignment_layer))
            )

            x, layer_attn, _ = layer(
                x,
                encoder_out["encoder_out"][0]
                if (encoder_out is not None and len(encoder_out["encoder_out"]) > 0)
                else None,
                encoder_out["encoder_padding_mask"][0]
                if (
                    encoder_out is not None
                    and len(encoder_out["encoder_padding_mask"]) > 0
                )
                else None,
                incremental_state,
                self_attn_mask=self_attn_mask,
                self_attn_padding_mask=self_attn_padding_mask,
                need_attn=need_attn,
                need_head_weights=need_attn,
            )
            inner_states.append(x)
            if layer_attn is not None:
                if layer_attn["cross_qk_attn"] is not None:
                    if idx == alignment_layer:
                        attn = layer_attn["cross_qk_attn"].float().to(x)
                    if idx in self.sync_cross_attention_idx:
                        cross_qk_attn.append(layer_attn["cross_qk_attn"].mean(dim=0)) 
                if layer_attn["cross_kq_attn"] is not None:
                    if idx in self.sync_cross_attention_idx:
                        cross_kq_attn.append(layer_attn["cross_kq_attn"].mean(dim=0))
                if layer_attn["self_attn"] is not None:
                    if idx in self.sync_decoder_self_attention_idx:
                        self_attn.append(layer_attn["self_attn"]) # NOTE: need_head=False in decoder_layer
        
        features = {
            "decoder_attn": self_attn,
            "cross_qk_attn": cross_qk_attn,
            "cross_kq_attn": cross_kq_attn,
        }

        if attn is not None:
            if alignment_heads is not None:
                attn = attn[:alignment_heads]
            # average probabilities over heads
            attn = attn.mean(dim=0)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        if self.project_out_dim is not None:
            x = self.project_out_dim(x)

        return x, {"attn": [attn], "inner_states": inner_states, "features": features}