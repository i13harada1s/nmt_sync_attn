from typing import Optional

import torch
from fairseq.models import (
    register_model, 
    register_model_architecture,
)
from fairseq.models.transformer import (
    base_architecture, 
    transformer_iwslt_de_en,
    TransformerModel
)

from ..modules.transformer_sync_encoder import TransformerSyncEncoder
from ..modules.transformer_sync_decoder import TransformerSyncDecoder


@register_model("transformer_sync")
class TransformerSyncModel(TransformerModel):
    """
    See "同期注意制約を与えた Transformer によるニューラル機械翻訳" 
    (deguchi et al., 言語処理学会 第26回年次大会, 2020).
    """
    def __init__(self, encoder, decoder, args):
        super().__init__(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        # fmt: off
        TransformerModel.add_args(parser)
        parser.add_argument('--sync-encoder-self-attention-idx', nargs='+', type=int,
                            help='Indexes of encoder layers to synchronize the attention.')
        parser.add_argument('--sync-decoder-self-attention-idx', nargs='+', type=int,
                            help='Indexes of decoder layers to synchronize the attention.')
        parser.add_argument('--sync-cross-attention-idx', nargs='+', type=int,
                            help='Indexes of decoder layers to synchronize the attention.')
        # fmt: on

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        return TransformerSyncEncoder(
            args, 
            src_dict, 
            embed_tokens
        )

    @classmethod
    def build_decoder(cls, args, tgt_dict, embed_tokens):
        return TransformerSyncDecoder(
            args,
            tgt_dict,
            embed_tokens,
            no_encoder_attn=getattr(args, "no_cross_attention", False),
        )

    @classmethod
    def build_model(cls, args, task):
        transformer_sync(args) # set any default arguments

        def _check_element_range(idx, low, high):
            return all((low <= i < high) for i in idx)
           
        # check the model argments 
        if not (
            len(args.sync_encoder_self_attention_idx) == \
            len(args.sync_decoder_self_attention_idx) == \
            len(args.sync_cross_attention_idx)
        ):
           raise ValueError(
               "indexs for syncronization requires the same length"
           )
        
        if not _check_element_range(args.sync_encoder_self_attention_idx, low=0, high=args.encoder_layers):
            raise ValueError(
                "--sync-encoder-self-attention-idx requires positive values less than --encoder-layers"
            )
        if not _check_element_range(args.sync_decoder_self_attention_idx, low=0, high=args.decoder_layers):
            raise ValueError(
                "--sync-decoder-self-attention-idx requires positive values less than --decoder-layers"
            )
        if not _check_element_range(args.sync_cross_attention_idx, low=0, high=args.decoder_layers):
            raise ValueError(
                "--sync-cross-attention-idx requires positive values less than --decoder-layers"
            )
        
        transformer_model = TransformerModel.build_model(args, task)
        encoder, decoder = transformer_model.encoder, transformer_model.decoder
        encoder = cls.build_encoder(args, encoder.dictionary, encoder.embed_tokens)
        decoder = cls.build_decoder(args, decoder.dictionary, decoder.embed_tokens)
        return TransformerSyncModel(encoder, decoder, args)

    def forward(
        self, 
        src_tokens, 
        src_lengths, 
        prev_output_tokens,
        return_all_hiddens: bool = True,
        return_all_attn: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(
            src_tokens, 
            src_lengths=src_lengths, 
            return_all_hiddens=return_all_hiddens,
        )
        decoder_out, decoder_extra = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        
        # concatenate a sequence of tensors
        encoder_attn  = torch.stack(encoder_out["encoder_attn"], dim=0)
        decoder_attn  = torch.stack(decoder_extra["features"]["decoder_attn"], dim=0)
        cross_qk_attn = torch.stack(decoder_extra["features"]["cross_qk_attn"], dim=0)
        cross_kq_attn = torch.stack(decoder_extra["features"]["cross_kq_attn"], dim=0)
        
        # (N x B x D x E) x (N x B x E x E) -> (N x B x D x E)
        project_attn = torch.einsum('abcd,abdd->abcd', cross_qk_attn, encoder_attn)
        # (N x B x D x E) x (N x B x E x D) -> (N x B x D x D)
        project_attn = torch.einsum('abcd,abde->abce', project_attn, cross_kq_attn)
    
        # mask the future tokens
        project_attn = torch.tril(project_attn)
 
        return decoder_out, (project_attn, decoder_attn)


@register_model_architecture("transformer_sync", "transformer_sync")
def transformer_sync(args):
    args.sync_encoder_self_attention_idx  = getattr(args, "sync_encoder_self_attention_idx", [0,1,2,3,4,5])
    args.sync_decoder_self_attention_idx  = getattr(args, "sync_decoder_self_attention_idx", [0,1,2,3,4,5])
    args.sync_cross_attention_idx = getattr(args, "sync_cross_attention_idx", [0,1,2,3,4,5])
    base_architecture(args)

@register_model_architecture("transformer_sync", "transformer_iwslt_de_en_sync")
def transformer_sync_iwslt_de_en(args):
    args.sync_encoder_self_attention_idx  = getattr(args, "sync_encoder_self_attention_idx", [5])
    args.sync_decoder_self_attention_idx  = getattr(args, "sync_decoder_self_attention_idx", [5])
    args.sync_cross_attention_idx = getattr(args, "sync_cross_attention_idx", [4])
    transformer_iwslt_de_en(args)

@register_model_architecture("transformer_sync", "transformer_wmt14_de_en_sync")
def transformer_sync_iwslt_de_en(args):
    args.sync_encoder_self_attention_idx  = getattr(args, "sync_encoder_self_attention_idx", [5])
    args.sync_decoder_self_attention_idx  = getattr(args, "sync_decoder_self_attention_idx", [5])
    args.sync_cross_attention_idx = getattr(args, "sync_cross_attention_idx", [4])
    base_architecture(args)