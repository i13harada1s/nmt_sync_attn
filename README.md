# Neural Machine Translation by Transformer with Synchronous Attention Constraint (Deguchi et. al., 2020)

## Introduction
This is an implementation of "[同期注意制約を与えた Transformer によるニューラル機械翻訳 (出口, 2020)](https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/A6-1.pdf)".


## Training and Inference:
### IWSLT'14 German to English
```bash
databin_dir=<path to iwslt14 binarized data>
model_path=<path to checkpoint>

# Train the model
fairseq-train ${databin_dir} \
    --user-dir examples/sync_attn/sync_attn_src \
    --task translation \
    --arch transformer_iwslt_de_en_sync \
    --criterion label_smoothed_cross_entropy_with_sync \
    --label-smoothing 0.1 --sync_lambda 1.0 \
    --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.001 \
    --save-dir ${model_path} --max-epoch 100 --keep-last-epochs 5 \
    --max-tokens 4096 --update-freq 1

# Make the avarege checkpoint
python scripts/average_checkpoints.py \
    --inputs ${model_path} \
    --num-epoch-checkpoints 5 \
    --output ${model_path}/checkpoint_avg.pt

# Evaluate the model
fairseq-generate ${databin_dir} \
    --user-dir examples/sync_attn/sync_attn_src \
    --path ${model_path}/checkpoint_avg.pt \
    --batch-size 128 --beam 4 --remove-bpe

# Transformer
# BLEU4 = 34.49, 68.5/42.7/28.7/19.7 (BP=0.962, ratio=0.963, syslen=126270, reflen=131161)
# w/. Synchronous Attention Constraint
# BLEU4 = 34.74, 68.8/43.0/29.0/19.9 (BP=0.961, ratio=0.961, syslen=126103, reflen=131161)
```

## Citation
```bibtex
@inproceedings{deguchi2020neural,
  title={同期注意制約を与えた Transformer によるニューラル機械翻訳},
  author={
    出口, 祥之 and 
    田村, 晃裕 and 
    二宮, 崇
  },
  booktitle={言語処理学会 第26回年次大会 発表論文集}
  url={https://www.anlp.jp/proceedings/annual_meeting/2020/pdf_dir/A6-1.pdf},
  year={2020}
}
```
