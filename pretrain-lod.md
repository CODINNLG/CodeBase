# Run1: Pretrain T5-base (220M Parameters) from scratch on pile (subset, 360B Tokens)
model=T5-base, batch_size=8\*16=128, block_size=512, GPU-Mem=19581MiB\*1+19581MiB\*7

# Run2: Finetune T5-base (220M Parameters) on CNN_DailyMail
