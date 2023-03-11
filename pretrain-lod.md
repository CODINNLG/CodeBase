# Run1: Pretrain T5-base (220M Parameters) from scratch on pile
model=T5-base, batch_size=8\*16=128, block_size=512, GPU-Mem=19581MiB\*1+19581MiB\*7

T5 details: 
- Pre-train each model for $2^{19}$ = **524,288** steps.
- A maximum sequence length of 512 and a batch size of 128 sequences. In total, this batch size and number of steps corresponds to pre-training on $2^{35}$ $\approx$ 34B tokens
- Note that $2^{35}$ tokens only covers a fraction of the entire C4 data set, so we never repeat any data during pre-training.

Time efficiency for training T5-base (220M Parameters) in 34B Tokens in 8\* A100:
- 1.5s/step, $\text{Time consumed} = \approx 1.5 * 2^{19} = 786432s \approx 10 \text{days}$

# Run2: Finetune T5-base (220M Parameters) on CNN_DailyMail
