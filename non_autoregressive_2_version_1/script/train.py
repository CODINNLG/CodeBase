import transformers
import numpy as np
import os
import torch
import math
import datasets
import logging
import argparse
import random
import nltk
from torch.utils.tensorboard import SummaryWriter

from datasets import (
    load_dataset, 
    load_metric,
    Dataset
)
from transformers import(
    AutoTokenizer, 
    RobertaConfig,
    AdamW, 
    DataCollatorForSeq2Seq,
    get_scheduler,
    SchedulerType,
    set_seed,
    RobertaModel,
)
from src.utils import(
    _make_causal_mask,
    _mask_casual_sential_attention_mask,
    _add_special_tokens,
)
from transformers.file_utils import get_full_repo_name, is_offline_mode
from src.model import PlanNAT, RobertaEncoder, RobertaDecoder
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from filelock import FileLock
from accelerate import Accelerator
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a summarization task")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="The path of the dataset to use",
    )
    parser.add_argument(
        "--train_file", type=str, default=None, help="A csv or a json file containing the training data."
    )
    parser.add_argument(
        "--validation_file", type=str, default=None, help="A csv or a json file containing the validation data."
    )
    parser.add_argument(
        "--ignore_pad_token_for_loss",
        type=bool,
        default=True,
        help="Whether to ignore the tokens corresponding to " "padded labels in the loss computation or not.",
    )
    parser.add_argument(
        "--max_source_length",
        type=int,
        default=512,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--source_prefix",
        type=str,
        default=None,
        help="A prefix to add before every source text",
    )
    parser.add_argument(
        "--preprocessing_num_workers",
        type=int,
        default=None,
        help="The number of processes to use for the preprocessing.",
    )
    parser.add_argument(
        "--overwrite_cache", type=bool, default=None, help="Overwrite the cached training and evaluation sets"
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=512,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``."
        "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated,"
        " sequences shorter will be padded if `--pad_to_max_lengh` is passed."
    )
    parser.add_argument(
        "--num_beams",
        type=int,
        default=None,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )
    parser.add_argument(
        "--pad_to_max_length",
        action="store_true",
        help="If passed, pad all samples to `max_length`. Otherwise, dynamic padding is used.",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the full texts (for summarization).",
    )
    parser.add_argument(
        "--summary_column",
        type=str,
        default=None,
        help="The name of the column in the datasets containing the summaries (for summarization).",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", 
        type=float, 
        default=0.0, 
        help="Weight decay to use."
    )
    parser.add_argument(
        "--num_train_epochs", 
        type=int, 
        default=3, 
        help="Total number of training epochs to perform."
    )
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform. If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None, 
        help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None, 
        help="A seed for reproducible training."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default=None,
        help="Model type to use if training from scratch.",
    )
    parser.add_argument(
        "--sim_alpha", 
        type=float, 
        default=0.0, 
        help="Control the extent of the plan positions."
    )
    parser.add_argument(
        "--pos_alpha", 
        type=float, 
        default=0.0, 
        help="Control the extent of the generation positions."
    )
    args = parser.parse_args()

    # Sanity checks
    if args.dataset_dir is None and args.train_file is None and args.validation_file is None:
        raise ValueError("Need either a dataset name or a training/validation file.")
    else:
        if args.train_file is not None:
            extension = args.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if args.validation_file is not None:
            extension = args.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
    return args

def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

def load_weight(new_model, raw_model):
    '''
    loading the weight from raw_model to new_model
    '''
    new_model_dict, raw_model_dict = new_model.state_dict(), raw_model.state_dict()
    raw_model_dict = dict(raw_model_dict)
    for name, para in new_model_dict.items():
        if name in raw_model_dict:
            if para.shape == raw_model_dict[name].shape:
                para.copy_(raw_model_dict[name])
            else:
                print(f"load initial weight from {name:}")
                # adding util match
                if (para.dim()==2):
                    len_idx = torch.arange(0, raw_model_dict[name].size(0))[:, None]
                    dim_idx = torch.stack([torch.arange(raw_model_dict[name].size(-1))]*raw_model_dict[name].size(0), dim=0)
                    para[len_idx, dim_idx] = raw_model_dict[name]
                elif (para.dim()==1):
                    dim_idx = torch.arange(0, raw_model_dict[name].size(0))
                    para[dim_idx] = raw_model_dict[name]
    return new_model


def main():
    args = parse_args()
    # initialize accelerator
    accelerator = Accelerator()
    # initialize the logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, "tensorboard_logdir"))
    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)
    
    # initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    
    # add some special tokens to tokenizer
    # tokenizer = _add_special_tokens(tokenizer)
    special_token_lst = ["<PLN>", "<COR>", "<SEN>", "<GEN>"]
    num_special_tokens = tokenizer.add_tokens(special_token_lst)
    cor_token_id = tokenizer.convert_tokens_to_ids("<COR>")
    pln_token_id = tokenizer.convert_tokens_to_ids("<PLN>")
    sen_token_id = tokenizer.convert_tokens_to_ids("<SEN>")
    gen_token_id = tokenizer.convert_tokens_to_ids("<GEN>")
    mask_token = tokenizer.mask_token

    # set the model config
    config = RobertaConfig.from_pretrained(args.config_name)
    added_config = {
        "apply_long_position": True, 
        "padding_idx": tokenizer.pad_token_id, 
        "max_perdict_length": 60, 
        "vocab_size": tokenizer.vocab_size, 
    }
    # initialize the model
    enc_model = RobertaEncoder(config, added_config, tokenizer)
    dec_model = RobertaDecoder(config, added_config, tokenizer)
    pretrained_model = RobertaModel.from_pretrained(args.model_name_or_path)

    
    # config.add_cross_attention = True
    # config.is_decoder = True
    # config.vocab_size = len(tokenizer)
    # dec_model = RobertaForCausalLM(config=config)  # train from scratch
    # raw_model = RobertaForCausalLM.from_pretrained(args.model_name_or_path)
    # dec_model = load_weight(dec_model, raw_model)
    # resize the embedding length
    enc_model.resize_token_embeddings(len(tokenizer))
    dec_model.resize_token_embeddings(len(tokenizer))

    model = PlanNAT(
        config=config, 
        added_config=added_config,
        enc_model=enc_model, 
        dec_model=dec_model , 
        tokenizer=tokenizer, 
        args=args, 
        logger=logger
    )

    logger.info(model)
    total = sum(p.numel() for p in model.parameters())
    logger.info("Total params: %.2fM" % (total/1e6))

    # load datasets 
    raw_dataset_paths = {
        "train": os.path.join(args.dataset_dir, args.train_file), 
        "validation": os.path.join(args.dataset_dir, args.validation_file)
    }
    raw_datasets = load_dataset("json", data_files=raw_dataset_paths)

    # preprocess dataset
    column_names = raw_datasets["train"].column_names
    prompt_column = column_names[0]
    text_column = column_names[1]

    # Temporarily set max_target_length for training.
    max_target_length = args.max_target_length
    padding = "max_length" if args.pad_to_max_length else False

    def preprocess_function(examples):
        input = examples[prompt_column]
        raw_label = examples[text_column]
        
        model_inputs = tokenizer(
            input, 
            max_length=max_target_length,
            padding=padding,
            truncation=True,
            return_tensors="pt",
        )
        # Setup the tokenizer for targets  
        label = tokenizer(
            raw_label, 
            max_length=max_target_length, 
            padding=padding, 
            truncation=True,
            return_tensors="pt"
        )

        # split the src according to the <mask> token
        raw_src = model_inputs["input_ids"]
        # import pdb; pdb.set_trace()
        split_ids = torch.where(raw_src == tokenizer.mask_token_id)[1]
        if split_ids.size(-1) == 0:
            segment_src = [raw_src]
            src_segment_length = [raw_src.size(-1)]
        else:
            segment_src = []
            src_segment_length = []
            st = 0
            for i in split_ids:
                segment_src.append(raw_src[:, st: i])
                src_segment_length.append(i - st)
                st = i
            segment_src.append(raw_src[:, st: torch.sum(model_inputs["attention_mask"]==1)])
            src_segment_length.append(torch.sum(model_inputs["attention_mask"]==1) - st)
            segment_src = torch.cat(segment_src, dim=-1)

        # split the trg according to the <mask> token
        raw_label= label["input_ids"]
        split_ids = torch.where(raw_label == tokenizer.mask_token_id)[1]
        if split_ids.size(-1) == 0:
            segment_tgt = [raw_label]
            tgt_segment_length = [raw_label.size(-1)]
        else:
            segment_tgt = []
            tgt_segment_length = []
            st = 0
            for i in split_ids:
                segment_tgt.append(raw_label[:, st: i])
                tgt_segment_length.append(i - st)
                st = i
            segment_tgt.append(raw_label[:, st: torch.sum(label["attention_mask"]==1)])
            tgt_segment_length.append(torch.sum(label["attention_mask"]==1) - st)
            segment_tgt = torch.cat(segment_tgt, dim=-1)
            segment_tgt[:, tgt_segment_length] = tokenizer.bos_token_id

        # mask the tgt sentence according to the length (uniform masking)
        mask_segment_tgt = segment_tgt.clone()
        cur_idx = 0
        for seg_length in tgt_segment_length:
            mask_num = torch.randint(0, seg_length.data-2, (1, ))  # except for the first and last positions
            mask_pos = torch.tensor(random.sample(range(1, seg_length.data), mask_num.data))
            mask_segment_tgt[:, mask_pos + cur_idx] = tokenizer.mask_token_id
            cur_idx += seg_length.data
        assert cur_idx == torch.sum(label["attention_mask"]).data

        # adding padding 
        model_inputs['tgt_attention_mask'] = label['attention_mask']
        model_inputs['label'] = label
        model_inputs['segment_tgt'] = segment_tgt
        model_inputs['mask_segment_tgt'] = mask_segment_tgt
        model_inputs['src_segment_length'] = src_segment_length
        model_inputs['tgt_segment_length'] = tgt_segment_length


        # set the last token as <SEN> to preevent truncation and padding
        # label["input_ids"][sum(label["attention_mask"])-2] = sen_token_id 
        # find all <SEN> positions
        # sen_pos = []
        # for i, element in enumerate(label["input_ids"]):  
        #     if element == sen_token_id:
        #         sen_pos.append(i + 1)  # the input is shifted to right when training

        # specialize the plan text input according to the label length
        # plan_text_ipt = " ".join([mask_token] * sum(label["attention_mask"]))
        # with tokenizer.as_target_tokenizer():
        #     plan_text_ipt = tokenizer(
        #         plan_text_ipt, 
        #         max_length=max_target_length, 
        #         padding=padding, 
        #         truncation=True
        #     )
        # plan_text_ipt['input_ids'][0] = pln_token_id

        # specialize the plan text label according to the sen_pos
        # plan_text_label = " ".join([mask_token] * (sum(plan_text_ipt["attention_mask"]) - 2))
        # with tokenizer.as_target_tokenizer():
        #     plan_text_label = tokenizer(
        #         plan_text_label, 
        #         max_length=max_target_length, 
        #         padding=padding, 
        #         truncation=True
        #     )
        # for ii in sen_pos:
        #     plan_text_label["input_ids"][ii - 1] = sen_token_id
        # plan_text_label["input_ids"][0] = pln_token_id
        
        # # specialize the gen text input according to the sen_pos
        # gen_text_input = plan_text_label["input_ids"]
        # gen_text_input[0] = gen_token_id
        
        # # generate special token idx for attention mask of the decoder, return : list
        # pos_lst = _mask_casual_sential_attention_mask(sen_pos, max_target_length)
        # # If we are padding here, replace all tokenizer.pad_token_id in the text by -100 when we want to ignore
        # # padding in the loss.
        # if padding == "max_length" and args.ignore_pad_token_for_loss:
        #     label["input_ids"] = [l if l != tokenizer.pad_token_id else -100 for l in label["input_ids"]]
        #     plan_text_label["input_ids"] = [l if l != tokenizer.pad_token_id else -100 for l in plan_text_label["input_ids"]]
        # model_inputs["labels"] = label["input_ids"]
        # model_inputs["sn_pos"] = pos_lst
        # model_inputs["plan_text_ipt"] = plan_text_ipt["input_ids"]
        # model_inputs["plan_text_label"] = plan_text_label["input_ids"]
        # model_inputs["tgt_attention_mask"] = label['attention_mask']
        # model_inputs["gen_text_ipt"] = gen_text_input

        return model_inputs
        
    with accelerator.main_process_first():
        processed_datasets = raw_datasets.map(
            preprocess_function,
            batched=False,
            num_proc=args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

    train_dataset = processed_datasets["train"]
    eval_dataset = processed_datasets["validation"]
    import pdb; pdb.set_trace()

    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 1):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    
    
    
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if accelerator.use_fp16 else None,
    )

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    
    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )
        
    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # Metric
    metric = load_metric("rouge")

    # Train!
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    global_step = 0
    for epoch in range(args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            loss, lm_loss, pos_loss, sim_loss, output_text, ipt_text, label_text = model(batch)
            global_step += 1
            writer.add_scalar(tag="lm_loss", scalar_value=lm_loss, global_step=global_step)
            writer.add_scalar(tag="pos_loss", scalar_value=pos_loss, global_step=global_step)
            writer.add_scalar(tag="sim_loss", scalar_value=sim_loss, global_step=global_step)
            loss = loss / args.gradient_accumulation_steps
            accelerator.backward(loss)
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                logger.info(f"epoch: {epoch}")
                logger.info(f"loss: {loss.data:.4f}, lm_loss: {lm_loss.data:.4f}, pos_loss: {pos_loss.data:.4f}, sim_loss: {sim_loss.data:.4f}")
                for i in range(len(ipt_text)):
                    ipt, gen, golden = ipt_text[i], output_text[i], label_text[i]
                    logger.info(f"decoder input: {ipt:}\ngeneration results: {gen:}\nlabel text: {golden:}")

            if completed_steps >= args.max_train_steps:
                break
        
        # save model at every checkpoint
        # if epoch < args.num_train_epochs - 1:  
        #     accelerator.wait_for_everyone()
        #     unwrapped_model = accelerator.unwrap_model(model)
        #     save_out = os.path.join(args.output_dir, str(epoch))
        #     unwrapped_model.save_pretrained(save_out, save_function=accelerator.save)
        #     if accelerator.is_main_process:
        #         tokenizer.save_pretrained(save_out)
                

        # eval on evaluation data
        # model.eval()
        # args.val_max_target_length = args.max_target_length

        # gen_kwargs = {
        #     "max_length": args.val_max_target_length if args is not None else config.max_length,
        #     "num_beams": args.num_beams,
        # }
        # for step, batch in enumerate(eval_dataloader):
        #     with torch.no_grad():
        #         generated_tokens = accelerator.unwrap_model(model).generate(
        #             batch["input_ids"],
        #             attention_mask=batch["attention_mask"],
        #             **gen_kwargs,
        #         )

        #         generated_tokens = accelerator.pad_across_processes(
        #             generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        #         )
        #         labels = batch["labels"]
        #         if not args.pad_to_max_length:
        #             # If we did not pad to max length, we need to pad the labels too
        #             labels = accelerator.pad_across_processes(batch["labels"], dim=1, pad_index=tokenizer.pad_token_id)

        #         generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
        #         labels = accelerator.gather(labels).cpu().numpy()

        #         if args.ignore_pad_token_for_loss:
        #             # Replace -100 in the labels as we can't decode them.
        #             labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        #         if isinstance(generated_tokens, tuple):
        #             generated_tokens = generated_tokens[0]
        #         decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        #         decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        #         decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

        #         metric.add_batch(predictions=decoded_preds, references=decoded_labels)
        # result = metric.compute(use_stemmer=True)
        # # Extract a few results from ROUGE
        # result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # result = {k: round(v, 4) for k, v in result.items()}

        # logger.info(result)
        

   
if __name__ == "__main__":
    main()