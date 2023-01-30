import pickle
import sys, os
from dataclasses import dataclass
import numpy as np
import torch
import random
from transformers import(
    BertModel, 
    BertConfig,
    BertTokenizer,
    AdamW, 
    get_linear_schedule_with_warmup,
    PreTrainedTokenizerBase,
)
from tqdm.auto import tqdm
import logging
import argparse
from datasets import Dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from mask_pytorch import Mask, PinyinConfusionSet, StrokeConfusionSet 

logger = logging.getLogger(__name__)

@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        model ([`PreTrainedModel`]):
            The model that is being trained. If set and has the *prepare_decoder_input_ids_from_labels*, use it to
            prepare the *decoder_input_ids*
            This is useful when using *label_smoothing* to avoid calculating loss twice.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[Any] = None
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=return_tensors,
        )

        # prepare decoder_input_ids
        if (
            labels is not None
            and self.model is not None
            and hasattr(self.model, "prepare_decoder_input_ids_from_labels")
        ):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids

        return features


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def evaluate(FLAGS, sess, model, data_processor, label_list=None):
    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    out_dir = FLAGS.output_dir
    batch_size = 20
    EPOCH = FLAGS.epoch
    init_bert_dir = FLAGS.init_bert_path
    learning_rate = FLAGS.learning_rate
    vocab_file = '%s/vocab.txt' % init_bert_dir
    init_checkpoint = '%s/bert_model.ckpt' % init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir
 

    test_num = data_processor.num_examples
    test_data = data_processor.build_data_generator(batch_size)
    iterator = test_data.make_one_shot_iterator()
    input_ids, input_mask, segment_ids, lmask, label_ids, masked_sample = iterator.get_next()
    pred_loss, pred_result, gold_result, gold_mask, r_loss = model.create_model(input_ids, input_mask, segment_ids, lmask, label_ids, batch_size=batch_size, masked_sample=masked_sample, is_training=False)

    label_list = data_processor.label_list
    ans_c, ans_py, ans = [], [], []
    all_inputs, all_golds, all_preds = [], [], []
    all_fusino_preds = []
    all_inputs_sent, all_golds_sent, all_preds_sent = [], [], []
    for step in range(test_num // batch_size):
        inputs, loss_value, preds, golds, gmask = sess.run([input_ids, pred_loss, pred_result, gold_result, gold_mask])
        for k in range(batch_size):
            tmp1, tmp2, tmp3, tmps4, tmps5, tmps6, tmps7 = [], [], [], [], [], [], []
            for j in range(max_sen_len):
                if gmask[k][j] == 0: continue
                all_golds.append(golds[k][j])
                all_preds.append(preds[k][j])
                all_inputs.append(inputs[k][j])
                tmp1.append(label_list[golds[k][j]])
                tmp2.append(label_list[preds[k][j]])
                tmp3.append(label_list[inputs[k][j]])
                
            all_golds_sent.append(tmp1)
            all_preds_sent.append(tmp2)
            all_inputs_sent.append(tmp3)
                
    all_golds = [label_list[k] for k in all_golds]
    all_preds = [label_list[k] for k in all_preds]
    all_inputs = [label_list[k] for k in all_inputs]
   
    print('zi result:')
    p, r, f = score_f((all_inputs, all_golds, all_preds), only_check=False)

    return f


def train(args):
    gpuid = FLAGS.gpuid
    max_sen_len = FLAGS.max_sen_len
    train_path = FLAGS.train_path
    test_file = FLAGS.test_path
    test_file_merr = FLAGS.test_path_merr
    out_dir = FLAGS.output_dir
    batch_size = FLAGS.batch_size
    EPOCH = FLAGS.epoch
    alpha = FLAGS.alpha
    init_bert_dir = FLAGS.init_bert_path
    learning_rate = FLAGS.learning_rate
    vocab_file = '%s/vocab.txt' % init_bert_dir
    init_checkpoint = '%s/bert_model.ckpt' % init_bert_dir
    bert_config_path = '%s/bert_config.json'% init_bert_dir
 
    if os.path.exists(out_dir) is False:
        os.mkdir(out_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = gpuid
    keep_prob = FLAGS.keep_prob
    print('test_file=', test_file)
    test_data_processor = DataProcessor(test_file, max_sen_len, vocab_file, out_dir, label_list=None, is_training=False)
    print('test_file_merr=', test_file_merr)
    test_data_processor_merr = DataProcessor(test_file_merr, max_sen_len, vocab_file, out_dir, label_list=None, is_training=False)
    print('train_file=', train_path)
    data_processor = DataProcessor(train_path, max_sen_len, vocab_file, out_dir, label_list=None, is_training=True)

    train_num = data_processor.num_examples
    train_data = data_processor.build_data_generator(batch_size)
    # iterator = tf.compat.v1.data.make_one_shot_iterator(train_data)
    iterator = iter(train_data)

    input_ids, input_mask, segment_ids, lmask, label_ids, masked_sample = iterator.get_next()

    model = BertTagging(bert_config_path, num_class=len(data_processor.get_label_list()), max_sen_len=max_sen_len, alpha=alpha, keep_prob=keep_prob)
    (loss, probs, golds, mask, r_loss) = model.create_model(input_ids, input_mask, segment_ids, lmask, label_ids, batch_size=batch_size, masked_sample=masked_sample, is_training=True)

    tf_config = tf.ConfigProto(log_device_placement=False)
    tf_config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=tf_config) as sess:
        if init_checkpoint is not None:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            keys = [x for x in assignment_map.keys()]
            for key in keys:
                print(key, '\t', assignment_map[key])
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        num_steps = train_num // batch_size * EPOCH
        num_warmup_steps = num_steps // 10
        train_op = optimization.create_optimizer(loss, learning_rate, num_steps, num_warmup_steps, use_tpu=False)
        init = tf.global_variables_initializer()
        sess.run(init)

        loss_values = []
        r_loss_values = []
        saver = tf.train.Saver()
        best_score = 0.0
        best_model_path = os.path.join(out_dir, 'best.ckpt')
        total_step = 0
        for epoch in range(EPOCH):
            for step in range(int(train_num / batch_size)):
                total_step += 1
                start_time = time.time()
                train_loss, _, train_r_loss = sess.run([loss,  train_op, r_loss]) 
                loss_values.append(train_loss)
                r_loss_values.append(train_r_loss)
                if step % 500 == 0:
                    duration = time.time() - start_time
                    examples_per_sec = float(duration) / batch_size
                    format_str = ('Epoch {} step {},  train loss = {:.4f},{:.4f},{:.4f} ( {:.4f} examples/sec; {:.4f} ''sec/batch)')
                    print (format_str.format(epoch, step, np.mean(loss_values),np.mean(loss_values[-500:]),np.mean(r_loss_values[-500:]), examples_per_sec, duration))
                    loss_values = loss_values[-500:]
                    r_loss_values = r_loss_values[-500:]
                    print('multi-error result:')
                    evaluate(FLAGS, sess, model, test_data_processor_merr)
                    print('overall result:')
                    f1 = evaluate(FLAGS, sess, model, test_data_processor)
                    if f1 > best_score:
                        saver.save(sess, best_model_path)
                        best_score = f1
                    sys.stdout.flush()
            f1 = evaluate(FLAGS, sess, model, test_data_processor)
            if f1 > best_score:
                saver.save(sess, best_model_path)
                best_score = f1
            sys.stdout.flush()
        print ('best f value:', best_score)


def load_dataset(args, file_name):
    file_path = os.path.join(args.data_dir, file_name)
    dataset = pickle.load(open(file_path, 'rb'))
    res = {}  # convert to dict
    res["text"] = []
    for item in dataset:
        res["text"].append([item['src'] + "|||" + item['tgt']])
    res = Dataset.from_dict(res)
    return res

class CRAModel(torch.nn.Module):
    def __init__(self, args, Backbone_model, config, tokenizer) -> None:
        super().__init__()
        self.args = args
        self.back_model = Backbone_model
        # freeze the last pooler module
        for n, p in self.back_model.named_parameters():
            if "pooler" in n:
                p.requires_grad=False

        self.config = config
        self.tokenizer = tokenizer
        self.convert_to_dict = torch.nn.Linear(config.hidden_size, config.vocab_size)
        self.copy_block = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_size, config.hidden_size//2),
            torch.nn.Linear(config.hidden_size//2, 1),
            torch.nn.Sigmoid()
        )
        self.alpha = args.alpha


    def valid_step(self, input_ids, token_type_ids, attention_mask, labels):
        with torch.no_grad():
            logits = self.back_model(
                input_ids, 
                token_type_ids=token_type_ids, 
                attention_mask=attention_mask
            )
        last_hidden_state = logits.last_hidden_state
        probs = torch.nn.functional.softmax(self.convert_to_dict(last_hidden_state), dim=-1)
        predict_results = torch.argmax(probs, dim=-1)
        loss = self.focalLoss(probs, labels, reduce='mean')
        return loss, predict_results


    def forward(self, input_ids, noise_input_ids, token_type_ids, attention_mask, labels):
        noise_mask = torch.eq(input_ids, noise_input_ids)  # [bsz, seq_len]
        # print(torch.sum(noise_mask))
        ns_mask = noise_mask * attention_mask  # place where to ignore
        ns_mask = noise_mask.view(-1)  # [bsz*seq_len]
        # print(ns_mask)
        logits = self.back_model(
            input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        last_hidden_state = logits.last_hidden_state
        generative_probs = torch.nn.functional.softmax(self.convert_to_dict(last_hidden_state), dim=-1)

        logits_noise = self.back_model(
            noise_input_ids, 
            token_type_ids=token_type_ids, 
            attention_mask=attention_mask
        )
        noise_last_hidden_state = logits_noise.last_hidden_state
        noise_probs = torch.nn.functional.softmax(self.convert_to_dict(noise_last_hidden_state), dim=-1)

        copy_prob = self.copy_block(last_hidden_state)
        one_hot_labels = torch.nn.functional.one_hot(input_ids, num_classes=self.config.vocab_size)
        cp_dis = copy_prob * one_hot_labels + (1 - copy_prob) * generative_probs
        # cp_dis = self.clip_by_tensor(cp_dis, 1e-10, 1.0-1e-7)  # original paper's code
        ce_loss = self.focalLoss(cp_dis, labels)
        # print(ce_loss)
        # print(torch.sum(ce_loss))
        # calculate ce loss only when two positions are insame
        ce_loss = self.alpha * ce_loss * (~ns_mask) + (1 - self.alpha) * ce_loss * ns_mask
        cp_per_example_loss = torch.sum(ce_loss) / torch.sum(attention_mask)
        # print("+++" * 20)
        # print(ce_loss)
        # print("***" * 20)
        # print(cp_per_example_loss)
        # print("---" * 20)

        raw_kl_loss = self.cal_kl(generative_probs, noise_probs)
        sum_kl_loss = torch.sum(raw_kl_loss, dim=-1)
        sum_kl_loss = sum_kl_loss.view(-1) # [bsz * seq_len]
        # calculate kl loss only when two positions are same
        kl_loss = torch.sum(sum_kl_loss * ns_mask * self.alpha) / torch.sum(ns_mask)
        # print(kl_loss)
        # print(torch.sum(kl_loss))
        # exit()

        final_loss = cp_per_example_loss + kl_loss

        predict_results = torch.argmax(cp_dis, dim=-1)
        # print(predict_results)
        # print(labels)
        # exit()
        return final_loss, ce_loss, kl_loss, predict_results


    def clip_by_tensor(self, t, t_min, t_max):
        t=t.float()
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def focalLoss(self, output, labels, reduce='none'):
        """
        Input:
            output: [bsz, seq_len, vocab_size]
            labels: [bsz, seq_len]
        Return:
            loss: [batch_size]
        """
        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction=reduce)
        output = output.view(-1, output.shape[-1])
        labels = labels.view(-1).to(torch.long)

        loss = loss_fct(output, labels)
        return loss


    def cal_kl(self, probs_1, probs_2, reduce='none'):
        kl_loss1 = torch.nn.functional.kl_div(probs_1.log(), probs_2, reduction=reduce)
        kl_loss2 = torch.nn.functional.kl_div(probs_2.log(), probs_1, reduction=reduce)
        return (kl_loss1 + kl_loss2)/2
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_name_or_path", default="", type=str,
                        help="Path to pre-trained model or shortcut name selected in the list: ")
    parser.add_argument("--data_dir", default="", type=str,
                        help="The path of the data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_file", default="", type=str,
                        help="name of the training file")
    parser.add_argument("--dev_file", default="", type=str,
                        help="The name of the evaluation file")
    parser.add_argument("--test_file", default="", type=str,
                        help="The name of the testing file")                    
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--label_list", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_sen_len", default=64, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true')
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")     
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=10, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument('--logging_steps', type=int, default=100,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--keep_prob', type=float, default=0.9, help='keep prob in dropout')
    parser.add_argument('--alpha', type=float, default=0.05, help='trade-off factor')
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    args = parser.parse_args()

    # initialize the accelerator
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    
    # initialize the parameters
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    set_seed(args)

    config = BertConfig.from_pretrained(args.model_name_or_path)
    bertmodel = BertModel.from_pretrained(args.model_name_or_path)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    model = CRAModel(args, bertmodel, config, tokenizer)

    # define the noise injector: masker
    same_py_file = '../datas/confusions/same_pinyin.txt'
    simi_py_file = '../datas/confusions/simi_pinyin.txt' 
    stroke_file = '../datas/confusions/same_stroke.txt'   
    pinyin = PinyinConfusionSet(tokenizer, same_py_file)
    jinyin = PinyinConfusionSet(tokenizer, simi_py_file)
    stroke = StrokeConfusionSet(tokenizer, stroke_file)  
    masker = Mask(same_py_confusion=pinyin, simi_py_confusion=jinyin, sk_confusion=stroke)


    def tokenize_fct(examples): # inject noise
        items = list(examples.values())[0]
        srcs = [item[0].split("|||")[0] for item in items]
        tgts = [item[0].split("|||")[1] for item in items]
        
        model_inputs = tokenizer(
            srcs,
            padding='max_length',
            return_tensors='np',  
            truncation=True,
            max_length=args.max_sen_len,
        )
        labels = tokenizer(
            tgts,
            return_tensors='np',  
            padding='max_length',
            truncation=True,
            max_length=args.max_sen_len,
        )
        noise_inputs = [masker.mask_process_rand(model_inputs['input_ids'][i], labels['input_ids'][i]) for i in range(len(model_inputs['input_ids']))]
        labels["input_ids"] = [
            [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
        ]    
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["noise_input_ids"] = noise_inputs
        return model_inputs

    logger.info(" | loading training dataset from {:} ...".format(args.data_dir))

    train_dataset = load_dataset(args, args.train_file)
    with accelerator.main_process_first():
        train_dataset = train_dataset.map(
            tokenize_fct, 
            batched=True,
            batch_size=args.per_gpu_train_batch_size, 
            num_proc=50,
            remove_columns=["text"]
        )
    logger.info(" | loading dev dataset from {:} ...".format(args.data_dir))
    dev_dataset = load_dataset(args, args.dev_file)
    with accelerator.main_process_first():
        dev_dataset = dev_dataset.map(
            tokenize_fct, 
            batched=True,
            batch_size=args.per_gpu_eval_batch_size, 
            num_proc=10,
        )
    logger.info(" | loading testing dataset from {:} ...".format(args.data_dir))
    test_dataset = load_dataset(args, args.test_file)
    with accelerator.main_process_first():
        test_dataset = test_dataset.map(
            tokenize_fct, 
            batched=True,
            batch_size=args.per_gpu_eval_batch_size, 
            num_proc=10,
        )
    logger.info("dataset set statics:")
    logger.info(train_dataset)
    logger.info(dev_dataset)
    logger.info(test_dataset)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=args.max_sen_len,
        return_tensors='pt'
    )

    train_dataloader = DataLoader(train_dataset, batch_size=args.per_gpu_train_batch_size, collate_fn=data_collator)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.per_gpu_eval_batch_size, collate_fn=data_collator)
    test_dataloader = DataLoader(test_dataset, batch_size=args.per_gpu_eval_batch_size, collate_fn=data_collator)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(args.train_dataset) // args.per_gpu_train_batch_size // args.gradient_accumulation_steps + 1
    else:
        t_total = len(train_dataset) // args.per_gpu_train_batch_size // args.gradient_accumulation_steps * args.num_train_epochs
    
    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    need_optimized_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in need_optimized_parameters if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in need_optimized_parameters if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    model, optimizer, training_dataloader, dev_dataloader, test_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, dev_dataloader, test_dataloader
    )
    ## train the model !
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset) * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.per_gpu_train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    set_seed(args)
    progress_bar = tqdm(total=t_total)
    to_step = 0
    for _ in range(args.num_train_epochs):
        logger.info(f"begin training at step {to_step}")
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch.to(accelerator.device)
            total_loss, ce_loss, kl_loss, predict_results = model(
                input_ids=batch["input_ids"], 
                noise_input_ids=batch["noise_input_ids"],
                token_type_ids=batch['token_type_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            # for name, param in model.named_parameters():
            #     if param.grad is None:
            #         print(name)

            total_loss = total_loss / args.gradient_accumulation_steps
            accelerator.backward(total_loss)
            to_step += 1
            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.update(args.gradient_accumulation_steps)
            
            if to_step % args.logging_steps == 0:
                logger.info("ce loss is {:.4f}".format(total_loss.data))
                logger.info(f"begin evaluation at step {to_step}")
                model.eval()
                losses, predicts = [], []
                for eval_step, batch in enumerate(dev_dataloader):
                    loss, result = model.valid_step(
                        batch['input_ids'], 
                        batch['token_type_ids'], 
                        batch['attention_mask'], 
                        batch['labels']
                    )
                    losses.append(loss)
                    predicts.append(result)

                losses = torch.mean(torch.cat(losses))
                predicts = torch.cat(predicts)
                logger.info("the loss of the valid dataset is {:.4f}".format(losses))
                # metric of gec: token level and sentence level
        
        # after each epoch, saving the checkpoint
        accelerator.wait_for_everyone()
        save_path = os.path.join(args.output_dir, "saved_ckp-{}".format(str(to_step)))
        logger.info(f"begin saving checkpoint at step {to_step} in path {save_path}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_path, save_function=accelerator.save)
        if accelerator.is_main_process:
            tokenizer.save_pretrained(save_path)



            # evaluate
        
        # save ckp + tokenizer

            


   
    
