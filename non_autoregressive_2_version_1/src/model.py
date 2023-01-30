import logging
import math
import os
import sys
from typing import List, Optional, Tuple, Union
from turtle import forward
import torch.nn.functional as F
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import copy
from transformers import (
    AutoTokenizer, 
    AutoModel,
    PreTrainedModel,
    RobertaForCausalLM,
    RobertaPreTrainedModel,
)
from transformers.models.roberta.modeling_roberta import (
    RobertaEmbeddings,
    RobertaAttention,
    RobertaIntermediate,
    RobertaOutput,
    RobertaLMHead
)
from .discriminator import Max_Discriminator
import numpy as np
from src.utils import _make_causal_mask
from transformers.modeling_utils import apply_chunking_to_forward

class PlanNAT(PreTrainedModel):
    def __init__(self, config, added_config,  enc_model, dec_model, tokenizer, args, logger):
        super(PlanNAT, self).__init__(config)
        config = config.update(added_config)
        self.config = config
        self.logger = logger
        self.encoder = enc_model
        self.decoder = dec_model
        self.tokenizer = tokenizer
        self.args = args
        self.load_pretrained_weight
        self.share_enc_dec_param() # share all the parameters except for the positional/embedding

        # self.dropout = nn.Dropout(dec_config.hidden_dropout_prob)
        # self.vocab_size = tokenizer.vocab_size
        # for encoder, we recommand sentence-bert
        # self.mealPooling_module = nn.AvgPool1d((enc_config.max_position_embeddings, 1))  # utilize pooling out instead of mean pooling
        # self.bos_token_id = tokenizer.bos_token_id
        # self.eos_token_id = tokenizer.eos_token_id
        # self.mask_token_ids = tokenizer.mask_token_id
        # self.cor_token_id = tokenizer.convert_tokens_to_ids("<COR>")
        # self.pln_token_id = tokenizer.convert_tokens_to_ids("<PLN>")
        # self.gen_token_id = tokenizer.convert_tokens_to_ids("<GEN>")
        # self.sen_token_id = tokenizer.convert_tokens_to_ids("<SEN>")
        # self.ce_loss_fct = CrossEntropyLoss()
        # self.max_discriminator = Max_Discriminator(dec_config.hidden_size, initrange=0.1)
        # self.init_weights()
        # self.args = args
        # self.sim_alpha = args.sim_alpha
        # self.pos_alpha = args.pos_alpha
    def load_pretrained_weight(self, empty_model, pretrained_model):
        # load pretrained model's weight to encoder/decoder
        for empty_layer, pre_layer in zip(empty_model, pretrained_model):
            empty_layer.attention.self.query.weight = pre_layer.attention.self.query.weight
            empty_layer.attention.self.query.bias = pre_layer.attention.self.query.bias
            empty_layer.attention.self.value.weight = pre_layer.attention.self.value.weight
            empty_layer.attention.self.value.bias = pre_layer.attention.self.value.bias
            empty_layer.attention.self.query.weight = pre_layer.attention.self.query.weight
            empty_layer.attention.self.query.bias = pre_layer.attention.self.query.bias
            empty_layer.attention.output.dense.weight = pre_layer.attention.output.dense.weight
            empty_layer.attention.output.dense.bias = pre_layer.attention.output.dense.bias 
            empty_layer.attention.output.LayerNorm.weight = pre_layer.attention.output.LayerNorm.weight
            empty_layer.attention.output.LayerNorm.bias = pre_layer.attention.output.LayerNorm.bias
            empty_layer.intermediate.dense.weight = pre_layer.intermediate.dense.weight
            empty_layer.intermediate.dense.bias = pre_layer.intermediate.dense.bias
            empty_layer.output.dense.weight = pre_layer.output.dense.weight
            empty_layer.output.dense.bias = pre_layer.output.dense.bias
            empty_layer.output.LayerNorm.weight = pre_layer.output.LayerNorm.weight 
            empty_layer.output.LayerNorm.bias = pre_layer.output.LayerNorm.bias


    def share_enc_dec_param(self):
        # share all the parameters between encoder and decoder
        # self.decoder.embed_positions.weight = self.decoder.embed_positions.weight
        for enc_layer, dec_layer in zip(self.encoder.layers, self.decoder.layers):
            enc_layer.attention.self.query.weight = dec_layer.attention.self.query.weight
            enc_layer.attention.self.query.bias = dec_layer.attention.self.query.bias
            enc_layer.attention.self.value.weight = dec_layer.attention.self.value.weight
            enc_layer.attention.self.value.bias = dec_layer.attention.self.value.bias
            enc_layer.attention.self.query.weight = dec_layer.attention.self.query.weight
            enc_layer.attention.self.query.bias = dec_layer.attention.self.query.bias
            enc_layer.attention.output.dense.weight = dec_layer.attention.output.dense.weight
            enc_layer.attention.output.dense.bias = dec_layer.attention.output.dense.bias 
            enc_layer.attention.output.LayerNorm.weight = dec_layer.attention.output.LayerNorm.weight
            enc_layer.attention.output.LayerNorm.bias = dec_layer.attention.output.LayerNorm.bias
            enc_layer.intermediate.dense.weight = dec_layer.intermediate.dense.weight
            enc_layer.intermediate.dense.bias = dec_layer.intermediate.dense.bias
            enc_layer.output.dense.weight = dec_layer.output.dense.weight
            enc_layer.output.dense.bias = dec_layer.output.dense.bias
            enc_layer.output.LayerNorm.weight = dec_layer.output.LayerNorm.weight 
            enc_layer.output.LayerNorm.bias = dec_layer.output.LayerNorm.bias
        self.encoder.embeddings.word_embeddings.weight = self.decoder.embeddings.word_embeddings.weight
    


    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


    def forward(
        self, 
        batch, 
        prev_output_tokens=None
    ):
        # input_ids = batch['input_ids']
        # encoder_attention_mask = batch['attention_mask']
        # decoder_attention_mask = batch['tgt_attention_mask']
        # plan_text_input = batch["plan_text_ipt"]
        # plan_text_label = batch["plan_text_label"]
        # gen_text_input = batch["gen_text_ipt"]
        # final_gen_label = batch["labels"]
        # sn_pos = batch['sn_pos'] if 'sn_pos' in batch else None
        # step 1: calculate the sim_loss loss and the positional loss of <SEN>

        src_segment = batch['src_segment']
        src_span_length = batch['src_span_length']
        tgt_segment = batch['tgt_segment']
        tgt_span_length = batch['tgt_span_length']
        tgt_mask_segment = batch['tgt_mask_segment']
        encoder_out = self.encoder(src_segment, src_span_length)
        output, length_out_list, length_tgt_list, extra_states = self.decoder(
            normalize=False,
            encoder_out=encoder_out,
            tgt_segment=tgt_segment,
            tgt_mask_segment=tgt_mask_segment,
            tgt_span_length=tgt_span_length,
        )
        return output, length_out_list, length_tgt_list, extra_states
        # import pdb; pdb.set_trace()
        enc_hidden_state = enc_out.last_hidden_state
        # ipt_features = self.mean_pooling(enc_hidden_state, encoder_attention_mask) # [bsz, dim]
        
        dec_out, dec_hidden_state = self.decoder(
            
            input_ids=plan_text_input, 
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc_hidden_state,
            is_decoder=True, 
        )

        sim_losses, position_losses = [], []
        for bsz_idx in range(input_ids.size(0)):
            ins_ipt_features = ipt_features[bsz_idx]
            ins_pos_sn = sn_pos[bsz_idx]
            ins_sen_features = dec_hidden_state[bsz_idx].squeeze().index_select(0, ins_pos_sn.unique()) # [num_position, dim]

            # sim loss
            ins_ipt_features = torch.stack([ins_ipt_features] * ins_sen_features.size(0), dim=0)
            instance_sim = self.multul_info(ins_ipt_features, ins_sen_features)
            sim_losses.append(instance_sim)

            # sen positional loss
            prediction_scores = dec_out.logits[bsz_idx].squeeze().index_select(0, ins_pos_sn.unique())  # prediction score
            lables = plan_text_label[bsz_idx].squeeze().index_select(0, ins_pos_sn.unique())
            instance_pos_loss = self.ce_loss_fct(prediction_scores.view(-1, len(self.tokenizer)), lables.view(-1))
            position_losses.append(instance_pos_loss)

        sim_loss = torch.mean(torch.tensor(sim_losses))
        pos_loss = torch.mean(torch.tensor(position_losses))
        
        # step 2 calculate the generation loss
        dec_out_2, dec_hidden_state = self.decoder(
            input_ids=final_gen_label,
            encoder_attention_mask=encoder_attention_mask,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=enc_hidden_state,
            is_decoder=True,
            sn_pos=sn_pos,
            _add_special_token_mask=True
        )

        predict_scores = dec_out_2.logits
        ipt_text = self.tokenizer.batch_decode(gen_text_input)
        output_text = self.log_output_text(predict_scores)

        lm_loss = self.ce_loss(logits=predict_scores, labels=final_gen_label)
        label_text_copy = final_gen_label.clone().detach()
        label_text_copy = torch.where(label_text_copy<0, torch.tensor(self.tokenizer.pad_token_id).to(device=label_text_copy.device), label_text_copy)
        label_text = self.tokenizer.batch_decode(label_text_copy)

        final_loss = lm_loss + self.pos_alpha * pos_loss + self.sim_alpha * sim_loss

        return final_loss, lm_loss, pos_loss, sim_loss, output_text, ipt_text, label_text
    
    def log_output_text(self, logits):
        max_logits = torch.argmax(logits, dim=-1, keepdim=True)
        output_text = self.tokenizer.batch_decode(max_logits.squeeze())
        return output_text

    def generate(self, input_ids, attention_mask, plan_ipt, device):
        bos_token_id = self.bos_token_id
        enc_out = self.encoder(input_ids, attention_mask)
        enc_hidden_state = enc_out.last_hidden_state
        dec_out, plan_hidden_state = self.decoder(plan_ipt)  # non-autoregressive
        plan_gen_ids = torch.argmax(plan_hidden_state, dim=-1, keepdim=True)
        sn_pos_mask = plan_gen_ids.eq(self.sen_token_id)
        full_mask = torch.empty(input_ids.size()).fill_(self.mask_token_ids).to(device)
        bsz_idx = torch.arange(input_ids.size(0))
        full_mask = full_mask.masked_fill(sn_pos_mask, self.sen_token_id)
        full_mask[bsz_idx, 0] = self.gen_token_id

        gen_out = self.decoder.generate(
            full_mask,
            max_length=512,    
        )
    
    # step2: generation according to the sen_features
    def multul_info(self, dis1, dis2):
        dis_rnd = torch.cat((dis2[1:], dis2[0].unsqueeze(0)), dim=0)
        Ej = -F.softplus(-self.max_discriminator(dis1, dis2))
        Em = F.softplus(self.max_discriminator(dis1, dis_rnd))
        return Em - Ej

    def ce_loss(self, logits, labels):
        # we are doing next-token prediction; shift prediction scores and input ids by one
        shifted_prediction_scores = logits.contiguous()
        labels = labels.contiguous()
        lm_loss = self.ce_loss_fct(shifted_prediction_scores.view(-1, len(self.tokenizer)), labels.view(-1))
        return lm_loss


class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = RobertaAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = RobertaAttention(config, position_embedding_type="absolute")
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)

    def forward_encoder(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def forward_decoder(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        assert attention_mask is not None
        assert encoder_attention_mask is not None

        concaten_padding_mask = torch.cat((encoder_attention_mask, attention_mask), dim=1)
        concaten_hidden_states = torch.cat((encoder_hidden_states, hidden_states), dim=0)

        self_attention_outputs = self.attention(
            hidden_states=hidden_states,
            encoder_attention_mask=concaten_padding_mask,
            encoder_hidden_states=concaten_hidden_states,
            output_attentions=output_attentions,
            
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



# Copied from transformers.models.bert.modeling_bert.BertEncoder with Bert->Roberta
class Roberta(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        if config.apply_long_position:
            self.long_positions = (
                nn.Embedding(
                    2048, 
                    embedding_dim=config.hidden_size,
                    padding_idx=config.padding_idx
                )
            )
            self.long_positions.weight.data.normal_(mean=0.0, std=0.02)
        self.emb_layer_norm = nn.LayerNorm(config.hidden_size) # apply layer normalization
    
    def forward_embedding(
        self, 
        src_segment, 
        src_span_length,
    ):
        span_lst = []
        span_embed_lst = []
        span_padding_lst = []
        span_start = 0

        if self.config.apply_long_position:
            raw_long_position = self.long_positions(src_segment)
            offset = torch.zeros(src_segment.size(0)).to(src_segment)
        
        for span_idx, span_len in enumerate(src_span_length):
            src_tokens = src_segment[:, span_start: span_start + span_len]
            padding_mask = src_tokens.eq(self.padding)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        is_decoder=False
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, past_key_value, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                    is_decoder=is_decoder
                )

            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )

   
class RobertaEncoder(RobertaPreTrainedModel):
    def __init__(self, config, added_config, tokenizer, add_pooling_layer=True):
        super(RobertaEncoder, self).__init__(config)
        config.update(added_config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.layers = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        if config.apply_long_position:
            self.long_positions = (
                nn.Embedding(
                    2048, 
                    embedding_dim=config.hidden_size,
                    padding_idx=config.padding_idx
                )
            )
            self.long_positions.weight.data.normal_(mean=0.0, std=0.02)
        self.emb_layer_norm = nn.LayerNorm(config.hidden_size) # apply layer normalization
        if config.apply_long_position:
            self.long_positions = (
                nn.Embedding(
                    2048, 
                    embedding_dim=config.hidden_size,
                    padding_idx=config.padding_idx
                )
            )
            self.long_positions.weight.data.normal_(mean=0.0, std=0.02)
        self.emb_layer_norm = nn.LayerNorm(config.hidden_size) # apply layer normalization
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward_embedding(self, src_segment, src_span_length):
        span_list = []
        span_embed_list = []
        span_padding_list = []
        span_start = 0 
        
        if self.apply_long_position:
            raw_long_positions = self.long_positions(src_segment)
            offset = torch.zeros(src_segment.size(0)).to(src_segment)

        for span_idx, span_len in enumerate(src_span_length):
            src_tokens = src_segment[:, span_start: span_start+span_len]
            
            padding_mask = src_tokens.eq(self.padding_idx)
            token_embedding = self.embeddings(src_tokens, "relative")  # return token embeddings only
            x = embed = token_embedding
            
            if self.config.positional_embed == "absulate":
                # absulate positional embeddings:
                pos_ids = self.embeddings.create_position_ids_from_inputs_embeds(span_idx)
                pos_embedding = self.embeddings.position_embeddings(pos_ids)
            else:
                # relative positional embeddings:
                init_index = torch.arange(src_tokens.size(1)).unsqueeze(0).repeat(src_tokens.size(0), 1).to(src_tokens)
                if len(span_padding_list) != 0:
                    offset += (~span_padding_list[-1]).sum(-1)
                init_index += offset.unsqueeze(-1)
                lp_list = []
                for lp_idx in range(init_index.size(0)):
                    lp_list.append(raw_long_positions[lp_idx].index_select(0, init_index[lp_idx]))
                long_positions = torch.stack(lp_list, dim=0)
                x += long_positions

            x = self.embeddings.LayerNorm(x)
            x = self.embeddings.dropout(x)

            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

            span_list.append(x)
            span_embed_list.append(embed)
            span_padding_list.append(padding_mask)
            span_start += span_len
            
        return span_list, span_embed_list, span_padding_list

    def forward(
        self,
        src_segment: Optional[torch.Tensor] = None,
        src_span_length: Optional[torch.Tensor] = None,
    ):

        span_list, span_embed_list, span_padding_list = self.forward_embedding(src_segment, src_span_length)

        context_states_list = []
        context_padding_mask_list = []
        for span_x, span_padding_mask in zip(span_list, span_padding_list):
            import pdb; pdb.set_trace()
            x = span_x.transpose(0, 1)
            encoder_padding_mask = span_padding_mask
            
            if len(context_states_list) == 0:
                for layer in self.layers:
                    x = layer.forward_encoder(x, encoder_padding_mask)
            else:
                context_x = torch.cat(context_states_list, dim=0)
                context_padding_mask = torch.cat(context_padding_mask_list, dim=-1) 
                for i, layer in enumerate(self.layers):    
                    x = layer.forward_decoder(
                        hidden_states=x,  # hidden state of decoder
                        attention_mask=encoder_padding_mask,  # attention mask of decoder
                        encoder_hidden_states=context_x,  # hidden state of context
                        encoder_attention_mask=context_padding_mask, # attention mask of context
                        
                    )
            context_states_list.append(x)
            context_padding_mask_list.append(span_padding_mask)
        return {
            "encoder_out": context_states_list,
            "encoder_padding_mask": context_padding_mask_list,
            "encoder_embedding": [],
            "encoder_states": [],
            "src_tokens": [],
            "src_lengths": [],
        }


class RobertaDecoder(RobertaPreTrainedModel):
    def __init__(self, config, added_config, tokenizer):
        super(RobertaDecoder, self).__init__(config)
        config.update(added_config)
        self.config = config
        self.embeddings = RobertaEmbeddings(config)
        self.layers = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False
        if config.apply_long_position:
            self.long_positions = (
                nn.Embedding(
                    2048, 
                    embedding_dim=config.hidden_size,
                    padding_idx=config.padding_idx
                )
            )
            self.long_positions.weight.data.normal_(mean=0.0, std=0.02)
        self.emb_layer_norm = nn.LayerNorm(config.hidden_size) # apply layer normalization
        if config.apply_long_position:
            self.long_positions = (
                nn.Embedding(
                    2048, 
                    embedding_dim=config.hidden_size,
                    padding_idx=config.padding_idx
                )
            )
            self.long_positions.weight.data.normal_(mean=0.0, std=0.02)
        self.emb_layer_norm = nn.LayerNorm(config.hidden_size) # apply layer normalization
        self.lm_head = RobertaLMHead(config)
        self.bos = tokenizer.bos_token_id
        self.eos = tokenizer.eos_token_id
        self.mask_idx = tokenizer.mask_token_id
        self.pad = tokenizer.pad_token_id
        self.max_perdict_length = config.max_perdict_length
        self.embed_length = Embedding(self.max_perdict_length, config.hidden_size, None)
        self.embed_length.weight.data.normal_(mean=0.0, std=0.02)


    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        normalize,
        encoder_out, 
        tgt_segment,
        tgt_mask_segment, 
        tgt_span_length,
        step=0,
    ):
        features, len_out, len_tgt, extra_states = self.extract_features(
            encoder_out=encoder_out,
            tgt_segment=tgt_segment,
            tgt_mask_segment=tgt_mask_segment,
            tgt_span_length=tgt_span_length,
            step=step,
        )
        decoder_out = self.lm_head(features)
        return F.log_softmax(decoder_out, -1) if normalize else decoder_out, len_out, len_tgt, extra_states

    def forward_length(self, normalize, enc_feats, src_masks):
        enc_feats = _mean_pooling(enc_feats, src_masks)
        enc_feats = enc_feats.detach()
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1) if normalize else length_out

    def forward_length_prediction(self, length_out, enc_feats, src_masks, tgt_tokens=None):
        if tgt_tokens is not None:
            tgt_lengs = tgt_tokens.ne(self.pad).sum(1).long()
            length_tgt = tgt_lengs.clamp(min=0, max=self.max_perdict_length - 1)
        else:
            pred_lengs = length_out.max(-1)[1]
            length_tgt = pred_lengs 
            import pdb; pdb.set_trace()
            
        return length_tgt

    def forward_span_embedding(self, tgt_segment, tgt_span_length):
        span_token_list = []
        span_list = []
        span_padding_list = []
        span_start = 0

        if self.config.apply_long_position:
            raw_long_positions = self.long_positions(tgt_segment)
            offset = torch.zeros(tgt_segment.size(0)).to(tgt_segment)

        for span_idx, span_len in enumerate(tgt_span_length):
            
            span_tokens = tgt_segment[:, span_start: span_start + span_len]
            span_token_list.append(span_tokens)

            x = self.embeddings.word_embeddings(span_tokens)
            x += self.embeddings.token_type_embeddings(span_tokens)          

            if self.config.apply_long_position:
                init_index = torch.arange(span_tokens.size(1)).unsqueeze(0).repeat(span_tokens.size(0), 1).to(span_tokens)
                if len(span_padding_list) != 0:
                    offset += (~span_padding_list[-1]).sum(-1)
                init_index += offset.unsqueeze(-1)
                lp_list = []
                for lp_idx in range(init_index.size(0)):
                    lp_list.append(raw_long_positions[lp_idx].index_select(0, init_index[lp_idx]))
                long_positions = torch.stack(lp_list, dim=0)
                x += long_positions

            x = self.embeddings.LayerNorm(x) 
            x = self.embeddings.dropout(x)
            # if self.quant_noise is not None:
            #     x = self.quant_noise(x)
            
            padding_mask = span_tokens.eq(self.pad)
            if padding_mask is not None:
                x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

            span_list.append(x)
            span_padding_list.append(padding_mask)
            span_start += span_len

        return span_token_list, span_list, span_padding_list

    def extract_features(
        self,
        encoder_out=None,
        tgt_segment=None,
        tgt_mask_segment=None,
        tgt_span_length=None,
        step=None,
    ):
        
        span_mask_tokens_list, span_mask_list, span_mask_padding_list = self.forward_span_embedding(tgt_mask_segment, tgt_span_length)
        span_tokens_list, span_list, _ = self.forward_span_embedding(tgt_segment, tgt_span_length)
        
        # Build Context
        context_states_list = encoder_out["encoder_out"]
        context_padding_mask_list = encoder_out["encoder_padding_mask"]

        lengt_out_list = []
        length_tgt_list = []

        decoder_out = []
        ct_states_list = []
        for mask_x, span_x, span_target, span_padding_mask in zip(span_mask_list, span_list, span_tokens_list, span_mask_padding_list):
            
            x = None
            t_x = None
            context_x = torch.cat(context_states_list, dim=0)
            context_padding_mask = torch.cat(context_padding_mask_list, dim=-1) 

            for i, layer in enumerate(self.layers):
                
                if x is None:
                    x = mask_x  # [bsz, seq, dim]
                    # x = x.transpose(0, 1)
                    t_x = span_x  # [bsz, seq, dim]
                    # t_x = t_x.transpose(0, 1)
                
                x = layer.forward_decoder(
                    hidden_states=x,
                    attention_mask=span_padding_mask,
                    encoder_hidden_states=context_x,
                    encoder_attention_mask=context_padding_mask
                )
                
                t_x = layer.forward_decoder(
                    hidden_states=t_x,
                    attention_mask=span_padding_mask,
                    encoder_hidden_states=context_x,
                    encoder_attention_mask=context_padding_mask
                )

            # For length prediction
            length_out = self.forward_length(False, context_x, context_padding_mask)
            length_tgt = self.forward_length_prediction(length_out, context_x, context_padding_mask, span_target)
            lengt_out_list.append(length_out)
            length_tgt_list.append(length_tgt)

            # x = x.transpose(0, 1)
            decoder_out.append(x)
            
            ct_states_list.append(x)
            context_states_list.append(t_x)
            context_padding_mask_list.append(span_padding_mask)

        features = torch.cat(decoder_out, dim=1)
        return features, lengt_out_list, length_tgt_list, {"states": ct_states_list, "pad": context_padding_mask_list[1:]}
        



"""
class RobertaLMHead(nn.Module):
    # Head for masked language modeling.

    def __init__(self, embed_dim, output_dim, activation_fn, weight=None):
        super().__init__()
        self.dense = nn.Linear(embed_dim, embed_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.layer_norm = LayerNorm(embed_dim)

        if weight is None:
            weight = nn.Linear(embed_dim, output_dim, bias=False).weight
        self.weight = weight
        self.bias = nn.Parameter(torch.zeros(output_dim))

    def forward(self, features):
        x = self.dense(features)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = F.linear(x, self.weight) + self.bias
        return x

"""


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def _mean_pooling(enc_feats, src_masks):
    # enc_feats: T x B x C
    # src_masks: B x T or None
    if src_masks is None:
        enc_feats = enc_feats.mean(0)
    else:
        src_masks = (~src_masks).transpose(0, 1).type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
    return enc_feats