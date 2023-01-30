from xmlrpc.client import Boolean
import torch
from torch import Tensor, device, nn
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

def _add_special_tokens(tokenizer):
    special_token_lst = ["<PLN>", "<COR>", "<SEN>", "<GEN>"]
    return None


def _expand_mask(mask: torch.Tensor, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    - 1.0 mask
    - 0.0 non mask
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len
    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len)
    inverted_mask = 1 - expanded_mask
    # return mask matirc where 1.0 is mask and 0 is non mask
    return inverted_mask
    # return the padding mask metric 
    # fill the padding position with -inf
    return inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(dtype).min)


def _make_causal_mask(input_ids_shape, device, past_key_values_length=0, sn_position=None, _add_special_token_mask=False):
        """
        Make causal mask used for span mask self-attention.
        The index of the sentence begins from 1 not 0.
        sn_position # [bsz, seq_len]
        """ 
        bsz, tgt_len = input_ids_shape
        span_mask = torch.zeros((bsz, tgt_len, tgt_len)).to(device)
        special_token_mask = torch.zeros((bsz, tgt_len)).to(device)  # first mask then expend
       
        if sn_position is not None:
            matrix1 = torch.stack([sn_position] * tgt_len, 1)
            matrix2 = matrix1.permute(0, 2, 1) + 1
            matrix2 = matrix2.to(device)
            span_mask.masked_fill_(matrix1 < matrix2, 1)

        if _add_special_token_mask:
            # mask according to the special tokens
            x_ids, y_ids = torch.arange(0, bsz)[:, None], sn_position - 1
            special_token_mask[x_ids, y_ids] = 1
            special_token_mask = torch.stack([special_token_mask] * tgt_len, 1)
            
        to_mask = span_mask + special_token_mask
        to_mask = torch.where(to_mask>=1, 0.0, 1.0)
        '''
        return shape
        - 1.0 mask
        - 0.0 non mask
        '''
        return to_mask
        # return to_mask.where(to_mask<1, torch.tensor(1)) # saint check
        to_mask = torch.where(to_mask>=1, 0, torch.tensor(torch.finfo(dtype).min))  # convert to 0 and -inf 
        if past_key_values_length > 0:
            to_mask = torch.cat([torch.zeros(tgt_len, past_key_values_length, dtype=dtype), to_mask], dim=-1)
        return to_mask[:, None, :, :].expand(bsz, 1, tgt_len, tgt_len + past_key_values_length)



def _prepare_decoder_attention_mask(attention_mask, input_shape, device, dtype, past_key_values_length, sn_position=None, _add_special_token_mask=False):
    # create casual mask
    # [bsz, seq_len] -> [batch_size, num_heads, seq_length, seq_length]
    combined_attention_mask = None
    if input_shape[-1] > 1:
        combined_attention_mask = _make_causal_mask(
            input_shape, device=device, past_key_values_length=past_key_values_length, 
            sn_position=sn_position, _add_special_token_mask=_add_special_token_mask
        )                              
        combined_attention_mask = combined_attention_mask[:,None,:,:]
    
    if attention_mask is not None:
        expanded_attn_mask = _expand_mask(attention_mask, tgt_len=input_shape[-1])
        combined_attention_mask = (
            expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
        )
    # print("-"*20)
    # print(sn_position)
    # print("combined_attention_mask")
    # print(combined_attention_mask)
    # print("expanded_attn_mask")
    # print(expanded_attn_mask)
    # print("-"*20)
    combined_attention_mask = combined_attention_mask.to(dtype=dtype)

    return combined_attention_mask

def _mask_casual_sential_attention_mask(special_pos: List, max_seq_len: int):
    '''
    pos in specical_pos should start from 1, not zero
    '''
    res = []
    prev = 0
    for i in special_pos:
        res += [i] * (i - prev)
        prev = i
    # add the last token 
    res += [prev]
    if len(res) < max_seq_len:
        res += [0] * (max_seq_len - len(res))
    assert len(res) == max_seq_len, \
        "position should utill meets max_seq_len, check if the last token is <SEN>"
    return res


def get_extended_attention_mask(attention_mask: Tensor, input_shape: Tuple[int], device: device, dtype, \
    is_decoder=False, past_key_values_length=None, sn_position=None, _add_special_token_mask=False) -> Tensor:
    """
    Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

    Arguments:
        attention_mask (`torch.Tensor`):
            Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
        input_shape (`Tuple[int]`):
            The shape of the input to the model.
        device: (`torch.device`):
            The device of the input to the model.

    Returns:
        `torch.Tensor` The extended attention mask, with a the same dtype as `attention_mask.dtype`.
    """
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
        extended_attention_mask = 1 - extended_attention_mask
    elif attention_mask.dim() == 2:
        # Provided a padding mask of dimensions [batch_size, seq_length]
        # - if the model is a decoder, apply a causal mask in addition to the padding mask
        # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if is_decoder and sn_position is not None:
            extended_attention_mask = _prepare_decoder_attention_mask(
                attention_mask, 
                input_shape, 
                device=device,
                dtype=dtype,
                past_key_values_length=past_key_values_length, 
                sn_position=sn_position, 
                _add_special_token_mask=_add_special_token_mask
            ) # return mask matric where 1.0 is mask and 0.o is no mask
        else:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = 1 - extended_attention_mask
    else:
        raise ValueError(
            f"Wrong shape for input_ids (shape {input_shape}) or attention_mask (shape {attention_mask.shape})"
        )
    
    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    extended_attention_mask = extended_attention_mask.to(dtype=dtype).to(device=device)  # fp16 compatibility
    extended_attention_mask = extended_attention_mask * -10000.0
    return extended_attention_mask


def invert_attention_mask(attention_mask: Tensor, dtype=torch.float32) -> Tensor:
    """
    Invert an attention mask (e.g., switches 0. and 1.).
    Args:
        encoder_attention_mask (`torch.Tensor`): An attention mask.

    Returns:
        `torch.Tensor`: The inverted attention mask.
    """
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    if attention_mask.dim() == 2:
        extended_attention_mask = attention_mask[:, None, None, :]
    extended_attention_mask = extended_attention_mask.to(dtype=dtype)
    if dtype == torch.float16:
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e4
    elif dtype in [torch.bfloat16, torch.float32]:
        extended_attention_mask = (1.0 - extended_attention_mask) * -1e9
    else:
        raise ValueError(
            f"{dtype} not recognized. `dtype` should be set to either `torch.float32` or `torch.float16`"
        )

    return extended_attention_mask