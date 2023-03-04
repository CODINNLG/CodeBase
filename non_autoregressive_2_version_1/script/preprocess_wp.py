import sys
import os
import json
import re

def generate_json_pairs(ori_source, ori_target=None, sep_token="<SEN>", add_=False, already_token=None):
    json_content = dict()
    json_content['src'] = ori_source
    if ori_target is not None:
        if add_:
            ori_target = add_special_token(ori_target, sep_token, already_special_token=already_token)
        json_content['tgt'] = ori_target
    return json.dumps(json_content)

def write_back_file(source_file, writeback_file, target_file=None, add_=False):
    with open(source_file, 'r') as f:
        src_con = f.readlines()
        src_con = [i.strip() for i in src_con]

    tgt_con = [None] * len(src_con)
    if target_file is not None:
        with open(target_file, 'r') as f:
            tgt_con = f.readlines()
            tgt_con = [i.strip() for i in tgt_con]

    with open(writeback_file, 'w') as f:
        for src, tgt in zip(src_con, tgt_con):
            f.write(generate_json_pairs(src, tgt, add_=add_) + '\n')

def add_special_token(text, spe_token,  already_special_token=None):
    '''
    if already_special_token, replace those token with sentence special token 
    '''
    if already_special_token is not None:
        text = text.replace(already_special_token, spe_token)
    else:
        # split the sentence with spe_token
        pattern = re.compile('[.!?]+')
        text = re.sub(pattern, spe_token, text)
    text_split = text.split(spe_token)
    if text_split[-1] != '':  # add one sep token at the end of the sentence
        text_split.append('')
    
    final_text = f" {spe_token} ".join(text_split)
    return final_text


if __name__ == '__main__':
    data_dir = sys.argv[1]
    file_names = [
        ('train.source', 'train.target'), 
        ('val.source', 'val.target'), 
        ('test.source', 'test.target')
    ]
    for source_file, tgt_file in file_names:
        src_path = os.path.join(data_dir, source_file)
        tgt_path = os.path.join(data_dir, tgt_file)
        json_file = os.path.join(data_dir, tgt_file.split('.')[0] + '.json')
        write_back_file(src_path, json_file, tgt_path, add_=True)

    
    


    

