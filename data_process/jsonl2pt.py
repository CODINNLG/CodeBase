import json
import torch
import sys
from omegaconf import DictConfig, OmegaConf
import hydra
import os


def jsonl2pt(src_path, tgt_path):
    f = open(src_path, 'r')
    res = []
    while True:
        line = f.readline()
        if not line: break
        line_data = json.loads(line)
        for i in line_data:
            if type(line_data[i]) == list:
                line_data[i] = torch.tensor(line_data[i])
            if line_data[i] == 'Us' and i == 'Task':
                line_data[i] = 'Uns'
        res.append(line_data)
    torch.save(res, tgt_path)
    

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config):
    parent_folder = os.path.abspath(config.src_folder+os.path.sep+"..") if not config.tgt_folder else config.tgt_folder
    src_path = os.path.join(parent_folder, "processed", str(config.file_id) + config.suffix)
    tgt_folder = os.path.join(parent_folder, "processed_pt")
    tgt_path = os.path.join(tgt_folder, str(config.file_id) + '.pt')

    if not os.path.exists(tgt_folder):
        os.makedirs(tgt_folder)

    jsonl2pt(src_path, tgt_path)

if __name__ == "__main__":
   main()
    