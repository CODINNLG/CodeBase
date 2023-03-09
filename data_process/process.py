import json
import hashlib
import multiprocessing
import transformers
import logging
import hashlib
import hydra
from omegaconf import DictConfig, OmegaConf
from collections import defaultdict
import os
import time
import psutil
import logging
import torch
import copy

class ProcessData():
    def __init__(self, config: DictConfig):
        """
        Summary of ProcessData:

        """
        self.config = config
        config.file_id = str(config.file_id)
        self.src_path = os.path.join(config.src_folder, config.file_id + config.suffix)

        self.tgt_folder = os.path.abspath(config.src_folder+os.path.sep+"..") if not config.tgt_folder else config.tgt_folder

        self.processed_folder = os.path.join(self.tgt_folder, 'processed')
        self.log_folder = os.path.join(self.tgt_folder, 'log')
        self.record_folder = os.path.join(self.tgt_folder, 'record')
        self.processed_path = os.path.join(self.processed_folder, config.file_id + config.suffix)
        self.log_path = os.path.join(self.log_folder, config.file_id + config.suffix)
        self.record_path = os.path.join(self.record_folder, config.file_id + config.suffix)

        self.domain_map = json.load(open(config.domain_map, 'r'))
        self.tokenizer = getattr(transformers, config.tokenizer.type)
        self.tokenizer = self.tokenizer.from_pretrained(config.tokenizer.name)
        self.cnt = 0
        self.len = 0

        self.record = {
            "src_path": self.src_path,
            "len": None,
            "num": 0,
            "len_sum": 0,
            "domain_cnt" : defaultdict(int),
            "data_src_cnt" : defaultdict(int)
        }
        

        self.create_dir(self.tgt_folder)
        self.create_dir(self.processed_folder)
        self.create_dir(self.log_folder)
        self.create_dir(self.record_folder)
        self.set_log()
        self.load()
        if not self.len: self.len = self.cal_len() 
        self.logger.info(f"{self.src_path} has {self.len} lines")
        self.f = open(self.src_path, 'r')
        self.logger.info(f"{self.src_path} is now open")
        self.speed = None
        self.start_time = time.time()
    
    def set_log(self):
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if self.config.mode == "debug":
            self.logger = logging.getLogger("debug.log")
        else:
            self.logger = logging.getLogger(self.config.file_id + ".log")
            handler = logging.FileHandler(os.path.join(self.tgt_folder, 'log', self.config.file_id + ".log"), encoding='UTF-8')
            handler.setLevel(logging.DEBUG)
            handler.setFormatter(formatter)
            console = logging.StreamHandler()
            console.setLevel(logging.DEBUG)
            self.logger.addHandler(handler)
            self.logger.addHandler(console)
        self.logger.info(f"Begin logging for file: {self.src_path}")
    
    def create_dir(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    def cal_len(self):
        with open(self.src_path) as f:
            cnt = 0
            for i in f: cnt += 1
        self.record['len'] = cnt
        return cnt

    def save_state(self):
        """
        save state
        """
        self.logger.info(f"saving state ...")
        with open(self.record_path, 'w') as f:
            f.write(json.dumps(self.record))

        self.logger.info(f"saving state's back up ...")
        with open(self.record_path + "_bk", 'w') as f:
            f.write(json.dumps(self.record))

    def load(self):
        """
        load state
        """
        self.logger.info(f"loading state from {self.record_path}")
        if os.path.exists(self.record_path):
            with open(self.record_path, 'r') as f:
                self.record = json.loads(f.read())
                self.len = self.record['len']
                self.record["domain_cnt"] = defaultdict(int, self.record["domain_cnt"])
                self.record["data_src_cnt"] = defaultdict(int, self.record["data_src_cnt"])
        

    def run(self):
        """
        process our data
        """
        for i in range(self.record['num'], self.len, self.config.record_interval):
            self.logger.info(f"{i} lines proccessed")
            lines4write = []
            self.num = i
            new_record = copy.deepcopy(self.record)
            for j in range(i, min(i + self.config.record_interval, self.len)):
                new_line = {}
                line = json.loads(self.f.readline())
                data_src = line['meta']['pile_set_name']
                ids = self.tokenizer(line['text'])['input_ids']

                new_line['Tok_ids'] = ids
                new_line['Corpus'] = data_src
                new_line['Domain'] = self.domain_map[data_src]
                new_line['Length'] = len(ids)
                new_line['Hash_id'] = hash(tuple(ids))
                new_line['Language'] = 'en'
                new_line['Id'] = i + j
                new_line['Task'] = 'Uns'
                new_line['Extra'] = {
                    "Tokenizer_name": self.config.tokenizer.name,
                    'Snapshot': self.config.file_id,
                    'num': self.len,
                }

                new_record['data_src_cnt'][data_src] += 1
                new_record['domain_cnt'][new_line['Domain']] += 1
                new_record['len_sum'] += new_line['Length']
                lines4write.append(json.dumps(new_line) + '\n')
            
            with open(self.processed_path, 'a+') as f:
                self.logger.info(f"{self.processed_path} is open, writing ...")
                for writeline in lines4write:
                    f.write(writeline)
                self.logger.info(f"{self.processed_path} is closing, done")

            new_record["num"] = min(i + self.config.record_interval, self.len)
            self.record = new_record
            self.save_state()
            now_time = time.time()
            cost_time = now_time - self.start_time
            avg_time = self.record['num'] / cost_time
            remain_time = (self.len - self.record['num']) / avg_time
            self.logger.info(f"Total {self.len} lines, processed {self.record['num']} lines.")
            self.logger.info(f"Avg speed is {avg_time} line/s, remain_time is {remain_time} s.")

        self.f.close()
        self.logger.info(f"{self.src_path} is now closed")
    
    def process_line(self):
        pass

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig) -> None:
    print(OmegaConf.to_yaml(config))
    p = ProcessData(config)
    p.run()
    
if __name__ == "__main__":
    main()
    # config = get_config()
    # print(config)
    # p = ProcessData()
    # p.run()