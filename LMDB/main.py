import lmdb, json
import psutil, os
from torch.utils.data import Dataset, DataLoader


'''
env = lmdb.open(): 创建 lmdb 环境
txn = env.begin(): 建立事务
txn.put(key, value): 进行插入和修改
txn.delete(key): 进行删除
txn.get(key): 进行查询
txn.cursor(): 进行遍历
txn.commit(): 提交更改
'''


def show_mem_used():
    print(u'Memory Used = %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024))


# ======================= Example 1: upload data to lmdb  ======================= #

'''
lmdb_path = "lmdb_dir"
data_dir = "/opt/data/private/Group1/dyy/data/c4-train.01818-of-07168.json"

os.system("rm -rf %s" % lmdb_path)


# upload dataset 1

# step 1: get lmdb size
env = lmdb.open(lmdb_path)
txn = env.begin(write=False)
tot_len = txn.stat()['entries']
env.close()

# step 2: get <key, value> pair and upload
env = lmdb.open(lmdb_path, map_size=1099511627776)
txn = env.begin(write=True)
with open(data_dir, "r") as f:
    for id, line in enumerate(f.readlines()):
        text = json.loads(line)
        txn.put(str(id + tot_len).encode(), text['text'].encode())
txn.commit()  # update should be committed
env.close()


# upload dataset 2
env = lmdb.open(lmdb_path)
txn = env.begin(write=False)
tot_len = txn.stat()['entries']
env.close()

env = lmdb.open(lmdb_path, map_size=1099511627776)
txn = env.begin(write=True)
with open(data_dir, "r") as f:
    for id, line in enumerate(f.readlines()):
        text = json.loads(line)
        txn.put(str(id + tot_len).encode(), ("new: " + text['text']).encode())
txn.commit()  # update should be committed
env.close()
'''

# ========================== Example 2: load datasets  ========================== #

class LMDBDataset(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        # self.env = lmdb.open(db_path)
        self.env = lmdb.open(db_path, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False, buffers=True) as txn:
            self.length = txn.stat()['entries']
        show_mem_used()
        
    def __getitem__(self, index):
        with self.env.begin(write=False, buffers=True) as txn:
            buf = txn.get(str(index).encode())
            val = bytes(buf)
        return val
        
    def __len__(self):
        return self.length


# lmdb_dataset = LMDBDataset("lmdb_dir")
# print(lmdb_dataset[0])
# for i in range(len(lmdb_dataset)):
#     data = lmdb_dataset[i]
#     show_mem_used()

env = lmdb.open('lmdb_dir', readonly=True, lock=False, readahead=False, meminit=False)
with env.begin(write=False) as txn:
    print(txn.stat())
    