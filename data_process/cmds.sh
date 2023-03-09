# process raw data

nohup python process.py --multirun file_id=10,20 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out10.log &
nohup python process.py --multirun file_id=11,21 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out11.log & 
nohup python process.py --multirun file_id=12,22 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out12.log &
nohup python process.py --multirun file_id=13,23 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out13.log &
nohup python process.py --multirun file_id=14,24 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out14.log &
nohup python process.py --multirun file_id=15,25 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out15.log &
nohup python process.py --multirun file_id=16,26 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out16.log & 
nohup python process.py --multirun file_id=17,27 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out17.log & 
nohup python process.py --multirun file_id=18,28 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out18.log & 
nohup python process.py --multirun file_id=19,29 > /opt/data/private/Group1/wpz/data/pile/nohup_out/out19.log &
nohup python process.py --multirun file_id=test,val > /opt/data/private/Group1/wpz/data/pile/nohup_out/test_val.log &



# trans .jsonl to .pt
# nohup python jsonl2pt.py --multirun file_id=test,val > /opt/data/private/Group1/wpz/data/pile/nohup_out/trans_test_val.log &
nohup python jsonl2pt.py --multirun file_id=00,01,02,03,04 > /opt/data/private/Group1/wpz/data/pile/nohup_out/trans_0-4l.log &
nohup python jsonl2pt.py --multirun file_id=05,06,07,08,09 > /opt/data/private/Group1/wpz/data/pile/nohup_out/trans_5-9.log &
nohup python jsonl2pt.py --multirun file_id=10,11,12,13,14 > /opt/data/private/Group1/wpz/data/pile/nohup_out/trans_10-14.log &
nohup python jsonl2pt.py --multirun file_id=15,16,17,18,19 > /opt/data/private/Group1/wpz/data/pile/nohup_out/trans_15-19.log &
nohup python jsonl2pt.py --multirun file_id=20,21,22,23,24 > /opt/data/private/Group1/wpz/data/pile/nohup_out/trans_20-24.log &
nohup python jsonl2pt.py --multirun file_id=25,26,27,28,29 > /opt/data/private/Group1/wpz/data/pile/nohup_out/trans_25-29.log &