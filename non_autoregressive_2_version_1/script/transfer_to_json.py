# src & trg ->json
import json
import sys

if __name__=='__main__':
    src=sys.argv[1]
    trg=sys.argv[2]
    json_result=sys.argv[3]

    with open(src, 'r') as s:
        content1=s.readlines()
    with open(trg, 'r') as t:
        content2=t.readlines()
    print(len(content1))
    print(len(content2))
    dict={}
    with open(json_result,'w+') as j:
        for i in range(len(content1)):
            dict['input']=content1[i][:-1]
            dict['label']=content2[i][:-1]
            js=json.dumps(dict)
            j.write(js)
            j.write('\n')
            dict.clear()
            # if i==300:
            #     break
