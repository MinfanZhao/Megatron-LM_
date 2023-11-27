import os
import json
import jsonlines
import multiprocessing


class Sampler(object):
    def __init__(self):
        pass
    
    def show(self, json_line):
        # print("json line type:", type(json_line))
        try:
            data = json.loads(json_line)
            # print(data)
        except:
            print(f"exception: {json_line}")
            return {}
        # print(data)
        return data

def sample_dataset(base_dir, save_dir,  dataset_name, dataset_path=None,black_list=[]):
    if dataset_path is not None:
        base_dir = os.path.join(base_dir, dataset_path)
        print(f"handling {dataset_name}")
    path_list = os.listdir(base_dir)
    # print(path_list)
    res = []
    for path in path_list:
        if path == '.' or path == '..':
            continue
        elif os.path.isdir(os.path.join(base_dir, path)):
            print(f"is dir: {os.path.join(base_dir, path)}")
            res += sample_dataset(base_dir, save_dir,  dataset_name, None, black_list)
        elif path.endswith('.jsonl') or path.endswith('.json'):
            file_path = os.path.join(base_dir, path)
            in_black_list = False
            for black_name in black_list:
                if black_name in file_path:
                    in_black_list = True
            
            if not in_black_list:
                file_size = os.path.getsize(file_path)
                if file_size / 1024 / 1024 > 1:
                    sample_num = 1000
                else:
                    sample_num = 100
                print(f"file path:{file_path}, file_size:{file_size/1024/1024:.2f} MB")
        sample_counter = 0
        print("open", file_path)
        fin = open(file_path, 'r', encoding='utf-8')
        # data = json.loads(fin)
        sampler = Sampler()
        pool = multiprocessing.Pool(1)
        sample_docs = pool.imap(sampler.show, fin, 1)
        for i, doc in enumerate(sample_docs, start=1):
            res.append(doc)
            sample_counter +=1 
            if sample_counter > sample_num:
                break
    if dataset_path is not None:
        save_path = os.path.join(save_dir, dataset_name + '_sample.jsonl')
        print(f"saving:{save_path}")
        with jsonlines.open(save_path, 'a') as fout:
            for line in res:
                fout.write(line) 
    
    return res

def get_dataset_weight(path, weight):
    for dataset_key in weight.keys():
        if dataset_key in path:
            return weight[dataset_key]
    raise ValueError("Unknown dataset")

if __name__ == "__main__":
    
    path_list = []
    datasets = {
        # 'baidu-baike':'baidu_baike', 
        # 'Wanjuan-ChinaNews-cn':'wanjuan/nlp/CN/ChinaNews-cn', 
        # 'Wanjuan-Law-cn':'wanjuan/nlp/CN/Law-cn', 
        # 'Wanjuan-Patent-cn':'wanjuan/nlp/CN/Patent-cn', 
        # 'Wanjuan-WebText-cn':'wanjuan/nlp/CN/WebText-cn', 
        # 'Wanjuan-Wiki-cn':'wanjuan/nlp/CN/Wiki-cn', 
        # 'Wanjuan-WebText-en':'wanjuan/nlp/EN/WebText-en',
        # 'pile-cc':'pile/pile-cc',
        # 'pile-owt':'pile/pile-owt',
        # 'wudao':'WuDaoCorpus2.0_base_200G',
        # 'Marxism-cn':'Marxism-cn',
        # 'openweb':'OpenWebText',
        'poem-cn':'poem-cn'
    }
    
    
    black_list = ['wikipedia-cn/0921_wiki_0002_text_document', 'wikipedia-cn/0921_wiki_0007_text_document','wikipedia-cn/0921_wiki_0008_text_document']
    for dataset_name, dataset_path in datasets.items():
        sample_dataset(
            '/acsa-med/dataset/sunway/chinese-pretrain-raw', 
            '/acsa-med/dataset/sunway/chinese-pretrain-raw/samples',
            dataset_name, dataset_path, black_list)

