import os

def list_dataset(base_dir, black_list):
    path_list = os.listdir(base_dir)
    print(f"handling {base_dir}")
    # print(path_list)
    res = []
    for path in path_list:
        if path == '.' or path == '..':
            continue
        elif os.path.isdir(os.path.join(base_dir, path)):
            res += list_dataset(os.path.join(base_dir, path), black_list)
        elif path.endswith('.idx'):
            file_path = os.path.join(base_dir, path)
            in_black_list = False
            for black_name in black_list:
                if black_name in file_path:
                    in_black_list = True
            if not in_black_list:
                file_size = os.path.getsize(file_path.replace('.idx', '.bin'))
                res.append((file_path.replace('.idx',''), file_size/1024/1024))
        elif path.endswith('bin'):
            if path.replace('.bin', '.idx') not in path_list:
                print("error in path:", path)
        
        
        
    return res

def get_dataset_weight(path, weight):
    for dataset_key in weight.keys():
        if dataset_key in path:
            return weight[dataset_key]
    raise ValueError("Unknown dataset")

if __name__ == "__main__":
    
    weight = {
        'baidu-baike/': 1,
        'ChinaNews-cn/': 1,
        'Law-cn/': 1,
        'Marxism-cn/':1,
        'openweb/': 1,
        'Patent-cn/': 1,
        'pile-cc':1,
        'pile-owt':1,
        'poem-cn/':1,
        'TextBook-cn/': 1,
        'WebText-cn/': 1,
        'WebText-en/': 1,
        'Wiki-cn/': 1,
        'wikipedia-cn/': 1,
        'wudao/':1,
    }
    path_list = []
    datasets = ['baidu-baike/', 'wanjuan/ChinaNews-cn/', 'wanjuan/Law-cn/', 'wanjuan/Patent-cn/', 
                'wanjuan/TextBook-cn/', 'wanjuan/WebText-cn/', 'wanjuan/Wiki-cn/', 'wanjuan/WebText-en/',  
                'pile/pile-cc', 'pile/pile-owt', 'wudao', 'Marxism-cn/', 'openweb', 'poem-cn/']
    
    
    # 'openweb/', 'baidu-baike/', 'wikipedia/', 'wudao/'
    black_list = ['wikipedia-cn/0921_wiki_0002_text_document', 'wikipedia-cn/0921_wiki_0007_text_document','wikipedia-cn/0921_wiki_0008_text_document']
    for dataset in datasets:
        path_list += list_dataset(os.path.join('/acsa-med/dataset/sunway/chinese-pretrain', dataset), black_list)
    
    data_path_str = ''
    total_size = 0
    
    for path in path_list:
        total_size += path[1]
    
    for path in path_list: 
        data_path_str += f"{get_dataset_weight(path[0], weight)* path[1]/ total_size:.8f} {path[0]} \\\n"
    
    print(path_list)
    print(f"{total_size/1024:.2f} GB")
        
    
    with open('./llama_pretrain_data.txt', 'w') as f:
        f.write(data_path_str)
    print(data_path_str)
