import os

def list_dataset(base_dir):
    path_list = os.listdir(base_dir)
    # print(path_list)
    res = []
    for path in path_list:
        if path == '.' or path == '..':
            continue
        elif os.path.isdir(os.path.join(base_dir, path)):
            res += list_dataset(os.path.join(base_dir, path))
        elif path.endswith('.idx'):
            file_path = os.path.join(base_dir, path)
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
        'pile/pile-cc':1,
        'pile/pile-owt':1,
        'wudao/':1,
        'wikipedia/': 1,
        'baidu-baike/': 1,
        'openweb/': 1,
        'wanjuan/ChinaNews-cn/': 1,
        'wanjuan/Law-cn/': 1,
        'wanjuan/Patent-cn/': 1,
        'wanjuan/TextBook-cn/': 1,
        'wanjuan/WebText-cn/': 1,
        'wanjuan/Wiki-cn/': 1,
        'wanjuan/WebText-en/': 1,
    }
    path_list = []
    datasets = ['wanjuan/ChinaNews-cn/', 'wanjuan/Law-cn/', 'wanjuan/Patent-cn/', 'wanjuan/TextBook-cn/', 'wanjuan/WebText-cn/', 'wanjuan/Wiki-cn/', 'wanjuan/WebText-en/',  'pile/pile-cc', 'pile/pile-owt']
    # 'openweb/', 'baidu-baike/', 'wikipedia/', 'wudao/'
    
    for dataset in datasets:
        path_list += list_dataset(os.path.join('/acsa-med/dataset/sunway/chinese-pretrain', dataset))
    
    data_path_str = ''
    total_size = 0
    
    for path in path_list:
        total_size += path[1]
    
    for path in path_list: 
        data_path_str += f"{get_dataset_weight(path[0], weight)* path[1]/ total_size:.8f} {path[0]} \\\n"
    
    print(path_list)
    print(f"{total_size/1024:.2f} GB")
        
    
    with open('scripts/pretrain/data_path_wanjuan+pile.txt', 'w') as f:
        f.write(data_path_str)
    print(data_path_str)
