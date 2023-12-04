

# def read_jsonl_file(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:  # 打开文件
#         for line in file:  # 遍历文件中的每一行
#             json_obj = json.loads(line.strip())  # 将每行的内容加载为JSON对象，并移除可能的空白字符
#             yield json_obj
    

# def process_file(input_file_path, output_file_path):
#     line_count = 0
#     # 确保输出目录存在
#     os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

#     for json_obj in read_jsonl_file(input_file_path):
#         line_count += 1
#         text = json_obj['text']
#         processed_text = process_text(text)
#         json_obj['text'] = processed_text
#         # 以追加模式写入对应的output_file_path
#         with open(output_file_path, 'a', encoding='utf-8') as f:
#             json.dump(json_obj, f, ensure_ascii=False)
#             f.write('\n')
#         if line_count % 5000 == 0:
#             print(f'Processed {line_count} lines')

# def main():
#     # Loop through all jsonl files in the directory
#     for file_index, file_name in enumerate(glob.glob(os.path.join(input_dir, '*.jsonl'))):
#         print(f'Processing file {file_index + 1}: {file_name}')  # 打印当前处理的文件信息
#         # 构建输出文件的路径
#         output_file_name = os.path.basename(file_name) + 'processed.jsonl'
#         output_file_path = os.path.join(output_dir, output_file_name)
#         process_file(file_name, output_file_path)  # 将输入文件和输出文件路径传递给process_file函数

# main()


# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Processing data for pretraining."""
import random, re, os
import argparse
import json
import multiprocessing
import os
import sys
import jsonlines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))
import time

special_word_list = ["作者:", "作者：", "编辑：", "编辑:", 
                   "记者/", "摄影/", " 文/摄", "快报记者 ", "（记者 ", "【来源】", "【全媒体记者】", " 摄\n", "联系邮箱：", "联系邮箱:", 
                   "版权所有", "版权声明", "备案号", "微信：","您的当前位置是：", "延伸阅读：", "链接：","图片来源:", "图片来源：", "本站声明:",
                   "本站声明：", "地址：", "下一篇：", "相关文章：", "电话：", "友情链接网站：", "免责声明："]
special_head_tail_word_list = ["转载","点赞", "关注", "点击", "打赏", "邮箱", "公众号", "qq", "QQ", "服务热线", "销售热线", "浏览数", "点赞", "留言", "分享", "关注", "访问"]

input_dir = '/acsa-med/dataset/sunway/chinese-pretrain-raw/wanjuan/nlp/CN/WebText-cn'
output_dir = '/acsa-med/dataset/sunway/chinese-pretrain-raw/wanjuan/nlp/CN/WebText-cn_processed'



class WebTextCleaner(object):
    def __init__(self, args):
        self.args = args

    def initializer(self):
        pass
    
    def clean_web_text(self, text):
        lines = text.split('\n')
        problem_lines = []

        # Define your special words here
        link_pattern = re.compile(r'http[s]?://\S+')
        more_link_pattern = re.compile(r'.*更多.*http[s]?://\S+')

        # Step 1: Remove lines containing words in special_word_list and add them to problem_lines
        lines, problem_lines = zip(*[(line, '') if not any(special_word in line for special_word in special_word_list) else ('', line) for line in lines])
        lines = list(filter(None, lines))  # Remove empty entries
        problem_lines = list(filter(None, problem_lines))  # Remove empty entries
        total_lines = len(lines)

        # Step 2: Process first 6% and last 6% of lines
        threshold = int(total_lines * 0.06)
        if threshold < 1:
            threshold = 1
        for i in list(range(threshold)) + list(range(-threshold, 0)):
            try:
                current_line = lines[i]
            except:
                return ''
            if any(word in current_line for word in special_head_tail_word_list):
                if random.random() < 0.95:
                    problem_lines.append(current_line)  # Add to problem lines
                    lines[i] = ''
            elif len(current_line) <= 4 or more_link_pattern.match(current_line):
                if random.random() < 0.95:
                    problem_lines.append(current_line)  # Add to problem lines
                    lines[i] = ''

        # Step 3: Remove invalid links
        for i in range(len(lines)):
            if link_pattern.search(lines[i]) and not any(keyword in lines[i] for keyword in ['链接', '地址', '来源', '入口', '平台', '官网', '网站', '网址','Website', 'website']):
                problem_lines.append(lines[i])
                lines[i] = link_pattern.sub('', lines[i])

        # Rejoin the lines into a single text string, omitting empty lines
        return '\n'.join(filter(None, lines))
        
    def clean_text(self, json_line):
        try:
            data = json.loads(json_line)
        except:
            print(f"exception: {json_line}")
            return {}, 0

        for key in self.args.json_keys:
            if not data.__contains__(key):
                print(f"not has {key}: {data}")
                return {}, 0
            elif len(data[key]) == 0:
                return {}, 0
            data[key] = self.clean_web_text(data[key])
            if len(data[key]) == 0:
                return {}, 0
            
        return data, len(json_line)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Path to input JSON')
    group.add_argument('--json-keys', nargs='+', default=['text'],
                       help='space separate listed of keys to extract from json')

    group = parser.add_argument_group(title='output data')
    group.add_argument('--output-prefix', type=str, required=True,
                       help='Path to binary output file without suffix')

    group = parser.add_argument_group(title='runtime')
    group.add_argument('--workers', type=int, required=True,
                       help='Number of worker processes to launch')
    group.add_argument('--chunk-size', type=int, required=True,
                       help='Chunk size assigned to each worker process')
    group.add_argument('--log-interval', type=int, default=100,
                       help='Interval between progress updates')
    args = parser.parse_args()


    return args

def main():
    args = get_args()
    startup_start = time.time()

    if os.path.isdir(args.input):
        dir_path, file_name_list = args.input, os.listdir(args.input)
    
    else:
        (dir_path, file_name) = os.path.split(args.input)
        file_name_list = [file_name]
    
    for file_name in file_name_list:

        if not (file_name.endswith('.json') or file_name.endswith('.jsonl')):
            continue
        
        file_path = os.path.join(dir_path, file_name)
        
        print("Opening", file_path)
        fin = open(file_path, 'r', encoding='utf-8')


        cleaner = WebTextCleaner(args)
        pool = multiprocessing.Pool(args.workers, initializer=cleaner.initializer)
        clean_docs = pool.imap(cleaner.clean_text, fin, args.chunk_size)


        print(f"Output prefix: {args.output_prefix}")
        result = []
        output_file = "{}/{}.jsonl".format(args.output_prefix, 
                                                            os.path.splitext(file_name)[0])
        print(f"Output file:{output_file}")

        startup_end = time.time()
        proc_start = time.time()
        total_bytes_processed = 0
        print("Time to startup:", startup_end - startup_start)

        for i, (doc, bytes_processed) in enumerate(clean_docs, start=1):
            total_bytes_processed += bytes_processed
            if bytes_processed == 0:
                continue
            result.append(doc)
            if i % args.log_interval == 0:
                current = time.time()
                elapsed = current - proc_start
                mbs = total_bytes_processed/elapsed/1024/1024
                print(f"Processed {i} documents",
                    f"({i/elapsed} docs/s, {mbs} MB/s).",
                    file=sys.stderr)
        print(f"Done! Now finalizing from {file_path}, total got {len(result)} samples")
        pool.terminate()
        with jsonlines.open(output_file, 'a') as fout:
            for line in result:
                fout.write(line) 
            

if __name__ == '__main__':
    main()
