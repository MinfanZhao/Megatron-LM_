
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import json


# from megatron.core import mpu
# from megatron import get_args


class FakeDataSet(Dataset):
    def __init__(self, length, max_seq_length, vocab_size):
        self.length = length
        self.max_seq_length = max_seq_length
        self.vocab_size = vocab_size
        
    def __getitem__(self, index):
        tokens = torch.randint(0, self.vocab_size, [self.max_seq_length+1])
        return {'text': tokens}

    def __len__(self):
        return self.length
    

def get_first_valid_pos(mask):
    for pos, value in enumerate(mask):
        if value == 1:
            break
    return pos


def get_last_valid_pos(mask, first_valid_pos):
    pos = first_valid_pos
    for pos, value in enumerate(mask[first_valid_pos:], start=first_valid_pos):
        if value == 0:
            return pos - 1
    return pos
        
        
    
def build_attn_mask_and_position_ids_with_padding(masks, device):
    att_mask_batch, seq_length = masks.size()
    
    first_valid_pos = [get_first_valid_pos(m) for m in masks]
    last_valid_pos = [get_last_valid_pos(m, f) for m, f in zip(masks, first_valid_pos)]
    
    attention_mask = torch.tril(torch.ones(
        (att_mask_batch, seq_length, seq_length), device=device)).view(
            att_mask_batch, 1, seq_length, seq_length)
    
    position_ids = torch.arange(seq_length, dtype=torch.long,
                                device=device)
    position_ids = position_ids.unsqueeze(0).expand_as(masks)
    # We need to clone as the ids will be modifed based on batch index.
    position_ids = position_ids.clone()
    
    loss_mask = torch.ones(masks.size(), dtype=torch.float, device=device)
    
    for b in range(att_mask_batch):
        p = first_valid_pos[b]
        l = last_valid_pos[b]
        attention_mask[b, 0, :p,:] = 0
        attention_mask[b, 0, :,:p] = 0
        attention_mask[b, 0, l+1:,:] = 0
        attention_mask[b, 0, :,l+1:] = 0

        position_ids[b] -= p
        
    position_ids = position_ids.clamp(min=0)    
    attention_mask = (attention_mask < 0.5)    
    
    return attention_mask, loss_mask, position_ids


class KXDigitDataset(Dataset):
    def __init__(self, data_path, seq_length=1024, data_length=1024, padding_direction='left'):
        self.data_list = self.read_json(data_path)
        self.padding_length = seq_length + 1 - data_length 
        self.padding_direction = padding_direction
        assert self.padding_length >= 0
        print(f">>>>> padding length: {self.padding_length}")
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        if self.padding_direction == 'left':
            sample['labels'] = [-100]*self.padding_length + sample['labels']
            sample['input_ids'] = [0]*self.padding_length + sample['input_ids']
            sample['attention_mask'] = [0]*self.padding_length + sample['attention_mask']
        else:
            sample['labels'] =  sample['labels'] + [-100]*self.padding_length
            sample['input_ids'] = sample['input_ids'] +  [0]*self.padding_length 
            sample['attention_mask'] =  sample['attention_mask'] + [0]*self.padding_length
        sample = dict((k, torch.tensor(v)) for k, v in sample.items())
        return sample
            
    def read_json(self, data_path):
        with open(data_path, 'r') as f:
            lines = f.readlines()

        data_list = []
        for line in lines:
            sample = json.loads(line)
            data_list.append(sample)
        return data_list


# class SFTDataset(Dataset):
#     def __init__(self, raw_dataset, tokenizer):
#         # for sample in raw_dataset.
#         pass

#     def __getitem__(self, index):
#         pass


# def get_raw_dataset(data_path):
#     if data_path == '':
#         return DahoasRmstaticDataset(data_path)
#     else:
#         raise NotImplementedError()
#     pass


# def build_sft_train_valid_test_datasets(data_path, splits_string, tokenizer):
#     raw_dataset = get_raw_dataset(data_path)
#     train_raw_data = raw_dataset.get_train_data()
#     length = len(train_raw_data)

#     splits = get_train_valid_test_split_(splits_string, length)

#     def build_dataset(index, name):
#         dataset = None
#         if splits[index + 1] > splits[index]:
#             ids = np.arange(start=splits[index], stop=splits[index + 1],
#                                   step=1, dtype=np.int32)
            
#             dataset = SFTDataset(raw_dataset, tokenizer)
#         return dataset

#     train_dataset = build_dataset(0, 'train')
#     valid_dataset = build_dataset(1, 'valid')
#     test_dataset = build_dataset(2, 'test')


    
def test_attn_mask_and_position_ids_building():
    mask = torch.ones(2, 10)
    mask[0][:4] = 0
    mask[1][:6] = 0
    mask[0][8:] = 0
    print(mask)
    attention_mask, loss_mask, position_ids = build_attn_mask_and_position_ids_with_padding(mask, 'cuda:0')
    print('attention_mask: ', attention_mask)
    print('loss_mask', loss_mask)
    print('position_ids', position_ids)

if __name__ == '__main__':
    test_attn_mask_and_position_ids_building()
