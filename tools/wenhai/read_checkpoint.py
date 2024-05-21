import torch,sys
sys.path.insert(0, '/root/staff/tys/Megatron-LM_/')
# Weight_PATH = '/root/staff/tys/Megatron-LM_/checkpoint/swin_transformer/test/iter_0000001/mp_rank_00_000/model_optim_rng.pt'
Weight_PATH = '/root/staff/tys/Megatron-LM_/checkpoint/swin_transformer/test_tp/iter_0000001/mp_rank_00/model_optim_rng.pt'
# Weight_PATH = 'epoch50_new0.pth'
weight = torch.load(Weight_PATH)

def print_ckpt(objects, keyname = ''):
    if isinstance(objects,dict):
        for key in objects.keys():
            print_ckpt(objects[key], key)
    elif isinstance(objects, torch.Tensor):
        print(keyname, objects.shape)
    else:
        return 
        # raise TypeError('not dict and tensor' + type(objects))  


if __name__ == '__main__':
    print_ckpt(weight, 'full')
    # print(weight.keys())