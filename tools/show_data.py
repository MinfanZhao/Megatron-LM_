import numpy as np 
# np_path = "test_data/backward/backward-10iter.npy"
np_path = "test_data/iter_0000101/param_fp16.npz"
# np_path = "test_data/iter_0000101/param_fp16.npz"
data = np.load(np_path, allow_pickle=True)
print(data.files)
key_name = 'module.module.language_model.encoder.layers.1.self_attention.query_key_value.weight'
# print(data[].shape)

iter_num=103
fp32_param = np.load(f"test_data/iter_{iter_num:07d}/param_fp32.npz", allow_pickle=True)
fp16_param = np.load(f"test_data/iter_{iter_num:07d}/param_fp16.npz", allow_pickle=True)
fp32_grad = np.load(f"test_data/iter_{iter_num:07d}/grad_fp32.npz", allow_pickle=True)
fp16_grad = np.load(f"test_data/iter_{iter_num:07d}/grad_fp16.npz", allow_pickle=True)

diff_param = fp32_param[key_name] - fp16_param[key_name]
diff_grad = fp32_grad[key_name] - fp16_grad[key_name]
print(np.mean(np.abs(diff_param)))
print(np.max(np.abs(diff_grad)))
print(np.max(fp16_grad[key_name]))
print(fp32_param[key_name])
print(fp16_param[key_name])
print(diff_param)

