def calcu_flops(num_layers=40, hidden_size=2048, vocab_size=32000,
                seq_length=2048, batch_size=1,  recompute=False):
    forward_flops = 24 * batch_size * seq_length * hidden_size * hidden_size + \
        4 * batch_size * seq_length * seq_length * hidden_size
    forward_flops *= num_layers
    vocab_flops = 6 * batch_size * seq_length * hidden_size * vocab_size
    if recompute:
        return 4 * forward_flops + vocab_flops
    else:
        return 3 * forward_flops + vocab_flops

if __name__ == "__main__":
    # model_args = {
    #     "num_layers" : 32, "hidden_size":4096, "vocab_size":50048, "seq_length":2048,  "batch_size":8}
    model_args = {
        "num_layers" : 40, "hidden_size":5120, "vocab_size":50048, "seq_length":2048,  "batch_size":1}
    time = 21.9
    device_num = 8
    global_batch_size = 128
    theoretical_peak = 312 # 14 for sw-26010pro, 
    
    
    
    flops = calcu_flops(**model_args)
    print(f"model args:{model_args}")
    print(f"model FLOPs:{flops / 1e12:.4} TFLOPs")
    
    flops_per_device = flops / 1e12 * global_batch_size / model_args["batch_size"] / time / device_num
    print(f"global batch size:{global_batch_size}, device_num:{device_num}, time cosumed:{time}")
    print(f"train flops per device:{flops_per_device:.4}FLOP/s")
    print(f"Percentage of theoretical peak FLOP/s {flops_per_device / theoretical_peak * 100:.4}%")