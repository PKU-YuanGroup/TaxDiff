"""
Sample new protein sequence from a pre-trained TaxDiff.
"""
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import argparse
import os
import random
import re
import time
import numpy as np
from data_reader.decoder import decode_protein
from diffusion import create_diffusion
from models import DiT_models

def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find DiT checkpoint at {model_name}'
    checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage)
    if "ema" in checkpoint:  # supports checkpoints from train.py
        checkpoint = checkpoint["ema"]
    return checkpoint

def main(args,class_lables):
    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    
    device = args.cuda_num
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    # Load model:
    latent_size = args.reshape_size // 4
    model = DiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes
    ).to(device)
    # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py: 
    state_dict = find_model(args.ckpt)
    model.load_state_dict(state_dict)
    
    model.eval()
    diffusion = create_diffusion(str(args.num_sampling_steps))

    # class_lables = [0] * args.num

    # Create sampling noise:
    n = len(class_lables)
    hidden_size=384 # see choose models
    z = torch.randn(args.num, hidden_size, 16, 16, device=device)
    y = torch.tensor(class_lables, device=device)
    print("# class_lables",y)
    # Setup classifier-free guidance:
    z = torch.cat([z, z], 0)
    y_null = torch.tensor([0] * n, device=device)
    # y_null = torch.tensor([args.num_classes] * n, device=device)
    y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
    
    # Sample protein sequence:
    samples = diffusion.p_sample_loop(
        model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

    # Save protein sequence:
    samples = samples.flatten(2).transpose(1,2)
    samples = torch.matmul(samples, model.embeding.weight.transpose(0,1))
    sub_tensor = samples.argmax(dim=-1)
    
    np.savetxt(args.output_file, sub_tensor.cpu().numpy(), fmt='%d')

    decode_protein(output_file,args.select_method,args.num,args.select_inner)
    end_time = time.time()
    print(f"completed in {end_time - start_times:.2f} seconds.")

if __name__ == "__main__":
    start_times = time.time() 
    file_name = '/remote-home/lzy/Tax_Diff/ckpt/0012802_eval.pt'
    output_file= '/remote-home/lzy/Tax_Diff/decode_data/'+'model_10.txt'
    num = 10
    torch.manual_seed(0)
    # conditional
    class_lables = torch.randint(low=1, high=int(23427), size=(1,num))

    # unconditional
    # class_lables = torch.randint(low=0, high=1, size=(1,num))

    class_lables = class_lables[0].numpy().tolist()
    lofo_file = output_file.replace('.txt', '_info.txt')
     
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-pro-12-h6-L16")
    parser.add_argument("--reshape-size", type=int, default=16)
    parser.add_argument("--num-classes", type=int, default=23427)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=file_name)
    parser.add_argument("--num", type=int, default=num)
    parser.add_argument("--output-file", type=str, default=output_file)
    parser.add_argument("--cuda-num", type=str, default='cuda:0')
    parser.add_argument("--select-method", type=bool, default=False)
    parser.add_argument("--select-inner", type=bool, default=True)
    parser.add_argument("--information", type=str, default='seed0')
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(os.path.dirname(lofo_file)):
        os.makedirs(os.path.dirname(lofo_file))
    with open(lofo_file, 'w') as file:
        file.write(str(args))
        file.write('\n')
        file.write(str(class_lables))
    main(args,class_lables)
    
    