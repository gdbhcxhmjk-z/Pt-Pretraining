import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm

# === 补丁区 ===
import torch.serialization
original_load = torch.load
def safe_load(*args, **kwargs):
    if 'weights_only' not in kwargs: kwargs['weights_only'] = False
    return original_load(*args, **kwargs)
torch.load = safe_load

# === 路径引用 ===
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path: sys.path.append(project_root)

import unimol 
from unicore import checkpoint_utils, tasks, utils

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dict-name", type=str, default="dict.txt")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # 1. 加载模型
    print(f"[Part 1] Loading model...")
    state = checkpoint_utils.load_checkpoint_to_cpu(args.ckpt_path)
    model_args = state["args"]
    
    model_args.data = args.data_path
    model_args.task = "unimol"
    model_args.mode = "infer"
    model_args.dict_name = args.dict_name
    model_args.batch_size = args.batch_size

    task = tasks.setup_task(model_args)
    model = task.build_model(model_args)
    model.load_state_dict(state["model"], strict=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.half()

    # 2. 加载数据
    print(f"[Part 1] Loading LMDB from {args.data_path}...")
    task.load_dataset("train")
    dataset = task.dataset("train")
    
    dataloader = task.get_batch_iterator(
        dataset=dataset,
        batch_size=args.batch_size,
        ignore_invalid_inputs=True,
        required_batch_size_multiple=8,
        seed=42,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)

    # 3. 推理
    print("[Part 1] Extracting embeddings...")
    embeddings_list = []
    ids_list = []
    current_idx = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            net_input = batch["net_input"]
            bsz = net_input["src_tokens"].size(0)
            net_input = utils.move_to_cuda(net_input)
            
            outputs = model(**net_input, features_only=True)
            cls_repr = outputs[0][0, :, :] 
            
            embeddings_list.append(cls_repr.detach().cpu().float().numpy())
            batch_ids = list(range(current_idx, current_idx + bsz))
            ids_list.extend(batch_ids)
            current_idx += bsz

    # 4. 保存 (使用正式文件名)
    print("[Part 1] Saving embeddings...")
    all_embeddings = np.concatenate(embeddings_list, axis=0)
    
    # 修改点：文件名更正式
    emb_path = os.path.join(args.save_dir, "step1_structure_embeddings.npy")
    id_path = os.path.join(args.save_dir, "step1_structure_ids.txt")
    
    np.save(emb_path, all_embeddings)
    np.savetxt(id_path, np.array(ids_list), fmt='%d')
    
    print(f"[Part 1] Done. Saved to:\n  - {emb_path}\n  - {id_path}")

if __name__ == "__main__":
    main()