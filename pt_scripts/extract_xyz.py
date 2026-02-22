import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Extract XYZ from LMDB based on indices.")
    parser.add_argument("--lmdb-path", type=str, required=True, help="Path to train.lmdb")
    parser.add_argument("--indices-file", type=str, required=True, help="Path to cold_start_indices.txt")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to save .xyz files")
    args = parser.parse_args()

    # 1. 创建输出目录
    os.makedirs(args.out_dir, exist_ok=True)

    # 2. 读取我们选中的 200 个分子的索引
    print(f"Loading indices from {args.indices_file}...")
    selected_indices = np.loadtxt(args.indices_file, dtype=int)
    print(f"Found {len(selected_indices)} molecules to extract.")

    # 3. 打开 LMDB 数据库
    print(f"Opening LMDB database: {args.lmdb_path}...")
    env = lmdb.open(
        args.lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=1,
    )

    # 4. 遍历提取
    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        # LMDB 的 key 通常是无序的二进制或字符串，但 Uni-Mol 数据集是按顺序写入的
        # 我们先获取所有的 keys
        keys = list(cursor.iternext(values=False))
        
        for idx in tqdm(selected_indices, desc="Extracting XYZs"):
            key = keys[idx]
            # 获取对应的序列化数据
            datapoint_pickled = txn.get(key)
            data = pickle.loads(datapoint_pickled)
            
            # 解析数据
            # Uni-Mol 默认的 key 是 'smi', 'atoms', 'coordinates'
            smi = data.get('smi', f'unknown_mol_{idx}')
            atoms = data.get('atoms')
            coords = np.array(data.get('coordinates'))
            
            # 如果坐标包含多个构象 (N_conf, N_atoms, 3)，默认取第一个
            if len(coords.shape) == 3:
                coords = coords[0]
            
            # 清理文件名 (去除 SMILES 中不能作为文件名的特殊字符，如 / \ * 等)
            safe_smi = "".join([c if c.isalnum() else "_" for c in smi])
            # 防止文件名过长
            safe_smi = safe_smi[:40] 
            
            # 生成 XYZ 文件路径
            filename = f"idx_{idx:07d}_{safe_smi}.xyz"
            filepath = os.path.join(args.out_dir, filename)
            
            # 写入标准的 XYZ 格式
            with open(filepath, 'w') as f:
                f.write(f"{len(atoms)}\n")
                f.write(f"SMILES: {smi} | Index: {idx}\n") # 第二行是注释行，保存完整的 SMILES
                for atom, coord in zip(atoms, coords):
                    f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")

    print(f"\n✅ All done! {len(selected_indices)} .xyz files have been saved to '{args.out_dir}'")

if __name__ == "__main__":
    main()