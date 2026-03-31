import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import argparse

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54,'Pt':78
}

def check_even_electrons(atoms):
    """检查分子的总电子数是否为偶数"""
    total_electrons = 0
    for atom in atoms:
        symbol = atom.capitalize()
        if symbol in ATOMIC_NUMBERS:
            total_electrons += ATOMIC_NUMBERS[symbol]
        else:
            return False
    return total_electrons % 2 == 0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb_path", type=str, required=True)
    # 此时读取的是包含备胎的矩阵
    parser.add_argument("--candidates_file", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    # 每个类簇我们需要多少个合规分子（之前200 + 新增200 = 总共400 -> 每个类簇需要2个）
    parser.add_argument("--need_per_cluster", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # 读取 200 x 10 的候选矩阵
    candidates_matrix = np.load(args.candidates_file)
    n_clusters = candidates_matrix.shape[0]

    env = lmdb.open(
        args.lmdb_path, subdir=False, readonly=True, lock=False, readahead=False, meminit=False, max_readers=1
    )

    valid_count = 0
    total_skipped = 0

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        keys = list(cursor.iternext(values=False))
        
        # 遍历 200 个聚类中心
        for cluster_idx in tqdm(range(n_clusters), desc="Processing Clusters"):
            candidates_for_this_cluster = candidates_matrix[cluster_idx]
            found_for_this_cluster = 0
            
            # 在这个聚类中心的候选池 (10个) 里依次寻找
            for rank, mol_id in enumerate(candidates_for_this_cluster):
                key = keys[mol_id]
                data = pickle.loads(txn.get(key))
                
                atoms = data.get('atoms')
                
                # 【核心逻辑】：如果不满足偶数电子，跳过，检查下一个(rank+1)
                if not check_even_electrons(atoms):
                    total_skipped += 1
                    continue
                
                # 满足条件，开始提取！
                smi = data.get('smi', f'unknown_mol_{mol_id}')
                coords = np.array(data.get('coordinates'))
                if len(coords.shape) == 3: coords = coords[0]
                    
                safe_smi = "".join([c if c.isalnum() else "_" for c in smi])[:40] 
                
                # 文件名中加入 cluster_id 和 rank，方便我们知道它是哪个类簇的第几选择
                filename = f"cluster{cluster_idx:03d}_rank{rank}_idx{mol_id:07d}_{safe_smi}.xyz"
                filepath = os.path.join(args.out_dir, filename)
                
                with open(filepath, 'w') as f:
                    f.write(f"{len(atoms)}\n")
                    f.write(f"SMILES: {smi} | Cluster: {cluster_idx} | Rank: {rank}\n")
                    for atom, coord in zip(atoms, coords):
                        f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
                
                valid_count += 1
                found_for_this_cluster += 1
                
                # 如果这个类簇已经找够了需要的分子（比如 2 个），就提前跳出循环，进入下一个类簇
                if found_for_this_cluster >= args.need_per_cluster:
                    break
            
            # 极端情况预警：如果 10 个备胎全是不合规的
            if found_for_this_cluster < args.need_per_cluster:
                print(f"\n[Warning] Cluster {cluster_idx} ran out of candidates! Only found {found_for_this_cluster}.")

    print(f"\n✅ All done!")
    print(f" - Valid .xyz Saved     : {valid_count} (Target was {n_clusters * args.need_per_cluster})")
    print(f" - Skipped Radicals     : {total_skipped}")

if __name__ == "__main__":
    main()