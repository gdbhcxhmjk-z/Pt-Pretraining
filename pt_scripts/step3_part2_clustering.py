import os
import lmdb
import pickle
import numpy as np
from tqdm import tqdm
import argparse

# 常见元素的原子序数（核外电子数），用于计算分子总电子数
ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Ag': 47, 'Cd': 48,
    'In': 49, 'Sn': 50, 'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Pt': 78
}

def check_even_electrons(atoms):
    """
    检查分子的总电子数是否为偶数 (闭壳层)
    """
    total_electrons = 0
    for atom in atoms:
        # 首字母大写，后续小写，匹配字典，例如 'c' -> 'C', 'cl' -> 'Cl'
        symbol = atom.capitalize()
        if symbol in ATOMIC_NUMBERS:
            total_electrons += ATOMIC_NUMBERS[symbol]
        else:
            # 如果遇到未知元素，为了安全起见默认视为不符合要求
            print(f"\nWarning: Unknown element '{atom}' found, skipping this molecule.")
            return False
            
    return total_electrons % 2 == 0

def main():
    parser = argparse.ArgumentParser(description="Extract XYZ from LMDB based on indices.")
    parser.add_argument("--lmdb_path", type=str, required=True, help="Path to train.lmdb")
    parser.add_argument("--indices_file", type=str, required=True, help="Path to cold_start_indices.txt")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save .xyz files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading indices from {args.indices_file}...")
    selected_indices = np.loadtxt(args.indices_file, dtype=int)
    print(f"Found {len(selected_indices)} candidate molecules to extract.")

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

    valid_count = 0
    odd_electron_count = 0

    with env.begin(write=False) as txn:
        cursor = txn.cursor()
        keys = list(cursor.iternext(values=False))
        
        for idx in tqdm(selected_indices, desc="Extracting XYZs"):
            key = keys[idx]
            datapoint_pickled = txn.get(key)
            data = pickle.loads(datapoint_pickled)
            
            smi = data.get('smi', f'unknown_mol_{idx}')
            atoms = data.get('atoms')
            coords = np.array(data.get('coordinates'))
            
            if len(coords.shape) == 3:
                coords = coords[0]
                
            # 【修改点】：执行电子数检查
            if not check_even_electrons(atoms):
                odd_electron_count += 1
                continue # 跳过奇数电子分子
            
            safe_smi = "".join([c if c.isalnum() else "_" for c in smi])
            safe_smi = safe_smi[:40] 
            
            filename = f"idx_{idx:07d}_{safe_smi}.xyz"
            filepath = os.path.join(args.out_dir, filename)
            
            with open(filepath, 'w') as f:
                f.write(f"{len(atoms)}\n")
                f.write(f"SMILES: {smi} | Index: {idx}\n")
                for atom, coord in zip(atoms, coords):
                    f.write(f"{atom:2s} {coord[0]:12.6f} {coord[1]:12.6f} {coord[2]:12.6f}\n")
            
            valid_count += 1

    print(f"\n✅ All done!")
    print(f" - Candidates Processed : {len(selected_indices)}")
    print(f" - Odd Electron skipped : {odd_electron_count}")
    print(f" - Valid .xyz Saved     : {valid_count} (Saved in '{args.out_dir}')")

if __name__ == "__main__":
    main()