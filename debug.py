import numpy as np
import torch
from unicore.data import Dictionary

# 1. 模拟您的真实数据结构
class MockDataset:
    def __init__(self):
        # 模拟您生成的 (10, 3) 坐标，并包在 list 里
        self.coords = [np.random.rand(10, 3).astype(np.float32)] 
        self.atoms = np.array([1] * 10)
        self.smi = "CC" # 模拟您的硬编码 SMILES
    
    def __getitem__(self, idx):
        return {"atoms": self.atoms, "coordinates": self.coords, "smi": self.smi}
    
    def __len__(self): return 1
    def set_epoch(self, epoch): pass

# 2. 模拟 ConformerSampleDataset
from unimol.data.conformer_sample_dataset import ConformerSampleDataset
raw_ds = MockDataset()
sample_ds = ConformerSampleDataset(raw_ds, 1, "atoms", "coordinates")

# 3. 模拟 NormalizeDataset
from unimol.data.normalize_dataset import NormalizeDataset
norm_ds = NormalizeDataset(sample_ds, "coordinates", normalize_coord=True)

# 4. 模拟 MaskPointsDataset
from unimol.data.key_dataset import KeyDataset
from unimol.data.mask_points_dataset import MaskPointsDataset

token_ds = KeyDataset(norm_ds, "atoms")
coord_ds = KeyDataset(norm_ds, "coordinates")
vocab = Dictionary()
vocab.add_symbol("[MASK]")

# 【修正】将 mask_prob 改为 0.99，避免 AssertionError
mask_ds = MaskPointsDataset(
    token_ds, coord_ds, vocab, 
    pad_idx=0, mask_idx=vocab.index("[MASK]"),
    noise_type="uniform", 
    noise=1.0, 
    mask_prob=0.99 # <--- 改这里
)

print("\n--- Noise Verification ---")
# 获取原始（归一化后）数据
clean_item = norm_ds[0]
# 获取加噪后数据
noised_item = mask_ds[0]

clean_coords = clean_item['coordinates']
noised_coords = noised_item['coordinates'].numpy()

print(f"Clean Coords Sample:\n{clean_coords[:2]}")
print(f"Noised Coords Sample:\n{noised_coords[:2]}")

# 计算差异
diff = np.abs(clean_coords - noised_coords)
total_diff = np.sum(diff)
print(f"\nTotal Difference Sum: {total_diff:.4f}")

if total_diff < 1e-4:
    print("\n[!!!严重错误!!!] 坐标完全一致，噪声没有加上去！")
    print("可能原因：mask 逻辑未生效，或 coordinates 引用被意外修改。")
else:
    print("\n[成功] 坐标已发生变化，噪声添加逻辑正常。")
    print("如果训练 Loss 仍为 0，可能是 Loss 计算部分的问题。")