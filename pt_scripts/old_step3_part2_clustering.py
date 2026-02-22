import os
import argparse
import numpy as np
import time
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-clusters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 修改点：读取正式文件名的中间结果
    emb_path = os.path.join(args.save_dir, "step1_structure_embeddings.npy")
    id_path = os.path.join(args.save_dir, "step1_structure_ids.txt")
    
    if not os.path.exists(emb_path):
        print(f"Error: {emb_path} not found. Please run Part 1 first.")
        sys.exit(1)

    print(f"[Part 2] Loading embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    ids = np.loadtxt(id_path, dtype=int)
    
    print(f"[Part 2] Matrix shape: {embeddings.shape}")

    # Faiss 聚类
    try:
        import faiss
        print("[Part 2] Mode: Faiss GPU")
        
        d = embeddings.shape[1]
        k = args.n_clusters
        
        res = faiss.StandardGpuResources()
        
        kmeans = faiss.Kmeans(d, k, niter=25, verbose=True, gpu=True, seed=args.seed)
        kmeans.train(embeddings)
        
        index_flat = faiss.IndexFlatL2(d)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        gpu_index.add(embeddings)
        
        D, I = gpu_index.search(kmeans.centroids, 1)
        raw_indices = I.flatten()
        
    except ImportError:
        print("[Part 2] Mode: Sklearn (CPU) - Warning: Slow!")
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed, batch_size=10000)
        kmeans.fit(embeddings)
        distances = kmeans.transform(embeddings)
        raw_indices = np.argmin(distances, axis=0)

    # 保存结果
    selected_ids = [ids[idx] for idx in raw_indices]
    selected_ids = sorted(selected_ids)
    
    out_path = os.path.join(args.save_dir, "cold_start_indices.txt")
    np.savetxt(out_path, np.array(selected_ids), fmt='%d')
    
    print(f"[Part 2] Success! {len(selected_ids)} indices saved to {out_path}")

if __name__ == "__main__":
    main()