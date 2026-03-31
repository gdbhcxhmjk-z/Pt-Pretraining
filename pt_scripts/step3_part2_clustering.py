import os
import argparse
import numpy as np
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", type=str, required=True)
    parser.add_argument("--n-clusters", type=int, default=200)
    # 增加候选池深度，每个聚类中心找 10 个备胎
    parser.add_argument("--buffer-k", type=int, default=10) 
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    emb_path = os.path.join(args.save_dir, "step1_structure_embeddings.npy")
    id_path = os.path.join(args.save_dir, "step1_structure_ids.txt")
    
    if not os.path.exists(emb_path):
        print(f"Error: {emb_path} not found. Please run Part 1 first.")
        sys.exit(1)

    print(f"[Part 2] Loading embeddings from {emb_path}...")
    embeddings = np.load(emb_path)
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    ids = np.loadtxt(id_path, dtype=int)
    
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
        
        # 搜索前 buffer_k 个最近邻 (200, 10)
        D, I = gpu_index.search(kmeans.centroids, args.buffer_k)
        
    except ImportError:
        print("[Part 2] Mode: Sklearn (CPU) - Warning: Slow!")
        from sklearn.cluster import MiniBatchKMeans
        kmeans = MiniBatchKMeans(n_clusters=args.n_clusters, random_state=args.seed, batch_size=10000)
        kmeans.fit(embeddings)
        distances = kmeans.transform(embeddings)
        
        I = np.zeros((args.n_clusters, args.buffer_k), dtype=int)
        for i in range(args.n_clusters):
            I[i] = np.argsort(distances[:, i])[:args.buffer_k]

    # 将 faiss 查出的索引映射回真实的分子 ID
    # 结果是一个 shape 为 (200, 10) 的矩阵
    real_ids_matrix = ids[I]

    # 保存这个候选矩阵为 npy 格式，供提取脚本使用
    out_path = os.path.join(args.save_dir, "cold_start_candidates.npy")
    np.save(out_path, real_ids_matrix)
    
    print(f"[Part 2] Success! Saved 200 clusters x {args.buffer_k} candidates to {out_path}")

if __name__ == "__main__":
    main()