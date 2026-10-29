| dataset   | index        | qps@0.90                            | config@0.90                 | qps@0.95                            | config@0.95                 |
|:----------|:-------------|:------------------------------------|:----------------------------|:------------------------------------|:----------------------------|
| sift      | HNSW         | 15463 (test recall 0.918)           | M=16, efC=200, efSearch=32  | 8894 (test recall 0.971)            | M=16, efC=200, efSearch=64  |
| sift      | IVF-Flat     | 3578 (test recall 0.929)            | nlist=4096, nprobe=32       | 2343 (test recall 0.958)            | nlist=4096, nprobe=48       |
| sift      | IVF-PQ       | not reached (max tune recall 0.726) |                             | not reached (max tune recall 0.726) |                             |
| sift      | Exact (Flat) | 42 (recall 1.000)                   |                             | 42 (recall 1.000)                   |                             |
| glove     | HNSW         | 2001 (test recall 0.918)            | M=32, efC=200, efSearch=192 | 995 (test recall 0.955)             | M=32, efC=200, efSearch=384 |
| glove     | IVF-Flat     | 870 (test recall 0.915)             | nlist=4096, nprobe=192      | 469 (test recall 0.954)             | nlist=4096, nprobe=384      |
| glove     | IVF-PQ       | not reached (max tune recall 0.865) |                             | not reached (max tune recall 0.865) |                             |
| glove     | Exact (Flat) | 50 (recall 1.000)                   |                             | 50 (recall 1.000)                   |                             |