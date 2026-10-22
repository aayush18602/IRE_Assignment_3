| dataset   | index        | config                                |   build_s |   index_MB |   bytes_per_vec | note                                                       |
|:----------|:-------------|:--------------------------------------|----------:|-----------:|----------------:|:-----------------------------------------------------------|
| sift      | HNSW         | M=16, efC=200, efSearch=64            |      54.5 |        626 |             656 |                                                            |
| sift      | IVF-Flat     | nlist=4096, nprobe=48                 |      15.3 |        498 |             522 |                                                            |
| sift      | IVF-PQ       | nlist=1024, m=32, nbits=8, nprobe=192 |       6.1 |         39 |              41 | 0.95 not reached; most accurate config (test recall 0.731) |
| sift      | Exact (Flat) |                                       |       0   |        488 |             512 |                                                            |
| glove     | HNSW         | M=32, efC=200, efSearch=384           |     113.1 |        759 |             672 |                                                            |
| glove     | IVF-Flat     | nlist=4096, nprobe=384                |      14.6 |        462 |             409 |                                                            |
| glove     | IVF-PQ       | nlist=1024, m=50, nbits=8, nprobe=512 |       5.5 |         66 |              58 | 0.95 not reached; most accurate config (test recall 0.866) |
| glove     | Exact (Flat) |                                       |       0   |        451 |             400 |                                                            |