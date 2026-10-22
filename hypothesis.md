# Hypotheses

Each section is committed to git before the experiment it describes is run.

## Part 3 — Query difficulty on GloVe-100

**Setup.** For each GloVe test query, measure difficulty with its *relative
contrast* (RC): the mean Euclidean distance from the query to the base vectors
(estimated on a fixed random sample of 10,000 base vectors) divided by the
distance to its 10th true nearest neighbour. Low RC means the true neighbours
are barely closer than a random vector, i.e. a hard query. Split the 5,000
test queries into four equal-size bins by RC quartile. Measure the recall@10 of
each index in each bin, at the operating point chosen on the tune queries in
Part 2 for overall recall 0.95 (for IVF-PQ, which cannot reach 0.95, its most
accurate configuration).

**Hypothesis.** Recall drops steadily from the easiest to the hardest bin for
every index, so the average of about 0.95 hides a much worse tail. I expect:

1. **IVF-Flat** to show a large drop (hardest bin at least 0.10 below the
   easiest). When neighbours are barely closer than everything else, they are
   spread over many Voronoi cells around the query, so a fixed nprobe that is
   enough for easy queries misses cells that hold true neighbours.
2. **HNSW** to drop too, but less than IVF-Flat. Greedy graph search follows
   distance gradients, and with a flat distance profile it can stop at a local
   minimum. A large efSearch still explores widely, which should soften the drop.
3. **IVF-PQ** to have the lowest recall in every bin and the largest absolute
   drop. Its quantisation error is roughly fixed per vector, so it swamps the
   small gap between the true 10th neighbour and the next candidates on
   low-contrast queries.

If recall is flat across the bins, relative contrast is not what makes GloVe
hard for these indexes, and the hypothesis is rejected.
