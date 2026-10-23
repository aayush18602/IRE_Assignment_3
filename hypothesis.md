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

## Part 4 — Drifting data (synthetic)

**Question.** If an index is built on one distribution and then vectors and
queries arrive from a shifted one, which index degrades, how, and does
rebuilding fix it?

**Setup.**
- **Data:** a seeded Gaussian mixture (d = 64, 50 clusters).
- **Build:** IVF-Flat and HNSW on 200K "old" vectors. The IVF centroids are
  trained on the old vectors only.
- **Drift:** then 100K "new" vectors are inserted, drawn from the same mixture
  shifted by δ along one fixed random direction. δ is measured in units of the
  old data's RMS radius (the RMS distance of old vectors from their mean).
- **Queries:** drawn from the new distribution. Ground truth is exact search
  over all 300K vectors.
- **The one thing varied:** δ ∈ {0, 0.25, 0.5, 1}. δ = 0 is the unchanged
  control: the insertions follow the same procedure but come from the old
  distribution.
- **Recovery:** for every δ, both indexes are also rebuilt from scratch on all
  300K vectors, with IVF retrained.
- **Measured:** recall at the search parameter that gives 0.95 on the control,
  QPS at recall 0.95, and for IVF the list-size imbalance and the number of
  vectors scanned per query.

**Hypothesis.**

1. **IVF-Flat gets slower, not less accurate.** The stale centroids do not
   cover the shifted region, so every new vector is assigned to the few old
   centroids on the side facing the shift, and those lists become very large.
   - A query from the new region probes exactly those lists. At a fixed nprobe
     it scans many more vectors, but it also finds most of its neighbours, so
     recall changes little and may even rise.
   - The damage shows up as cost. I expect the imbalance factor and the vectors
     scanned per query to grow steadily with δ (at least 3× the control at
     δ = 1), and QPS at recall 0.95 to fall by more than 2× at δ = 1.
2. **Retraining fixes IVF.** Rebuilding IVF on all 300K vectors puts centroids
   back where the data is, bringing its QPS at recall 0.95 back to within about
   20% of the control.
3. **HNSW is robust.** Each inserted vector is linked to its nearest existing
   vectors, including earlier new ones, so the new region gets an ordinary
   local graph however far it has moved. I expect recall at fixed efSearch to
   stay within 0.02 of the control and QPS at recall 0.95 to stay within about
   20% of a fresh rebuild, for every δ.
   - **Risk:** at δ = 1 the new region may be connected to the old graph only
     through a few links, so search entering from the old region could get
     stuck. If so, HNSW recall would drop at large δ, which would reject this
     part.
