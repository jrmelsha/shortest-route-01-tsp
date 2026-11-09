# shortest-route-01 entry

A make.fun `shortest-route-01` entry: Euclidean TSP (TSPLIB EUC_2D) on eight city sets.

`main.ts` (Node 22, standard library only):

1. Candidate lists: the 8 nearest cities plus the 2 nearest in each quadrant around a city.
2. Nearest-neighbour start, then 2-opt and Or-opt (segments of 1 to 3 cities, either orientation) driven by a
   work queue of cities (don't-look bits).
3. Iterated local search: each kick swaps two adjacent short segments, the local search repairs around it, and a
   kick that lengthens the tour is undone from a reversal journal.

The search is bounded by fixed kick counts and seeded with constants, so two runs write the same bytes.

Run locally: `mkdir -p out && MAKEFUN_INPUT_DIR=<dir> MAKEFUN_OUTPUT_DIR=out node main.ts`.
