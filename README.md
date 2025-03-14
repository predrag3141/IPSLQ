# Why Two PSLQ Repositories
This repository, IPSLQ, has a sibling, [PSLQ](https://github.com/predrag3141/pslq). This sibling repository contains a detailed README on PSLQ and research on modifying PSLQ to be used in cryptanalysis. But the code in the PSLQ repository incorporates many tried and abandoned ideas like "gentle reduction". This code in this repository replaces the abandoned code in [PSLQ](https://github.com/predrag3141/pslq). The README in PSLQ will eventually be curated and moved here as well. But in the meantime, the README in [PSLQ](https://github.com/predrag3141/pslq) remains the best place to go for an understanding of the modifications and improvements both in the code there and in the code here.

# Inverse PSLQ (IPSLQ)
The code in this repository builds on the ideas in the sibling repository, [PSLQ](https://github.com/predrag3141/pslq). It implements a technique for continuing PSLQ beyond the point where adjacent row swaps in H no longer improve its diagonal or place zeroes in its last row. By the time this happens, H has only zeroes in its last row, and can be considered an invertible square matrix by ignoring the last (all-zero) row. The inverse of H, called M in the code, accepts column operations in lieu of the row operations in H. Unlike H, which cannot easily be improved with non-adjacent row swaps, M can be improved with non-adjacent column swaps (column operations on M being the equivalent of row operations on H).

The reason non-adjacent column swaps improve M, but non-adjacent row swaps would not improve H, is that in M, the target column is on thr right-hand side of the matrix. If the right-most column of M is shortened, the best solution in B, also in the right-most column, automatically improves! Contrast this to the situation in H: If you swap non-adjacent rows, you have to calculate the entire corner removal to find out what the effect is on the bottom of the two rows.

# Trying it Out
To try out this library, clone the repository and run the unit test in `knownanswertest`:
```bash
cd knownanswertest
go test -v -timeout 2h
```

This unit test creates a random 50-long input with a known, short, orthogonal integer vector, and runs PSLQ against it. Once PSLQ finds the known answer, the known answer unit test writes "H!" or (more likely) "M!" to console every hundredth iteration. Prior to finding the known answer, the unit test instead writes "H?" or "M?". As you might guess, the letter written, "H" or "M", indicates what matrix PSLQ is using during the iteration when one of those letters is written. Once PSLQ switches from H to M, it sticks with M until M cannot be improved with non-adjacent column swaps.
