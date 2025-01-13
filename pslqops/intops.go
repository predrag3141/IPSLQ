package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
	"math"
)

// IntOperation holds the information necessary to perform an integer operation
// on A and B. The operation on A can also be applied to H; the operation on B
// can also be applied to M. The operation on A is a row operation, and the
// operation on B is a column operation.
//
// In most cases, the row operation to perform is a permutation. When the operation
// is a permutation, PermutationOfA and PermutationOfB are populated (non-zero in
// length). Otherwise, OperationOnA and OperationOnB must be populated (non-zero
// length). It is an error to populate both the matrices and the permutations.
type IntOperation struct {
	Indices        []int   // indices of rows affected by the row operation
	OperationOnA   []int   // sub-matrix for the row operation on H and/or A
	OperationOnB   []int   // sub-matrix for the row operation on B
	PermutationOfA [][]int // cycles of the permutation for the row operation on H and/or A
	PermutationOfB [][]int // cycles of the permutation for the row operation on B
}

// NextIntOp returns the next integer row operation to perform on H or M.
// The client for NextIntOp is a strategy, so NextIntOp is exported.
//
// # If hOrM is square, it must be M and
//
// - The returned operation (if non-nil) is a column operation to be performed on M
//
// - A nil column operation indicates that the entire algorithm should be terminated.
//
// # If hOrM is not square, it must be H and
//
// - The returned operation (if non-nil) is a row operation to be performed on H.
//
//   - A nil row operation indicates that no row operation on H improves its diagonal,
//     and H has all zeroes in its last row. In this situation, it is time for the calling
//     function to set M equal to the inverse of H (excluding its all-zero bottom row)
//     and proceed with m != nil on the next call to NextIntOp.
func NextIntOp(hOrM *bigmatrix.BigMatrix) (*IntOperation, error) {
	if hOrM.NumRows() == hOrM.NumCols() {
		// hOrM is M
		return bestSwapInM(hOrM, "NextIntOp")
	}

	// hOrM is H
	const log2threshold = -20 // limits reductions using the last row to a factor of about 2^20
	diagonalRowOp, err := bestDiagonalRowOpInH(hOrM, "NextIntOp")
	if err != nil {
		return nil, err
	}
	if diagonalRowOp != nil {
		return diagonalRowOp, nil
	}
	brh, err := getBottomRightOfH(hOrM, "NextIntOp")
	if err != nil {
		return nil, err
	}
	if brh.Found {
		return brh.reduce(1<<(-log2threshold), log2threshold, "NextIntOp")
	}
	return nil, nil
}

// bestDiagonalRowOpInH returns the row swap on H is that increases the bottom-right entry
// of a 2x2 diagonal sub-matrix of H by the greatest factor; or if no row swap increases
// that, a nil  operation is returned.
func bestDiagonalRowOpInH(h *bigmatrix.BigMatrix, caller string) (*IntOperation, error) {
	// Iterate through diagonal 2x2 sub-matrices of H starting at the top left using the
	// notation where the first row of a sub-matrix is [t, 0] and the second row is [u, v].
	caller = fmt.Sprintf("%s-bestDiagonalRowOpInH", caller)
	bestScore := bignumber.NewFromInt64(0)
	var bestRowOp *IntOperation = nil
	numCols := h.NumCols()
	var t, u, v *bignumber.BigNumber
	var err error
	t, err = h.Get(0, 0)
	if err != nil {
		return nil, fmt.Errorf(
			"%s: could not get H[%d][%d]: %q", caller, 0, 0, err.Error(),
		)
	}
	t = bignumber.NewFromBigNumber(t)
	tSq := bignumber.NewFromInt64(0).Mul(t, t)
	for j := 0; j < numCols-1; j++ {
		u, err = h.Get(j+1, j)
		if err != nil {
			return nil, fmt.Errorf(
				"%s: could not get H[%d][%d]: %q", caller, j+1, j, err.Error(),
			)
		}
		u = bignumber.NewFromBigNumber(u)
		v, err = h.Get(j+1, j+1)
		if err != nil {
			return nil, fmt.Errorf(
				"%s: could not get H[%d][%d]: %q", caller, j+1, j+1, err.Error(),
			)
		}
		v = bignumber.NewFromBigNumber(v)

		// Check whether |t| needs reducing, if possible
		vSq := bignumber.NewFromInt64(0).Mul(v, v)
		if tSq.Cmp(vSq) <= 0 {
			// |t| does not need reducing
			t.Set(v)
			tSq.Set(vSq)
			continue
		}

		// |t| needs reducing. Reasons to consider the current value of j are:
		// - sqrt(u^2+v^2) < |t| (i.e., swapping reduces |t|).
		// - 3v^2 < u^2, so at least one general row operation may be better than
		//   a row swap.
		// Set flags for each way to reduce |t| and skip this j if both are false
		uSq := bignumber.NewFromInt64(0).Mul(u, u)
		threeVSq := bignumber.NewFromInt64(0).Int64Mul(3, vSq)
		swapScore := bignumber.NewFromInt64(0).Add(uSq, vSq)
		trySwap, tryGeneralRowOperations := false, false
		if swapScore.Cmp(tSq) < 0 {
			trySwap = true
		}
		if threeVSq.Cmp(uSq) < 0 {
			tryGeneralRowOperations = true
		}
		if (!trySwap) && (!tryGeneralRowOperations) {
			t.Set(v)
			tSq.Set(vSq)
			continue
		}

		// The score calculation is:
		// - swapScore = u^2 + v^2. This was computed above.
		// - generalRowOpScore = (r0 t + r1 u)^2 + (r1 v)^2 where r0 and r1 form the top row
		//   of generalRowOp.OperationOnA. This is computed below, in getGeneralRowOperation.
		// - If both swapScore and generalRowOpScore are defined, then
		//   score <- t^2/min(swapScore, generalRowOpScore)
		// - If only swapScore is defined, score <- t^2/swapScore
		// - If only generalRowOpScore is defined, score score <- t^2/generalRowOpScore
		// - If neither swapScore nor generalRowOpScore is defined, the loop continues
		var generalRowOp *IntOperation
		var generalRowOpScore *bignumber.BigNumber
		if tryGeneralRowOperations {
			generalRowOp, generalRowOpScore, err = getGeneralRowOperation(j, t, u, v, uSq, vSq, caller)
			if err != nil {
				return nil, err
			}
		}
		generalRowOpImprovesAndBeatsSwap :=
			(generalRowOpScore != nil) && (generalRowOpScore.Cmp(tSq) < 0) &&
				(generalRowOpScore.Cmp(swapScore) < 0)
		if trySwap && (!generalRowOpImprovesAndBeatsSwap) {
			// Swapping rows
			// - Improves the diagonal of H since trySwap == true
			// - Beats the best general row operation, if any
			var score *bignumber.BigNumber
			score, err = bignumber.NewFromInt64(0).Quo(tSq, swapScore)
			if err != nil {
				// swapScore is no less than the square of a diagonal element, so if
				// this was a divide-by-zero it is a catastrophic error.
				_, swapScoreAsStr := swapScore.String()
				return nil, fmt.Errorf(
					"%s: could not divide by H[%d][%d]^2+H[%d][%d]^2 = %q: %q",
					caller, j+1, j, j+1, j+1, swapScoreAsStr, err.Error(),
				)
			}
			if score.Cmp(bestScore) > 0 {
				bestScore.Set(score)
				bestRowOp = &IntOperation{
					Indices:        []int{j, j + 1},
					OperationOnA:   []int{},
					OperationOnB:   []int{},
					PermutationOfA: [][]int{{0, 1}},
					PermutationOfB: [][]int{{0, 1}},
				}
			}
			t.Set(v)
			tSq.Set(vSq)
			continue
		}

		// Swapping was not done for this value of j. If that was because a general row
		// operation improves the diagonal and beats swapping, score the general row operation.
		if generalRowOpImprovesAndBeatsSwap {
			var score *bignumber.BigNumber
			score, err = bignumber.NewFromInt64(0).Quo(tSq, generalRowOpScore)
			if err != nil {
				// generalRowOpScore is the square of a non-zero error in the estimation of t/u by
				// continued fractions. It is a catastrophic error for this to be zero.
				_, generalRowOpScoreAsStr := score.String()
				return nil, fmt.Errorf(
					"%s: could not divide by general row operation score %q: %q",
					caller, generalRowOpScoreAsStr, err.Error(),
				)
			}
			if score.Cmp(bestScore) > 0 {
				bestScore.Set(score)
				bestRowOp = generalRowOp
			}
			t.Set(v)
			tSq.Set(vSq)
			continue
		}

		// Neither swap nor general row operation improves the diagonal.
		t.Set(v)
		tSq.Set(vSq)
	}
	return bestRowOp, nil
}

// bestSwapInM returns a column operation for swapping a pair of columns of M
// such that
//
//   - The swap, followed by corner removal, decreases the diagonal entry in
//     the right-hand column
//
// - The right-hand column is as close to the right of M as possible
//
// - The left-hand column is as close to the right-hand column as possible
//
// If no column swap in M meets the first criterion, bestSwapInM returns nil.
// If an error occurs, bestSwapInM returns the error and a nil IntOperation.
func bestSwapInM(m *bigmatrix.BigMatrix, caller string) (*IntOperation, error) {
	// Initialize mSq, a float64 copy of M but with entries squared
	caller = fmt.Sprintf("%s-bestSwapInM", caller)
	numRows := m.NumRows()
	columnNorms := make([]float64, numRows)
	mSq, err := m.AsFloat64() // currently not the squares of the entries but soon will be
	if err != nil {
		return nil, fmt.Errorf("%s: could not convert M to float64: %q", caller, err)
	}
	for i := 0; i < numRows; i++ {
		for j := 0; j <= i; j++ {
			entrySq := mSq[i*numRows+j] * mSq[i*numRows+j]
			mSq[i*numRows+j] = entrySq
			columnNorms[j] += entrySq
		}
	}

	// The right index of the column pair returned should be as far to the right in M as
	// possible, so start at the right-most column of M.
	for j1 := numRows - 1; 0 < j1; j1-- {
		// Iterate through columns to the left of j1 looking for the shortest column. To save
		// time on updates, start from the right because the most likely case is that the minimum
		// column is near j1, since those are the shortest candidate columns.
		minColumnNorm := columnNorms[j1]
		columnWithMinNorm := j1
		for j0 := j1 - 1; 0 <= j0; j0-- {
			// Update min column norm
			if columnNorms[j0] < minColumnNorm {
				minColumnNorm = columnNorms[j0]
				columnWithMinNorm = j0
			}

			// Update column norms for the next iteration of j1. This is only necessary if
			// there might be no column swap involving j1.
			if columnWithMinNorm == j1 {
				// It still might be necessary to continue the j1 loop, so update this column
				// norm for the next iteration of j1.
				columnNorms[j0] -= mSq[j1*numRows+j0]
			}
		}
		if columnWithMinNorm < j1 {
			return &IntOperation{
				Indices:        []int{columnWithMinNorm, j1},
				OperationOnA:   []int{},
				OperationOnB:   []int{},
				PermutationOfA: [][]int{{0, 1}},
				PermutationOfB: [][]int{{0, 1}},
			}, nil
		}
	}

	// No column swap increases the bottom-right entry of a pair of diagonal elements
	return nil, nil
}

// getGeneralRowOperation returns the non-swap row operation with matrix R that achieves the best
// score when it right-multiplies the sub-matrix T of H, [[t, 0], [u, v]]; along with the score
// of R. The score of R is RTQ[0][0]^2, where Q is a Givens rotation that zeroes out RTQ[0][1]. The
// lower the score, the better R is as a candidate row operation. If no score is below 1, however,
// nil is returned as the row operation. The returned score is always non-nil, but by default, in
// the case where there are no general row operations to try, it is a ridiculous math.MaxInt64.
//
// The best score found here will ultimately be compared to STQ'[0][0], where S is a row swap and
// Q' is the Givens rotation that zeroes out ST[0][1].
//
// Since Givens rotations performed with a right-multiply preserve row lengths, and RTQ[0][1] = 0,
// RTQ[0][0] = sqrt((RT)[0][0]^2+(RT)[0][1]^2). The score of R is just the square of that entry of RTQ.
func getGeneralRowOperation(
	j int,
	t, u, v, uSq, vSq *bignumber.BigNumber,
	caller string,
) (*IntOperation, *bignumber.BigNumber, error) {
	caller = fmt.Sprintf("%s-getGeneralRowOperation", caller)
	var err error
	retScore := bignumber.NewFromInt64(math.MaxInt64)
	var retRowOp *IntOperation
	var uSqOverVSq *bignumber.BigNumber
	uSqOverVSq, err = bignumber.NewFromInt64(0).Quo(uSq, vSq)
	if err != nil {
		_, vSqAsStr := vSq.String()
		return nil, nil, fmt.Errorf(
			"%s: could not divide by %s: %q", caller, vSqAsStr, err.Error(),
		)
	}
	uSqOverVSqPlusOne := bignumber.NewFromInt64(0).Add(uSqOverVSq, bignumber.NewFromInt64(1))
	maxBSqPtr := uSqOverVSqPlusOne.Int64RoundTowardsZero()
	if maxBSqPtr == nil {
		// Returning a nil score and nil error indicates no error, but no general row
		// operation either.
		return nil, nil, nil
	}

	// maxBAsInt is set to the maximum possible value of b in a general row
	// operation with rows [a, b] and [-w, v] that can possibly outperform a row
	// swap, as described in the section, "General Row Operations", in the README.
	var maxBAsFloat64 float64
	var maxBAsInt int
	maxBAsFloat64 = math.Sqrt(float64(*maxBSqPtr))
	if maxBAsFloat64 > float64(math.MaxInt) {
		// Returning a nil score and nil error indicates no error, but no general row
		// operation either.
		return nil, nil, nil
	}
	maxBAsInt = 1 + int(maxBAsFloat64)  // Adding 1 provides a margin of safety
	t0 := bignumber.NewFromBigNumber(t) // Modified by reducePair below
	u0 := bignumber.NewFromBigNumber(u) // Modified by reducePair below
	err = reducePair(
		t0, u0, maxBAsInt, "GetNextRowOperation",
		func(r []int) bool {
			// Keep track of which matrix, r, has the smallest Euclidean norm in
			// the top row of its product with the matrix, [[t, 0], [u, v]].
			//
			// Because this anonymous callback is called from reducePair after t0 and
			// u0 have been combined to make the result -- t0 -- small, t0 is already
			// the first entry in this top row. The second entry in the top row is
			// r[0][1] v, which has not been updated since reducePair reduces t against u,
			// not against v.
			r0tSq := bignumber.NewFromInt64(0).Mul(t0, t0)
			r1v := bignumber.NewFromInt64(0).Int64Mul(int64(r[1]), v)
			r1vSq := bignumber.NewFromInt64(0).Mul(r1v, r1v)
			thisScore := bignumber.NewFromInt64(0).Add(r0tSq, r1vSq)
			if (thisScore.Cmp(retScore) < 0) && (r[0] != 0) {
				// A non-swap (r[0] != 0) has been found that improves on the current best
				retScore.Set(thisScore)
				det := r[0]*r[3] - r[1]*r[2]
				retRowOp = &IntOperation{
					Indices:        []int{j, j + 1},
					OperationOnA:   []int{r[0], r[1], r[2], r[3]},
					OperationOnB:   []int{det * r[3], -det * r[1], -det * r[2], det * r[0]},
					PermutationOfA: [][]int{},
					PermutationOfB: [][]int{},
				}
			}

			// Return from reducePair if larger values of b as described above
			// cannot possibly outperform a row swap.
			if (r[1] > maxBAsInt) || (-r[1] > maxBAsInt) {
				// r[1] is r[0][1] = b when r is considered to be the matrix, [[a, b], [-w, v]],
				// as described in the section, "General Row Operations" in the README. b, a.k.a. r[1],
				// has exceeded the maximum possible value, maxBAsInt, for which any 2x2 matrix with
				// that upper-right entry could outperform a row swap. Returning true here terminates
				// reducePair, since a row swap would outperform anything found by continuing.
				return true
			}

			// See comment for "if r[1] > maxBAsInt". Since b, a.k.a. r[1], has not yet
			// exceeded maxBAsInt, the possibility still exists of outperforming a row swap
			// in the next iteration. Returning false here causes reducePair to keep
			// reducing t and u to see if that happens.
			return false
		},
	)
	return retRowOp, retScore, nil
}

// performRowOp left-multiplies X in-place by an expansion, R^-1, of intOperation.OperationOnA
// into the numRows x numRows identity matrix, injecting entries of intOperation.OperationOnA
// into entries of the identity indicated by the variable, indices.
func (io *IntOperation) performRowOp(x *bigmatrix.BigMatrix, caller string) error {
	if io.isPermutation() {
		if (len(io.Indices) != 2) || (len(io.PermutationOfA) != 1) || (len(io.PermutationOfA[0]) != 2) {
			return fmt.Errorf(
				"%s: non-swap permutation with indices %v and cycles %v is not supported",
				caller, io.Indices, io.PermutationOfA,
			)
		}

		// Assume that the cycle in io.PermutationOfA is (0,1). Unless and until permutations
		// other than swaps are supported, the cycle does not add information to what is already
		// in io.Indices, and can be ignored.
		return x.PermuteRows([][]int{{io.Indices[0], io.Indices[1]}})
	}
	numIndices := len(io.Indices)
	xNumCols := x.NumCols()
	caller = fmt.Sprintf("%s-performRowOp", caller)

	// Compute the entries of X that left-multiplying by the row operation modifies
	newSubMatrixOfX := make([]*bignumber.BigNumber, numIndices*xNumCols)
	cursor := 0
	for i := 0; i < numIndices; i++ {
		for j := 0; j < xNumCols; j++ {
			newEntry := bignumber.NewFromInt64(0)
			for k := 0; k < numIndices; k++ {
				xKJ, err := x.Get(io.Indices[k], j)
				if err != nil {
					return fmt.Errorf(
						"%s: could not get X[%d][%d]: %q",
						caller, io.Indices[k], j, err.Error(),
					)
				}
				rIK := int64(io.OperationOnA[i*numIndices+k])
				if (rIK == 0) || xKJ.IsSmall() {
					continue
				}
				newEntry.Int64MulAdd(rIK, xKJ)
			}
			newSubMatrixOfX[cursor] = newEntry
			cursor++
		}
	}

	// Replace the affected entries of X, consisting of numIndices rows
	cursor = 0
	for i := 0; i < numIndices; i++ {
		for j := 0; j < xNumCols; j++ {
			err := x.Set(io.Indices[i], j, newSubMatrixOfX[cursor])
			if err != nil {
				return fmt.Errorf(
					"%s: error setting x[%d][%d]: %q", caller, i, j, err.Error(),
				)
			}
			cursor++
		}
	}
	return nil
}

// performColumnOp right-multiplies X in-place by an expansion, R^-1, of intOperation.OperationOnB
// into the numRows x numRows identity matrix, injecting entries of intOperation.OperationOnB
// into entries of the identity indicated by the variable, indices.
func (io *IntOperation) performColumnOp(x *bigmatrix.BigMatrix, caller string) error {
	if io.isPermutation() {
		if (len(io.Indices) != 2) || (len(io.PermutationOfB) != 1) || (len(io.PermutationOfB[0]) != 2) {
			return fmt.Errorf(
				"%s: non-swap permutation with indices %v and cycles %v is not supported",
				caller, io.Indices, io.PermutationOfA,
			)
		}

		// Assume that the cycle in io.PermutationOfB is (0,1). Unless and until permutations
		// other than swaps are supported, the cycle does not add information to what is already
		// in io.Indices, and can be ignored.
		return x.PermuteColumns([][]int{{io.Indices[0], io.Indices[1]}})
	}
	numIndices := len(io.Indices)
	xNumRows := x.NumRows()
	caller = fmt.Sprintf("%s-performColumnOp", caller)

	// Compute the entries of X that right-multiplying by the colum operation modifies
	newSubMatrixOfX := make([]*bignumber.BigNumber, xNumRows*numIndices)
	cursor := 0
	for i := 0; i < xNumRows; i++ {
		for j := 0; j < numIndices; j++ {
			newEntry := bignumber.NewFromInt64(0)
			for k := 0; k < numIndices; k++ {
				xIK, err := x.Get(i, io.Indices[k])
				//_, xIKAsStr := xIK.String()
				if err != nil {
					return fmt.Errorf(
						"%s: could not get X[%d][%d]: %q",
						caller, i, io.Indices[k], err.Error(),
					)
				}
				rKJ := int64(io.OperationOnB[k*numIndices+j])
				if (rKJ == 0) || xIK.IsSmall() {
					continue
				}
				newEntry.Int64MulAdd(rKJ, xIK)
			}
			newSubMatrixOfX[cursor] = newEntry
			cursor++
		}
	}

	// Replace the entries of X, consisting of numIndices columns
	cursor = 0
	for i := 0; i < xNumRows; i++ {
		for j := 0; j < numIndices; j++ {
			err := x.Set(i, io.Indices[j], newSubMatrixOfX[cursor])
			if err != nil {
				return fmt.Errorf(
					"%s: could not set X[%d][%d]: %q",
					caller, i, io.Indices[j], err.Error(),
				)
			}
			cursor++
		}
	}
	return nil
}

// validateIndices performs a quick check on io.Indices
func (io *IntOperation) validateIndices(numRows int, caller string) error {
	// Indices are needed for corner removal, even when the row operation is a permutation
	caller = fmt.Sprintf("%s-validateIndices", caller)
	numIndices := len(io.Indices)
	if numIndices < 2 {
		return fmt.Errorf(
			"%s: length of indices must be zero at least 2 but is 1", caller,
		)
	}

	// Indices should be strictly increasing within bounds dictated by numRows and numCols
	if io.Indices[0] < 0 {
		return fmt.Errorf("%s: io.Indices[0] = %d is negative", caller, io.Indices[0])
	}
	if numRows <= io.Indices[numIndices-1] {
		// No index can be numRows or more, but since indices is an increasing array, the
		// only index to check is the last one (and it failed)
		return fmt.Errorf(
			"%s: numRows = %d <= %d = io.Indices[%d]",
			caller, numRows, io.Indices[numIndices-1], numIndices-1,
		)
	}
	for i := 1; i < numIndices; i++ {
		if io.Indices[i] <= io.Indices[i-1] {
			return fmt.Errorf("%s: io.Indices %v is not stricty increasing", caller, io.Indices)
		}
	}
	return nil
}

// validateAll performs a quick validation on a IntOperation instance.  PermutationOfA
// and PermutationOfB are not validated, as they should be set by the trusted constructor,
// NewFromPermutation.
//
// numRows and numCols refer to the dimensions of H.
func (io *IntOperation) validateAll(numRows int, caller string) error {
	caller = fmt.Sprintf("%s-validateAll", caller)

	// Check compatibility of matrix lengths
	numIndices := len(io.Indices)
	if len(io.OperationOnA) != len(io.OperationOnB) {
		return fmt.Errorf(
			"%s: mismatched lengths %d and %d of io.OperationOnH and io.OperationOnB",
			caller, len(io.OperationOnA), len(io.OperationOnB),
		)
	}
	if (len(io.OperationOnA) != numIndices*numIndices) && (len(io.OperationOnA) != 0) {
		return fmt.Errorf(
			"%s: non-zero length %d of io.OperationOnH is incompatible with numIndices = %d",
			caller, numIndices, len(io.OperationOnA),
		)
	}

	// Check compatibility of matrix and permutation lengths against each other
	if (len(io.OperationOnA) != 0) && (len(io.PermutationOfA) != 0) {
		return fmt.Errorf("%s: both matrix and permutation are populated", caller)
	}
	if (len(io.OperationOnA) == 0) && (len(io.PermutationOfA) == 0) {
		return fmt.Errorf("%s: neither matrix nor permutation is populated", caller)
	}

	// Indices must still be validated
	return io.validateIndices(numRows, caller)
}

// equals returns whether io is equal to other. In the case where io and other contain
// permutations, equality means the cycles in io and other come in the same order,
// though the starting point of cycles in io can differ from their counterparts in other.
func (io *IntOperation) equals(other *IntOperation) bool {
	// Equality of Indices
	if len(io.Indices) != len(other.Indices) {
		return false
	}
	for i := 0; i < len(io.Indices); i++ {
		if io.Indices[i] != other.Indices[i] {
			return false
		}
	}

	// Equality of OperationOnA
	if len(io.OperationOnA) != len(other.OperationOnA) {
		return false
	}
	for i := 0; i < len(io.OperationOnA); i++ {
		if io.OperationOnA[i] != other.OperationOnA[i] {
			return false
		}
	}

	// Equality of OperationOnB
	if len(io.OperationOnB) != len(other.OperationOnB) {
		return false
	}
	for i := 0; i < len(io.OperationOnB); i++ {
		if io.OperationOnB[i] != other.OperationOnB[i] {
			return false
		}
	}

	// Equality of PermutationOfH and PermutationOfB
	if !permutationsAreEqual(io.PermutationOfA, other.PermutationOfA) {
		return false
	}
	return permutationsAreEqual(io.PermutationOfB, other.PermutationOfB)
}

// isPermutation returns whether io.PermutationOfA has non-zero length
func (io *IntOperation) isPermutation() bool {
	return len(io.PermutationOfA) != 0
}

func permutationsAreEqual(x [][]int, y [][]int) bool {
	xLen := len(x)
	if len(y) != xLen {
		return false
	}
	for i := 0; i < xLen; i++ {
		cycleLen := len(x[i])
		if cycleLen != len(y[i]) {
			return false
		}
		equalsAtSomeOffset := false
		for offset := 0; offset < cycleLen; offset++ {
			equalsAtThisOffset := true
			for j := 0; j < cycleLen; j++ {
				offsetOfJ := j + (offset % cycleLen)
				if x[i][offsetOfJ] != y[i][offsetOfJ] {
					equalsAtThisOffset = false
					break
				}
			}
			if equalsAtThisOffset {
				equalsAtSomeOffset = true
				break
			}
		}
		if !equalsAtSomeOffset {
			return false
		}
	}
	return true
}
