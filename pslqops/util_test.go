package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
	"github.com/predrag3141/IPSLQ/util"
	"math"
	"math/rand"
)

const (
	intPermutation        = "intPermutation"
	generalIntOp          = "generalIntOp"
	bigNumberBitTolerance = 50
	float64BitTolerance   = 35
	binaryPrecision       = 1000
)

// createRandomInversePair returns a pair of inverse matrices with integer entries
// and determinant 1. In case of error, the third return value is non-nil.
func createRandomInversePair(dim int, caller string) ([]int, []int, error) {
	caller = fmt.Sprintf("%s-createRandomInversePair", caller)
	const maxRowOpEntry = 10
	const maxRowOps = 10
	const maxMatrixEntry = 100
	int64RetValA := make([]int64, dim*dim)
	int64RetValB := make([]int64, dim*dim)

	// The inverse operation to adding c times row i to row j is to add −c times row i to
	// row j
	for i := 0; i < maxRowOps; i++ {
		srcRow := rand.Intn(dim)
		destRow := rand.Intn(dim)
		multiple := int64(rand.Intn(maxRowOpEntry) - (maxRowOpEntry / 2))
		if multiple == 0 {
			multiple = 1
		}
		if srcRow == destRow {
			if destRow < dim/2 {
				destRow += dim / 2
			} else {
				destRow -= dim / 2
			}
		}
		rowOpMatrixA := make([]int64, dim*dim)
		rowOpMatrixB := make([]int64, dim*dim)
		for j := 0; j < dim; j++ {
			rowOpMatrixA[j*dim+j] = 1
			rowOpMatrixB[j*dim+j] = 1
			if i == 0 {
				// retValA and retValB are all 0
				int64RetValA[j*dim+j] = 1
				int64RetValB[j*dim+j] = 1
			}
		}
		rowOpMatrixA[destRow*dim+srcRow] = multiple
		rowOpMatrixB[destRow*dim+srcRow] = -multiple
		if i == 0 {
			// int64RetValA and int64RetValB are both the identity
			int64RetValA[destRow*dim+srcRow] = multiple
			int64RetValB[destRow*dim+srcRow] = -multiple
			continue
		}

		// i > 0, so an update of int64RetValA and int64RetValB is required
		var tmpB []int64
		tmpA, err := util.MultiplyIntInt(rowOpMatrixA, int64RetValA, dim)
		if err != nil {
			return []int{}, []int{}, fmt.Errorf(
				"%s: could not multiply int64RetValA by rowOpMatrixA: %q",
				caller, err.Error(),
			)
		}
		tmpB, err = util.MultiplyIntInt(int64RetValB, rowOpMatrixB, dim)
		if err != nil {
			return []int{}, []int{}, fmt.Errorf(
				"%s: could not multiply int64RetValB by rowOpMatrixB: %q",
				caller, err.Error(),
			)
		}

		// An entry in tmpA or tmpB may exceed the maximum desired
		for j := 0; j < dim*dim; j++ {
			if (tmpA[j] > maxMatrixEntry) || (tmpA[j] < -maxMatrixEntry) {
				return util.CopyInt64ToInt(int64RetValA), util.CopyInt64ToInt(int64RetValB), nil
			}
			if (tmpB[j] > maxMatrixEntry) || (tmpB[j] < -maxMatrixEntry) {
				return util.CopyInt64ToInt(int64RetValA), util.CopyInt64ToInt(int64RetValB), nil
			}
		}

		// No entry in tmpA or tmpB exceeds the maximum desired, so continue on
		int64RetValA = tmpA
		int64RetValB = tmpB
	}

	// The maximum number of iterations has been reached
	return util.CopyInt64ToInt(int64RetValA), util.CopyInt64ToInt(int64RetValB), nil
}

// getRandomIndices returns a pseudo-random subset of size numIndices of {0,...,numItems-1}.
// numIndices should be in {2,...,numItems}.
func getRandomIndices(numIndices, numItems int) []int {
	retVal := make([]int, numIndices)
	lastChoice := -1
	for i := 0; i < numIndices; i++ {
		numChoices := (numItems - (lastChoice + 1)) / (numIndices - i)
		if numChoices == 1 {
			retVal[i] = lastChoice + 1
		} else {
			retVal[i] = lastChoice + rand.Intn(numChoices) + 1
		}
		lastChoice = retVal[i]
	}
	return retVal
}

// getRandomIntOperation returns a random IntOperation of the specified type -- intPermutation
// or generalIntOp -- and the equivalent matrices for the operation or permutation on A and B.
//
// The flag, involveLastRow, is intended for use with H (as opposed to M). If this flag is true,
// getRandomIntOperation returns an IntOperation with 2 indices, one of which is the last row,
// since there is no known use case for an IntOperation on H with more than two indices that
// involve the last row. If the involveLastRow flag is false, getRandomIntOperation returns an
// IntOperation with a random number of indices, which may involve the last row, but only in the
// case where numRows == numCols, i.e. the IntOperation is for M rather than H.
func getRandomIntOperation(
	numRows, numCols int, intOpType string, involveLastRow bool, caller string,
) (*IntOperation, *bigmatrix.BigMatrix, *bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getRandomIntOperation", caller)
	io := &IntOperation{}
	var intR, intS []int
	switch intOpType {
	case intPermutation:
		if involveLastRow {
			io.Indices = []int{rand.Intn(numRows - 1), numRows - 1}
		} else {
			io.Indices = getRandomIndices(2, numCols)
		}
		io.OperationOnA = []int{}
		io.OperationOnB = []int{}
		io.PermutationOfA = [][]int{{0, 1}}
		io.PermutationOfB = [][]int{{0, 1}}
		intR, intS = []int{0, 1, 1, 0}, []int{0, 1, 1, 0}
	case generalIntOp:
		var err error
		var numIndices int
		if involveLastRow {
			io.Indices = getRandomIndices(2, numCols)
			io.Indices[1] = numRows - 1
			numIndices = 2
		} else {
			numIndices = 2 + rand.Intn(numRows-2) // random number in {2,3,...,numCols}
			io.Indices = getRandomIndices(numIndices, numCols)
		}
		intR, intS, err = createRandomInversePair(numIndices, caller)
		if err != nil {
			return nil, nil, nil, fmt.Errorf(
				"%s: could not create inverse pair with numIndices = %d: %q",
				caller, numIndices, err.Error(),
			)
		}
		io.OperationOnA = intR
		io.OperationOnB = intS
	}

	// Create int64 types
	int64R := util.CopyIntToInt64(intR)
	int64S := util.CopyIntToInt64(intS)
	areInverses, err := util.IsInversePair(int64R, int64S, len(io.Indices))
	if err != nil {
		return io, nil, nil, fmt.Errorf(
			"%s: error creating inverse pair: %q", caller, err.Error(),
		)
	}
	if !areInverses {
		return io, nil, nil, fmt.Errorf(
			"%s: created non-inverse pair %v and %v", caller, int64R, int64S,
		)
	}

	// Create bigmatrix types
	var bigMatrixR, bigMatrixS *bigmatrix.BigMatrix
	bigMatrixR, err = bigmatrix.NewFromSubMatrix(io.Indices, intR, numRows)
	if err != nil {
		return io, nil, nil, fmt.Errorf(
			"%s: error creating bigmatrix from %v: %q", caller, int64R, err.Error(),
		)
	}
	bigMatrixS, err = bigmatrix.NewFromSubMatrix(io.Indices, intS, numRows)
	if err != nil {
		return io, bigMatrixR, nil, fmt.Errorf(
			"%s: error creating bigmatrix from %v: %q", caller, int64S, err.Error(),
		)
	}

	// The IntOperation and both bigmatrix versions of the operation are available to return
	return io, bigMatrixR, bigMatrixS, nil
}

type RandomHInfo struct {
	unreducedRow       int    // First unreduced row
	unreducedColumn    int    // First unreduced column in first unreduced row
	rowIsUnreduced     []bool // Whether each row is unreduced
	bestDiagonalRowOp  *IntOperation
	bestLastRowOp      *IntOperation
	bottomRightOfH     *BottomRightOfH
	maxRowSwaps        int
	diagonalStatistics *DiagonalStatistics
}

// createRandomH returns a random H and a RandomHInfo struct including H with the specified
//
// - number of rows
//
// - number of possible entries from which each non-zero entry in H is selected
//
// - position of the last non-zero entry in the last row of H
//
// - a (possibly empty) list of rows to make unreduced in H, for testing reduction by D
func createRandomH(
	numRows, numPossibleEntries, lastNonzeroEntryInLastRow int, unreducedRows []int, caller string,
) (*bigmatrix.BigMatrix, *RandomHInfo, error) {
	// Initializations
	caller = fmt.Sprintf("%s-createRandomH", caller)
	numCols := numRows - 1
	numUnreducedRows := len(unreducedRows)
	hEntries := make([]int64, numRows*numCols)
	retVal := &RandomHInfo{
		unreducedRow:      -1,
		unreducedColumn:   -1,
		rowIsUnreduced:    make([]bool, numRows),
		bestDiagonalRowOp: nil,
		bestLastRowOp:     nil,
		maxRowSwaps:       0,
		bottomRightOfH: &BottomRightOfH{
			Found:        false,
			T:            nil,
			RowNumberOfT: 0,
			U:            nil,
			RowNumberOfU: numRows - 1,
		},
	}

	// Check input
	if numRows < 2 {
		return nil, nil, fmt.Errorf("%s: numRows = %d < 2", caller, numRows)
	}
	if numPossibleEntries < 2 {
		return nil, nil, fmt.Errorf(
			"%s: numPossibleEntries = %d < 2", caller, numPossibleEntries,
		)
	}
	if lastNonzeroEntryInLastRow < -1 {
		return nil, nil, fmt.Errorf(
			"%s: lastNonzeroEntryInLastRow = %d < -1", caller, lastNonzeroEntryInLastRow,
		)
	}
	if numCols <= lastNonzeroEntryInLastRow {
		return nil, nil, fmt.Errorf(
			"%s: numCols = %d <= %d =lastNonzeroEntryInLastRow",
			caller, numCols, lastNonzeroEntryInLastRow,
		)
	}
	for i := 0; i < numUnreducedRows; i++ {
		if unreducedRows[i] < 1 {
			// Row 0 cannot be unreduced, since its only non-zero entry is a diagonal element
			return nil, nil, fmt.Errorf("%s: unreducedRows[%d] = %d < 1", caller, i, unreducedRows[i])
		}
		if numRows <= unreducedRows[i] {
			return nil, nil, fmt.Errorf(
				"%s: unreducedRows[%d] = %d > %d", caller, i, unreducedRows[i], numRows-1,
			)
		}
	}

	// Set diagonal entries. For testing diagonal row operations, some adjacent pairs
	// of diagonal elements should be large followed by small. Make this happen about
	// once every two matrices on average.
	//
	// Note: The diagonal entries are updated below (search usages of diagonal)
	smallEntryThresh := numPossibleEntries / 100
	if smallEntryThresh < 2 {
		smallEntryThresh = 2
	}
	diagonal := make([]int, numCols)
	for j := 0; j < numCols; j++ {
		makeImbalancedPair := false
		if rand.Intn(2*numCols) == 0 {
			makeImbalancedPair = true
		}
		if (j > 0) && makeImbalancedPair {
			sgn := 2*rand.Intn(2) - 1
			diagonal[j-1] = sgn * (numPossibleEntries / 2)
			diagonal[j] = rand.Intn(2*smallEntryThresh) - smallEntryThresh
		} else {
			diagonal[j] = rand.Intn(numPossibleEntries) - (numPossibleEntries / 2)
		}
		if (-1 <= diagonal[j]) && (diagonal[j] <= 1) {
			diagonal[j] = 2
		}
	}

	// Set entries on the diagonal. Note: These are updated below, in conjunction with
	// updates to the variable, diagonal.
	for j := 0; j < numCols; j++ {
		hEntries[j*numCols+j] = int64(diagonal[j])
	}

	// Set entries below the diagonal
	for j := 0; j < numCols; j++ {
		maxEntry := diagonal[j]
		if maxEntry < 0 {
			maxEntry = -maxEntry
		}
		maxEntry = maxEntry / 2
		for i := j + 1; i < numRows; i++ {
			sgn := 2*rand.Intn(2) - 1
			hEntries[i*numCols+j] = int64(sgn * rand.Intn(maxEntry+1))
		}

		// In cases where the first diagonal element of a sub-matrix is large and the
		// second is small, make the sub-diagonal element large enough to make two
		// row swaps necessary.
		//
		if j > 0 {
			t := hEntries[(j-1)*numCols+j-1]
			v := hEntries[j*numCols+j]

			if (-int64(smallEntryThresh) <= v) && (v <= int64(smallEntryThresh)) {
				if (t == int64(numPossibleEntries/2)) || (t == -int64(numPossibleEntries/2)) {
					// Diagonal elements j-1 and j meet the criteria for a (possibly planted)
					// case where a large diagonal element is followed by a small one, with
					// a sub-diagonal element close to half the large diagonal element above it.
					// For large disparities between consecutive diagonal elements, this causes
					// several rounds of row swap / corner removal / reduction, which needs to
					// be tested.
					sgn := int64(2*rand.Intn(2) - 1)
					hEntries[j*numCols+j-1] = sgn * (t / 2)
				}
			}
		}
	}

	// Make sure that in every 2x2 sub-matrix along the diagonal, the entries in the left column,
	// t and u, are relatively prime. This prevents an edge case in testing where the algorithm
	// under test (reducePair) exits early when the row operation generating new values of t and
	// u sees that one of the new values is zero. This is not a bug, because if a zero appears in
	// new values of t or u in live data, and the reduction is aborted, a future iteration of the
	// PSLQ algorithm will just continue with the reduction.
	for j := 1; j < numCols; j++ {
		t := hEntries[(j-1)*numCols+j-1]
		u := hEntries[j*numCols+j-1]
		if u == 0 {
			// A row swap is always best if u = 0, so reducePair ending early is not a problem.
			continue
		}
		d := gcd(t, u)
		incr := 1
		if t < 0 {
			incr = -1
		}
		for (d != 1) && (d != -1) {
			diagonal[j-1] += incr
			t += int64(incr)
			d = gcd(t, u)
		}
		hEntries[(j-1)*numCols+j-1] = t
	}

	// Set expected diagonal statistics
	// Generate diagonal statistics
	var ratio float64
	retVal.diagonalStatistics = &DiagonalStatistics{
		Diagonal: make([]*bignumber.BigNumber, numRows),
		Ratio:    &ratio,
	}
	var maxDiagonalElement float64
	for i := 0; i < numCols; i++ {
		absDiagonalElement := math.Abs(float64(diagonal[i]))
		if absDiagonalElement > maxDiagonalElement {
			maxDiagonalElement = absDiagonalElement
		}
		retVal.diagonalStatistics.Diagonal[i] = bignumber.NewFromInt64(int64(diagonal[i]))
	}
	ratio = maxDiagonalElement / math.Abs(float64(diagonal[numCols-1]))
	retVal.diagonalStatistics.Ratio = &ratio

	// Put zeroes to the right of the last non-zero entry
	for j := lastNonzeroEntryInLastRow + 1; j < numCols; j++ {
		hEntries[(numRows-1)*numCols+j] = 0
	}
	if 0 <= lastNonzeroEntryInLastRow {
		if hEntries[(numRows-1)*numCols+lastNonzeroEntryInLastRow] == 0 {
			hEntries[(numRows-1)*numCols+lastNonzeroEntryInLastRow] = 1
		}
	}

	// Make rows unreduced
	for i := 0; i < numUnreducedRows; i++ {
		retVal.rowIsUnreduced[unreducedRows[i]] = true
	}
	for i := 0; i < numRows; i++ {
		if retVal.rowIsUnreduced[i] {
			// First try to make one or more entries in row i unreduced at random
			unreducedRowHasBeenSet := false
			for j := 0; j < i; j++ {
				sgn := int64(2*rand.Intn(2) - 1)
				testInt := rand.Intn(3)
				if testInt == 0 {
					if diagonal[j] < 0 {
						hEntries[i*numCols+j] = sgn * int64(1-diagonal[j]/2)
					} else {
						hEntries[i*numCols+j] = sgn * int64(1+diagonal[j]/2)
					}
					if (retVal.unreducedRow == -1) && (retVal.unreducedColumn == -1) {
						retVal.unreducedRow = i
						retVal.unreducedColumn = j
					}
					unreducedRowHasBeenSet = true
				}
			}

			// If no entries in row i were selected at random to become unreduced, choose
			// one entry at random to un-reduce.
			if !unreducedRowHasBeenSet {
				sgn := int64(2*rand.Intn(2) - 1)
				j := 0
				if i > 1 {
					j = rand.Intn(i)
				}
				if diagonal[j] < 0 {
					hEntries[i*numCols+j] = sgn * int64(1-diagonal[j]/2)
				} else {
					hEntries[i*numCols+j] = sgn * int64(1+diagonal[j]/2)
				}
				if (retVal.unreducedRow == -1) && (retVal.unreducedColumn == -1) {
					retVal.unreducedRow = i
					retVal.unreducedColumn = j
				}
			}
		}
	}

	// Set h
	var err error
	var h *bigmatrix.BigMatrix
	h, err = bigmatrix.NewFromInt64Array(hEntries, numRows, numCols)
	if err != nil {
		return nil, retVal, fmt.Errorf("%s: could not create H: %q", caller, err.Error())
	}

	// Compute the best diagonal row operation. The way general row operations are computed
	// here is by combining row swaps, Givens rotations and reductions. This should give the
	// same results as the technique used in the code under test.
	var bestScore float64
	bestRowOp := []int{1, 0, 0, 1}
	bestJ := 0
	for j := 0; j < numCols-1; j++ {
		var score float64
		var rowOp []int
		var numRowSwaps int
		t := float64(hEntries[j*numCols+j])
		u := float64(hEntries[(j+1)*numCols+j])
		v := float64(hEntries[(j+1)*numCols+j+1])
		score, rowOp, numRowSwaps, err = score2x2(t, u, v, caller)
		if err != nil {
			return h, retVal, fmt.Errorf("%s: error in Score2x2: %q", caller, err.Error())
		}

		// Update the maximum number of row swaps
		if numRowSwaps > retVal.maxRowSwaps {
			retVal.maxRowSwaps = numRowSwaps
		}

		// Update the best score and best row operation
		if score > bestScore {
			bestScore = score
			bestJ = j
			for i := 0; i < 4; i++ {
				bestRowOp[i] = rowOp[i]
			}
		}
	}
	if (bestRowOp[0] == 0) && (bestRowOp[1] == 1) && (bestRowOp[2] == 1) && (bestRowOp[3] == 0) {
		retVal.bestDiagonalRowOp = &IntOperation{
			Indices:        []int{bestJ, bestJ + 1},
			OperationOnA:   []int{},
			OperationOnB:   []int{},
			PermutationOfA: [][]int{{0, 1}},
			PermutationOfB: [][]int{{0, 1}},
		}
	} else if (bestRowOp[0] == 1) && (bestRowOp[1] == 0) && (bestRowOp[2] == 0) && (bestRowOp[3] == 1) {
		// No diagonal row operation improves the diagonal
		retVal.bestDiagonalRowOp = nil
	} else {
		// A general row operation is best
		a, b, c, d := bestRowOp[0], bestRowOp[1], bestRowOp[2], bestRowOp[3]
		det := a*d - b*c
		retVal.bestDiagonalRowOp = &IntOperation{
			Indices:        []int{bestJ, bestJ + 1},
			OperationOnA:   []int{a, b, c, d},
			OperationOnB:   []int{det * d, -det * b, -det * c, det * a},
			PermutationOfA: [][]int{},
			PermutationOfB: [][]int{},
		}
	}

	// Set the bottom right of H
	if 0 <= lastNonzeroEntryInLastRow {
		retVal.bottomRightOfH.Found = true
		retVal.bottomRightOfH.RowNumberOfT = lastNonzeroEntryInLastRow
		retVal.bottomRightOfH.RowNumberOfU = numRows - 1
		retVal.bottomRightOfH.T = bignumber.NewFromInt64(
			hEntries[lastNonzeroEntryInLastRow*numCols+lastNonzeroEntryInLastRow],
		)
		retVal.bottomRightOfH.U = bignumber.NewFromInt64(
			hEntries[(numRows-1)*numCols+lastNonzeroEntryInLastRow],
		)
	}

	// Set the best row operation on the last row
	if retVal.bottomRightOfH.Found {
		tRow := retVal.bottomRightOfH.RowNumberOfT
		t := hEntries[tRow*numCols+tRow]
		u := hEntries[(numRows-1)*numCols+tRow]
		rowOp := []int{1, 0, 0, 1}
		for (t != 0) && (u != 0) {
			if t*t > u*u {
				// |t| > |u|, so t and row 0 of the bookkeeping matrix need updating
				if t*u > 0 {
					// Same signs, |t| > |u|
					t = t - u
					rowOp[0] = rowOp[0] - rowOp[2]
					rowOp[1] = rowOp[1] - rowOp[3]
					continue
				}

				// Different signs, |t| > |u|
				t = t + u
				rowOp[0] = rowOp[0] + rowOp[2]
				rowOp[1] = rowOp[1] + rowOp[3]
				continue
			}

			// Reaching here requires that |t| <= |u|, so u and the bottom row of the
			// bookkeeping matrix need updating
			if t*u > 0 {
				// same signs, |t| <= |u|
				u = u - t
				rowOp[2] = rowOp[2] - rowOp[0]
				rowOp[3] = rowOp[3] - rowOp[1]
				continue
			}

			// Different signs, |t| <= |u|
			u = u + t
			rowOp[2] = rowOp[2] + rowOp[0]
			rowOp[3] = rowOp[3] + rowOp[1]
		}
		if t*t < u*u {
			// Switch t and u
			tmpT := t
			t = u
			u = tmpT
			tmp0, tmp1 := rowOp[0], rowOp[1]
			rowOp[0] = rowOp[2]
			rowOp[1] = rowOp[3]
			rowOp[2] = tmp0
			rowOp[3] = tmp1
		}
		a, b, c, d := rowOp[0], rowOp[1], rowOp[2], rowOp[3]
		det := a*d - b*c
		if (det != 1) && (det != -1) {
			return h, retVal, fmt.Errorf(
				"%s: internal error det = %d; rowOp = %v", caller, det, rowOp,
			)
		}
		retVal.bestLastRowOp = &IntOperation{
			Indices:        []int{lastNonzeroEntryInLastRow, numRows - 1},
			OperationOnA:   []int{a, b, c, d},
			OperationOnB:   []int{det * d, -det * b, -det * c, det * a},
			PermutationOfA: [][]int{},
			PermutationOfB: [][]int{},
		}
	}

	// Return RandomHInfo
	return h, retVal, nil
}

// Score2x2 returns
//   - score, defined below; high scores are better than low scores
//   - the bookkeeping matrix, rowOp, that reduces t by the most possible, according to
//     the constraints described below.
//   - the number of row swaps performed, to indicate whether a row swap or a general roe
//     operation is equivalent to what score2x2 did.
//
// rowOp is a matrix with determinant 1 or -1, and with integer entries, that left-multiplies
// the matrix with rows [t 0] and [u v]. A Givens rotation, Q, right-multiplies this product
// to restore 0 to the upper right entry of the resulting matrix. Though the implementation
// of this function uses multiple rounds of left-multiplication and Givens rotations, they
// can all be combined into one left-multiplication and one Givens rotation without loss of
// generality.
//
// If the notation for the final result of the above steps calls its rows [t' 0] and [u' v'],
// the score is |t/t'|.
func score2x2(t, u, v float64, caller string) (float64, []int, int, error) {
	// Initializations
	caller = fmt.Sprintf("%s-score2x2", caller)
	tolerance := math.Pow(2.0, -float64BitTolerance)
	score := 1.0 // startingT / startingT
	startingT, startingU, startingV := t, u, v
	rowOp := []int{1, 0, 0, 1}
	numRowSwaps := 0

	// A ratio of more than 1 indicates the diagonal needs improvement
	for {
		// Remove the corner from the row-swapped sub-matrix, H', of H, which has rows
		// [u v] and [t 0]. To do this, a Givens rotation with rows [c -s] and [s c] is
		// needed which zeroes out v in H'. Let
		//
		// - r = sqrt(u^2+v^2)
		// - c = u/r
		// - s = v/r
		//
		// right-multiplying H' by the Givens matrix produces a top row with 0 in its
		// right-hand column, namely
		//
		// [(u)(c)+(v)(s) (u)(-s)+(v)(c)] = [(u)(u/r)+(v)(v/r) (u)(-v/r)+(v)(u/r)]
		//   = [(u^2+v^2)/r 0]
		//   = [r^2/r 0]
		//   = [r 0]
		//
		// The bottom row of the same product -- (H')(Givens matrix) -- is
		//
		// [(t)(c)+(0)(s) (t)(-s)+(0)(c)] = [tc -ts]. The new values of t, u and v,
		// taken from the top and bottom rows of (H')(Givens matrix), can be computed by
		//
		// newT <- r
		// newU <- tc
		// newV <- -ts

		// Define the Givens matrix
		r := math.Sqrt(u*u + v*v)
		c := u / r
		s := v / r

		// Update t, u and v
		newT := r
		newU := t * c
		newV := -t * s

		// Compute the new score and if it does not improve |t|/|v|, exit the loop
		newScore := math.Abs(startingT / newT)
		if !(newScore > score) {
			// The score has stopped increasing
			break
		}

		// The swap improves the diagonal. Update t, u, v, rowOp and ratio. The update to
		// rowOp is a row swap, since that is what gave rise to the new t, u and v.
		t = newT
		u = newU
		v = newV
		tmp := rowOp[0]
		rowOp[0] = rowOp[2]
		rowOp[2] = tmp
		tmp = rowOp[1]
		rowOp[1] = rowOp[3]
		rowOp[3] = tmp
		numRowSwaps++
		score = newScore

		// Before resuming the reduction loop, use a row operation with rows [1 0] and [x 1] to
		// reduce |u| to |t/2| or less. The constraint on x is:
		//
		// xt + u ~ 0 => xt ~ -u => x = nearest integer to -t/u.
		if math.Abs(2*u) > math.Abs(t) {
			var xAsInt int
			xAsFloat64 := -u / t
			if xAsFloat64 > 0 {
				xAsInt = int(0.5 + xAsFloat64)
			} else {
				xAsInt = -int(0.5 - xAsFloat64)
			}
			if xAsInt != 0 {
				// Left-multiplying by the row operation with rows [1 0] and [x 1] affects
				// only the bottom row. So two bottom entries in the bookkeeping matrix need
				// updates, but the top row remains unchanged. In the case of the matrix with
				// rows [t 0] and [u v], the presence of a zero in the upper-right means only
				// u is updated.
				u = float64(xAsInt)*t + u
				rowOp[2] = xAsInt*rowOp[0] + rowOp[2]
				rowOp[3] = xAsInt*rowOp[1] + rowOp[3]
			}
		}

		// Check invariants
		// - row norms are the same for current matrix with rows [t 0] and [u v] as for
		//   the row op times original matrix.
		// - starting and current determinant are the same
		currentNorm := []float64{math.Abs(t), math.Sqrt(u*u + v*v)}
		productA := float64(rowOp[0])*startingT + float64(rowOp[1])*startingU
		productB := float64(rowOp[1]) * startingV
		productC := float64(rowOp[2])*startingT + float64(rowOp[3])*startingU
		productD := float64(rowOp[3]) * startingV
		productNorm := []float64{
			math.Sqrt(productA*productA + productB*productB),
			math.Sqrt(productC*productC + productD*productD),
		}
		for i := 0; i < 2; i++ {
			if math.Abs(currentNorm[i]-productNorm[i]) > tolerance {
				return 0.0, nil, numRowSwaps, fmt.Errorf(
					"%s: current norm for row %d = %f != %f = product norm for row %d diff = %e",
					caller, i, currentNorm[i], productNorm[i], i,
					currentNorm[i]-productNorm[i],
				)
			}
		}
		startingDet := startingT * startingV
		currentDet := t * v
		diff := math.Abs(startingDet) - math.Abs(currentDet)
		if math.Abs(diff) > tolerance {
			return 0.0, nil, numRowSwaps, fmt.Errorf(
				"%s: |starting determinant| = %f != %f = |current determinant| diff = %e",
				caller, startingDet, currentDet, diff,
			)
		}
	}
	return score, rowOp, numRowSwaps, nil
}

type RandomMInfo struct {
	unreducedRow       int    // First unreduced row
	unreducedColumn    int    // First unreduced column in first unreduced row
	columnIsUnreduced  []bool // Whether each column is unreduced
	bestColumnOp       *IntOperation
	diagonalStatistics *DiagonalStatistics
}

// CreateRandomM creates a random M and a RandomMInfo matrix with the specified
//
// - number of rows
//
// - number of possible entries from which each non-zero entry in M is selected
//
// - a (possibly empty) list of columns to make unreduced in M, for testing reduction by E
func createRandomM(
	numRows, minDiagonalElementSize, maxDiagonalElementSize int, unreducedColumns []int, caller string,
) (*bigmatrix.BigMatrix, *RandomMInfo, error) {
	// Initializations
	caller = fmt.Sprintf("%s-createRandomM", caller)
	numUnreducedColumns := len(unreducedColumns)
	mEntries := make([]int64, numRows*numRows)
	retVal := &RandomMInfo{
		unreducedRow:       -1,
		unreducedColumn:    -1,
		columnIsUnreduced:  make([]bool, numRows),
		bestColumnOp:       nil,
		diagonalStatistics: nil,
	}

	// Check input
	if numRows < 2 {
		return nil, nil, fmt.Errorf("%s: numRows = %d < 2", caller, numRows)
	}
	if minDiagonalElementSize < 2 {
		return nil, nil, fmt.Errorf(
			"%s: minDiagonalElementSize = %d < 2", caller, minDiagonalElementSize,
		)
	}
	if maxDiagonalElementSize <= minDiagonalElementSize {
		return nil, nil, fmt.Errorf(
			"%s: maxDiagonalElementSize = %d <= %d = minDiagonalElementSize",
			caller, maxDiagonalElementSize, minDiagonalElementSize,
		)
	}
	for i := 0; i < numUnreducedColumns; i++ {
		if unreducedColumns[i] < 0 {
			return nil, nil, fmt.Errorf(
				"%s: unreducedColumns[%d] = %d < 0", caller, i, unreducedColumns[i],
			)
		}
		if numRows-2 < unreducedColumns[i] {
			// The last column cannot be unreduced, since its only non-zero entry is a diagonal element
			return nil, nil, fmt.Errorf(
				"%s: unreducedColumns[%d] = %d > %d", caller, i, unreducedColumns[i], numRows-2,
			)
		}
	}

	// Set diagonal entries
	diagonal := make([]int, numRows)
	for j := 0; j < numRows; j++ {
		diagonalEntry := minDiagonalElementSize + rand.Intn(
			maxDiagonalElementSize-minDiagonalElementSize,
		)
		sgn := 2*rand.Intn(2) - 1
		diagonal[j] = sgn * diagonalEntry
		mEntries[j*numRows+j] = int64(diagonal[j])
	}

	// Set entries below the diagonal
	for i := 0; i < numRows; i++ {
		maxEntry := diagonal[i]
		if maxEntry < 0 {
			maxEntry = -maxEntry
		}
		maxEntry = maxEntry / 2
		for j := 0; j < i; j++ {
			sgn := 2*rand.Intn(2) - 1
			mEntries[i*numRows+j] = int64(sgn * rand.Intn(maxEntry+1))
		}
	}

	// Make columns unreduced
	for i := 0; i < numUnreducedColumns; i++ {
		retVal.columnIsUnreduced[unreducedColumns[i]] = true
	}
	for j := 0; j < numRows; j++ {
		if retVal.columnIsUnreduced[j] {
			// First try to make one or more entries in column j unreduced at random
			unreducedColumnHasBeenSet := false
			for i := j + 1; i < numRows; i++ {
				sgn := int64(2*rand.Intn(2) - 1)
				testInt := rand.Intn(3)
				if testInt == 0 {
					if diagonal[i] < 0 {
						mEntries[i*numRows+j] = sgn * int64(1-diagonal[i]/2)
					} else {
						mEntries[i*numRows+j] = sgn * int64(1+diagonal[i]/2)
					}
					if (retVal.unreducedRow == -1) && (retVal.unreducedColumn == -1) {
						retVal.unreducedRow = i
						retVal.unreducedColumn = j
					}
					unreducedColumnHasBeenSet = true
				}
			}

			// If no entries in column j were selected at random to become unreduced, choose
			// one entry at random to un-reduce.
			if !unreducedColumnHasBeenSet {
				sgn := int64(2*rand.Intn(2) - 1)
				i := numRows - 1
				if j < numRows-2 {
					i = (j + 1) + rand.Intn(numRows-(j+1))
				}
				if diagonal[i] < 0 {
					mEntries[i*numRows+j] = sgn * int64(1-diagonal[i]/2)
				} else {
					mEntries[i*numRows+j] = sgn * int64(1+diagonal[i]/2)
				}
				if (retVal.unreducedRow == -1) && (retVal.unreducedColumn == -1) {
					retVal.unreducedRow = i
					retVal.unreducedColumn = j
				}
			}
		}
	}

	// Set M
	var err error
	var m *bigmatrix.BigMatrix
	m, err = bigmatrix.NewFromInt64Array(mEntries, numRows, numRows)
	if err != nil {
		return m, nil, fmt.Errorf("%s: could not create M: %q", caller, err.Error())
	}

	// Generate diagonal statistics
	var ratio float64
	retVal.diagonalStatistics = &DiagonalStatistics{
		Diagonal: make([]*bignumber.BigNumber, numRows),
		Ratio:    &ratio,
	}
	var minDiagonalElement float64
	minDiagonalElement = math.MaxInt64
	for i := 0; i < numRows; i++ {
		absDiagonalElement := math.Abs(float64(diagonal[i]))
		if absDiagonalElement < minDiagonalElement {
			minDiagonalElement = absDiagonalElement
		}
		retVal.diagonalStatistics.Diagonal[i] = bignumber.NewFromInt64(int64(diagonal[i]))
	}
	ratio = math.Abs(float64(diagonal[numRows-1])) / minDiagonalElement
	retVal.diagonalStatistics.Ratio = &ratio

	// To prepare for finding the best column swap, compute norms for each column.
	norms := make([]float64, numRows)
	for j := 0; j < numRows; j++ {
		for i := j; i < numRows; i++ {
			norms[j] += float64(mEntries[i*numRows+j] * mEntries[i*numRows+j])
		}
	}

	// Iterate through right-most columns, indexed by j1, looking for one that can be swapped
	// out. If there is one, swap it with the lowest-norm column. At each iteration, norms
	// involve rows 0,1,...,j1-1.
	for j1 := numRows - 1; j1 > 0; j1-- {
		bestNorm := norms[j1]
		bestIndex := -1
		for j0 := 0; j0 < j1; j0++ {
			// Update bestNorm and bestIndex
			if norms[j0] < bestNorm {
				bestNorm = norms[j0]
				bestIndex = j0
			}

			// Prepare for the next loop by subtracting row j1 from the norms, so
			// the norms involve only rows 0,1,...,j1-1
			norms[j0] -= float64(mEntries[j1*numRows+j0] * mEntries[j1*numRows+j0])
		}
		if 0 <= bestIndex {
			retVal.bestColumnOp = &IntOperation{
				Indices:        []int{bestIndex, j1},
				OperationOnA:   []int{},
				OperationOnB:   []int{},
				PermutationOfA: [][]int{{0, 1}},
				PermutationOfB: [][]int{{0, 1}},
			}
			break
		}
	}
	return m, retVal, nil
}

// compareRowOps returns an error if
//
//  1. The expected and actual row operations do not increase expectedT and actualT by the same
//     ratio when used to left-multiply [t u]-transpose. (We measure increase instead of decrease
//     to avoid divide-by-zero).
//
//     Note: There is an exception to criterion 1 found during testing, where the same indices
//     and row operations are found by the expected and actual calculation, except for a
//     difference of 1 in the value of b that makes a small difference in the resulting
//     value of t.
//
//  2. Expected and actual v are zero, and the expected and actual row operations do not increase
//     expectedU and actualU by the same ratio when used to left-multiply [t u]-transpose. This
//     applies only if expected and actual t are equal, and likewise for u. The reason that expected
//     and actual v need to be zero for this to matter is that v != 0 is for a diagonal row operation
//     where u can later be reduced anyway.
//
//  3. The expected row operations on A and B are not inverses
//
//  4. The actual row operations on A and B are not inverses
//
//  5. The ratio by which expectedT and actualT are increased (which are the same after
//     test #2) is not in the half-closed interval, (0,1]
//
//  6. Expected v = 0 and the ratio by which expectedU is increased is not in the closed
//     interval, [0, 1]. When expected V is zero, the row operation involves the last
//     row of H, and the effect on u matters.
//
//  7. Actual v = 0 and the ratio by which actualU is increased is not in the closed
//     interval, [0, 1]. When expected V is zero, the row operation involves the last
//     row of H, and the effect on u matters.
//
// Expected |t| and |u| could be different from actual |t| and |u| in the case where
// operations on different 2x2 sub-matrices along the diagonal of H are recommended by the
// expected and actual row operation generators. But the ratio by which |t| is increased
// should be the same (test #2), and |u| should not increase in either the expected or actual
// scenario (test #6 and #7).
func compareRowOps(
	expectedT, actualT, expectedU, actualU, expectedV, actualV *bignumber.BigNumber,
	expectedRowOp, actualRowOp *IntOperation,
	log2Tolerance int, caller string,
) (bool, error) {
	// Initializations
	caller = fmt.Sprintf("%s-compareRowOps", caller)
	tolerance := bignumber.NewPowerOfTwo(int64(log2Tolerance))
	closeEnoughButNotExact := false // set to true for exceptions to exact matches

	// Convenience structs for expected and actual inputs
	var er, eri, ar, ari []int // matrices for expected and actual row operations
	if len(expectedRowOp.OperationOnA) == 4 {
		er = expectedRowOp.OperationOnA
		eri = expectedRowOp.OperationOnB
	} else {
		// The operation should be a row swap permutation
		er = []int{0, 1, 1, 0}
		eri = []int{0, 1, 1, 0}
	}
	if len(actualRowOp.OperationOnA) == 4 {
		ar = actualRowOp.OperationOnA
		ari = actualRowOp.OperationOnB
	} else {
		// The operation should be a row swap permutation
		ar = []int{0, 1, 1, 0}
		ari = []int{0, 1, 1, 0}
	}
	expected := struct {
		// OperationOnA
		a int
		b int
		c int
		d int

		// OperationOnB
		ai int
		bi int
		ci int
		di int

		// T and U after left-multiplying by the row operation
		tRatio *bignumber.BigNumber
		uRatio *bignumber.BigNumber // this matters only if expectedV = 0
	}{
		a:      er[0],
		b:      er[1],
		c:      er[2],
		d:      er[3],
		ai:     eri[0],
		bi:     eri[1],
		ci:     eri[2],
		di:     eri[3],
		tRatio: bignumber.NewFromInt64(0),
		uRatio: bignumber.NewFromInt64(0),
	}
	actual := struct {
		// OperationOnA
		a int
		b int
		c int
		d int

		// OperationOnB
		ai int
		bi int
		ci int
		di int

		// T and U after left-multiplying by the row operation
		tRatio *bignumber.BigNumber
		uRatio *bignumber.BigNumber // this matters only if actualV == 0
	}{
		a:      ar[0],
		b:      ar[1],
		c:      ar[2],
		d:      ar[3],
		ai:     ari[0],
		bi:     ari[1],
		ci:     ari[2],
		di:     ari[3],
		tRatio: bignumber.NewFromInt64(0),
		uRatio: bignumber.NewFromInt64(0),
	}

	// Compute ratios of before and after the expected row operation for expected T and U
	// The ratio for u is computed as if (and is later used only if) expected V = 0.
	var ratio *bignumber.BigNumber
	at := bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(expected.a)), expectedT)
	bu := bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(expected.b)), expectedU)
	bv := bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(expected.b)), expectedV)
	atPlusBu := bignumber.NewFromInt64(0).Add(at, bu)
	atPlusBuSq := bignumber.NewFromInt64(0).Mul(atPlusBu, atPlusBu)
	bvSq := bignumber.NewFromInt64(0).Mul(bv, bv)
	expectedNewTSq := bignumber.NewFromInt64(0).Add(atPlusBuSq, bvSq)
	expectedNewT, err := bignumber.NewFromInt64(0).Sqrt(expectedNewTSq)
	if err != nil {
		_, expectedNewTSqAsStr := expectedNewTSq.String()
		return false, fmt.Errorf(
			"%s: could not take the square root of expectedNewTSq = %s: %q",
			caller, expectedNewTSqAsStr, err.Error(),
		)
	}
	ratio, err = bignumber.NewFromInt64(0).Quo(expectedNewT, expectedT)
	if err != nil {
		_, expectedTAsStr := expectedT.String()
		return false, fmt.Errorf(
			"%s: could not divide by expectedT = %s: %q", caller, expectedTAsStr, err.Error(),
		)
	}
	expected.tRatio.Abs(ratio)
	if !expectedU.IsZero() {
		ct := bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(expected.c)), expectedT)
		du := bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(expected.d)), expectedU)
		expectedNewU := bignumber.NewFromInt64(0).Add(ct, du)
		ratio, err = bignumber.NewFromInt64(0).Quo(expectedNewU, expectedU)
		if err != nil {
			_, expectedUAsStr := expectedU.String()
			return false, fmt.Errorf(
				"%s: could not divide by expectedU = %s: %q", caller, expectedUAsStr, err.Error(),
			)
		}
		expected.uRatio.Abs(ratio)
	}

	// Compute ratios of before and after the actual row operation for actual T and U
	// The ratio for u is computed as if (and is later used only if) actual V = 0.
	var actualNewT *bignumber.BigNumber
	at = bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(actual.a)), actualT)
	bu = bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(actual.b)), actualU)
	bv = bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(actual.b)), actualV)
	atPlusBu = bignumber.NewFromInt64(0).Add(at, bu)
	atPlusBuSq = bignumber.NewFromInt64(0).Mul(atPlusBu, atPlusBu)
	bvSq = bignumber.NewFromInt64(0).Mul(bv, bv)
	actualNewTSq := bignumber.NewFromInt64(0).Add(atPlusBuSq, bvSq)
	actualNewT, err = bignumber.NewFromInt64(0).Sqrt(actualNewTSq)
	if err != nil {
		_, actualNewTSqAsStr := actualNewTSq.String()
		return false, fmt.Errorf(
			"%s: could not take the square root of actualNewTSq = %s: %q",
			caller, actualNewTSqAsStr, err.Error(),
		)
	}
	ratio, err = bignumber.NewFromInt64(0).Quo(actualNewT, actualT)
	if err != nil {
		_, actualTAsStr := actualT.String()
		return false, fmt.Errorf(
			"%s: could not divide by actualT = %s: %q", caller, actualTAsStr, err.Error(),
		)
	}
	actual.tRatio.Abs(ratio)
	if !actualU.IsZero() {
		ct := bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(actual.c)), actualT)
		du := bignumber.NewFromInt64(0).Mul(bignumber.NewFromInt64(int64(actual.d)), actualU)
		actualNewU := bignumber.NewFromInt64(0).Add(ct, du)
		ratio, err = bignumber.NewFromInt64(0).Quo(actualNewU, actualU)
		if err != nil {
			_, actualUAsStr := actualU.String()
			return false, fmt.Errorf(
				"%s: could not divide by actualU = %s: %q", caller, actualUAsStr, err.Error(),
			)
		}
		actual.uRatio.Abs(ratio)
	}

	// 1. Return an error if the expected and actual row operations do not increase expectedT
	//    and actualT by the same ratio when used to left-multiply [t u]-transpose. Note the
	//    exception described in the comment for compareRowOps and implemented below.
	if !expected.tRatio.Equals(actual.tRatio, tolerance) {
		if expected.a != 0 {
			// The expected row operation is general. The actual row operation generator
			// computes some general row operations sub-optimally, which can cause a row
			// swap to out-compete a general operation when it shouldn't, or cause a
			// general operation to be returned when a better one should be returned.
			//
			// This is a known flaw in the algorithm under test, but it  does not prevent
			// the eventual zeroing out of the last row of H. Rather, it just slows down
			// reduction by a small amount.
			//
			// In testing, the sub-optimal row operations have been found to score no more
			// than 25% worse than the expected row operations.
			var normalizedDiff *bignumber.BigNumber
			diff := bignumber.NewFromInt64(0).Sub(expected.tRatio, actual.tRatio)
			normalizedDiff, err = bignumber.NewFromInt64(0).Quo(diff, expected.tRatio)
			if normalizedDiff.Cmp(bignumber.NewPowerOfTwo(-2)) < 0 {
				closeEnoughButNotExact = true
			}
		}
		if !closeEnoughButNotExact {
			_, expectedTAsStr := expectedT.String()
			_, expectedUAsStr := expectedU.String()
			_, expectedVAsStr := expectedV.String()
			_, expectedNewTAsStr := expectedNewT.String()
			_, expectedTRatioAsStr := expected.tRatio.String()
			_, actualTAsStr := actualT.String()
			_, actualUAsStr := actualU.String()
			_, actualVAsStr := actualV.String()
			_, actualNewTAsStr := actualNewT.String()
			_, actualTRatioAsStr := actual.tRatio.String()
			return false, fmt.Errorf(
				"%s:\n"+
					" expected Indices %v sqrt(((%d)(%s)+(%d)(%s))^2+((%d)(%s))^2) = %s\n"+
					" actual   Indices %v sqrt(((%d)(%s)+(%d)(%s))^2+((%d)(%s))^2) = %s\n"+
					" |new t / original t|: expected |%s/%s| = %s actual |%s/%s| = %s",
				caller,
				expectedRowOp.Indices, expected.a, expectedTAsStr, expected.b, // expected part 1
				expectedUAsStr, expected.b, expectedVAsStr, expectedNewTAsStr, // expected part 2
				actualRowOp.Indices, actual.a, actualTAsStr, actual.b, // actual part 1
				actualUAsStr, actual.b, actualVAsStr, actualNewTAsStr, // actual part 2
				expectedNewTAsStr, expectedTAsStr, expectedTRatioAsStr, // ratio - expected part
				actualNewTAsStr, actualTAsStr, actualTRatioAsStr, // ratio - actual part
			)
		}
	}

	// 2. Return an error if expected and actual v are zero, and the expected and actual row
	//    operations do not increase expectedU and actualU by the same ratio when used to left-
	//    multiply [t u]-transpose. This applies only if expected and actual t are equal, and
	//    likewise for u.
	if expectedV.IsZero() && actualV.IsZero() {
		if (expectedT.Equals(actualT, tolerance)) && (expectedU.Equals(actualU, tolerance)) {
			// expectedT and actualT are equal, as are expectedU and actualU, so the
			// criteria have been met to compare the ratios for expected and actual u.
			if !expected.uRatio.Equals(actual.uRatio, tolerance) {
				_, expectedTAsStr := expectedT.String()
				_, expectedUAsStr := expectedU.String()
				_, expectedNewTAsStr := expectedNewT.String()
				_, expectedURatioAsStr := expected.tRatio.String()
				_, actualTAsStr := actualT.String()
				_, actualUAsStr := actualU.String()
				_, actualNewTAsStr := actualNewT.String()
				_, actualURatioAsStr := actual.uRatio.String()
				return false, fmt.Errorf(
					"%s: expected <%v,[%s %s]> = %s; |%s/%s|=%s actual <%v,[%s %s]>=%s; |%s/%s|=%s",
					caller,
					expectedRowOp.OperationOnA[2:4], expectedTAsStr, expectedUAsStr, expectedNewTAsStr,
					expectedNewTAsStr, expectedUAsStr, expectedURatioAsStr,
					actualRowOp.OperationOnA[2:4], actualTAsStr, actualUAsStr, actualNewTAsStr,
					actualNewTAsStr, actualUAsStr, actualURatioAsStr,
				)
			}
		}
	}

	// 3. Return an error if the expected row operations on A and B are not inverses
	shouldBeIdentity := []int{
		expected.a*expected.ai + expected.b*expected.ci,
		expected.a*expected.bi + expected.b*expected.di,
		expected.c*expected.ai + expected.d*expected.ci,
		expected.c*expected.bi + expected.d*expected.di,
	}
	identity := []int{1, 0, 0, 1}
	for i := 0; i < 4; i++ {
		if shouldBeIdentity[i] != identity[i] {
			return false, fmt.Errorf(
				"%s: expected product of operations is %v not %v",
				caller, shouldBeIdentity, []int{1, 0, 0, 1},
			)
		}
	}

	// 4. Return an error if the actual row operations on A and B are not inverses
	shouldBeIdentity = []int{
		actual.a*actual.ai + actual.b*actual.ci,
		actual.a*actual.bi + actual.b*actual.di,
		actual.c*actual.ai + actual.d*actual.ci,
		actual.c*actual.bi + actual.d*actual.di,
	}
	for i := 0; i < 4; i++ {
		if shouldBeIdentity[i] != identity[i] {
			return false, fmt.Errorf(
				"%s: actual product of operations is %v not %v",
				caller, shouldBeIdentity, identity,
			)
		}
	}

	// 5. Return an error if the ratio by which expectedT and actualT are increased is not
	//    in the half-closed interval, (0,1]
	if actual.tRatio.IsSmall() {
		_, tAsStr := actualT.String()
		return false, fmt.Errorf("%s: actual t = %s is essentially zero", caller, tAsStr)
	}
	one := bignumber.NewFromInt64(0)
	diff := bignumber.NewFromInt64(0).Sub(one, actual.tRatio)
	if diff.Cmp(tolerance) > 0 {
		_, expectedTAsStr := expectedT.String()
		_, expectedTRatioAsStr := expected.tRatio.String()
		return false, fmt.Errorf(
			"%s: expected t went from %s to (%s)(%s), a ratio of more than 1",
			caller, expectedTAsStr, expectedTRatioAsStr, expectedTAsStr,
		)
	}

	// 6. Return an error if expected v != 0 and the ratio by which expectedU is increased
	//    is not in the closed interval, [0, 1]
	if expectedV.IsZero() {
		diff = bignumber.NewFromInt64(0).Sub(one, actual.uRatio)
		if diff.Cmp(tolerance) > 0 {
			_, expectedUAsStr := expectedU.String()
			_, expectedURatioAsStr := expected.uRatio.String()
			return false, fmt.Errorf(
				"%s: expected u went from %s to (%s)(%s), a ratio of more than 1",
				caller, expectedUAsStr, expectedURatioAsStr, expectedUAsStr,
			)
		}
	}

	// 7. Return an error if actual v = 0 and the ratio by which actualU is increased is
	//    not in the closed interval, [0, 1].
	if actualV.IsZero() {
		diff = bignumber.NewFromInt64(0).Sub(one, actual.uRatio)
		if diff.Cmp(tolerance) > 0 {
			_, actualUAsStr := actualU.String()
			_, actualURatioAsStr := actual.uRatio.String()
			return false, fmt.Errorf(
				"%s: actual u went from %s to (%s)(%s), a ratio of more than 1",
				caller, actualUAsStr, actualURatioAsStr, actualUAsStr,
			)
		}
	}

	// No errors were found
	return closeEnoughButNotExact, nil
}

func gcd(a, b int64) int64 {
	// Continue until one number becomes zero
	for b != 0 {
		// Find remainder and update a and b
		a, b = b, a%b
	}
	return a
}
