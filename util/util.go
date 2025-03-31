package util

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"math"
)

// CopyInt64ToInt converts an int64 matrix to an int matrix
func CopyInt64ToInt(input []int64) []int {
	retVal := make([]int, len(input))
	for i := 0; i < len(input); i++ {
		retVal[i] = int(input[i])
	}
	return retVal
}

// CopyIntToInt64 converts an int matrix to an int64 matrix
func CopyIntToInt64(input []int) []int64 {
	retVal := make([]int64, len(input))
	for i := 0; i < len(input); i++ {
		retVal[i] = int64(input[i])
	}
	return retVal
}

// MultiplyFloatInt returns the matrix product, x * y, for []float64
// x and []int64 y
func MultiplyFloatInt(x []float64, y []int, n int) ([]float64, error) {
	// x is mxn, y is nxp and xy is mxp.
	m, p, err := getDimensions(len(x), len(y), n, "MultiplyFloatInt")
	if err != nil {
		return []float64{}, err
	}
	xy := make([]float64, m*p)
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			xy[i*p+j] = x[i*n] * float64(y[j]) // x[i][0] * y[0][j]
			for k := 1; k < n; k++ {
				xy[i*p+j] += x[i*n+k] * float64(y[k*p+j]) // x[i][k] * y[k][j]
			}
		}
	}
	return xy, nil
}

// MultiplyIntInt returns the matrix product, x * y, for []int64
// x and []int64 y. n must equal the number of columns in x and
// the number of rows in y.
func MultiplyIntInt(x []int64, y []int64, n int) ([]int64, error) {
	// x is mxn, y is nxp and xy is mxp.
	m, p, err := getDimensions(len(x), len(y), n, "MultiplyIntInt")
	if err != nil {
		return nil, err
	}
	largeEntryThresh := int64(math.MaxInt32 / m)
	if err != nil {
		return []int64{}, err
	}
	xy := make([]int64, m*p)
	for i := 0; i < m; i++ {
		for j := 0; j < p; j++ {
			xyEntry := x[i*n] * y[j] // x[i][0] * y[0][j]
			for k := 1; k < n; k++ {
				xyEntry += x[i*n+k] * y[k*p+j] // x[i][k] * y[k][j]
			}
			if (xyEntry > largeEntryThresh) || (xyEntry < -largeEntryThresh) {
				return []int64{}, fmt.Errorf(
					"in a matrix multiply, entry (%d,%d) = %d is large enough to risk future overflow",
					i, j, xyEntry,
				)
			}
			xy[i*p+j] = xyEntry
		}
	}
	return xy, nil
}

// IsInversePair returns whether x and y are inverses of each other
func IsInversePair(x, y []int64, dim int) (bool, error) {
	shouldBeInverse, err := MultiplyIntInt(x, y, dim)
	if err != nil {
		return false, fmt.Errorf(
			"IsInversePair: could not multiply x (%d-long) by y (%d-long): %q", len(x), len(y), err.Error(),
		)
	}
	for i := 0; i < dim; i++ {
		for j := 0; j < dim; j++ {
			if (i == j) && (shouldBeInverse[i*dim+j] != 1) {
				return false, nil
			} else if (i != j) && (shouldBeInverse[i*dim+j] != 0) {
				return false, nil
			}
		}
	}
	return true, nil
}

func GetPermutationMatrices(indices, perm []int, numRows int) ([]int64, []int64, error) {
	// Need permutation matrices to calculate expected values the slow-but-sure way,
	// by multiplying an input matrix by a permutation matrix.
	numIndices := len(indices)
	rowPermutationMatrix := make([]int64, numRows*numRows)
	colPermutationMatrix := make([]int64, numRows*numRows)
	for i := 0; i < numIndices; i++ {
		for j := 0; j < numIndices; j++ {
			if perm[j] == i {
				rowPermutationMatrix[indices[i]*numRows+indices[j]] = 1
			}
			if perm[i] == j {
				colPermutationMatrix[indices[i]*numRows+indices[j]] = 1
			}
		}
	}

	// The permutation matrices have 1s in the sub-matrices with coordinates from
	// indices. In rows that are still all-zero, the permutation matrices need ones
	// on the diagonal.
	for i := 0; i < numRows; i++ {
		needDiagonalEntry := true
		for j := 0; j < numIndices; j++ {
			if i == indices[j] {
				needDiagonalEntry = false
				break
			}
		}
		if needDiagonalEntry {
			rowPermutationMatrix[i*numRows+i] = 1
			colPermutationMatrix[i*numRows+i] = 1
		}
	}

	// The row and column permutation matrices should be inverses of each other
	areInverses, err := IsInversePair(rowPermutationMatrix, colPermutationMatrix, numRows)
	if err != nil {
		return []int64{}, []int64{},
			fmt.Errorf("getPermutationMatrices: isInversePair returned an error: %q", err.Error())
	}
	if !areInverses {
		return []int64{}, []int64{},
			fmt.Errorf("getPermutationMatrices: permutation matrices are not inverses: %q", err.Error())
	}
	return rowPermutationMatrix, colPermutationMatrix, nil
}

// getDimensions returns the dimensions m and p for a matrix multiply
// xy where x has mn entries, y has np entries, and the number of columns
// in x (= the number of rows in y) is n.
func getDimensions(mn, np, n int, caller string) (int, int, error) {
	caller = fmt.Sprintf("%s-getDimensions", caller)
	if mn%n != 0 {
		return 0, 0, fmt.Errorf(
			"%s: multiplyIntFloat: non-integer number of rows %d / %d in x", caller, mn, n,
		)
	}
	if np%n != 0 {
		return 0, 0, fmt.Errorf(
			"%s: multiplyIntFloat: non-integer number of columns  %d / %d in y", caller, np, n,
		)
	}
	return mn / n, np / n, nil
}
