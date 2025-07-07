package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/util"
	"math"
	"strings"
	"testing"

	"github.com/predrag3141/IPSLQ/bignumber"
	"github.com/stretchr/testify/require"
)

func TestBestDiagonalRowOp(t *testing.T) {
	const (
		numPossibleEntries = 1000
		numRows            = 17
		numCols            = 16
		log2tolerance      = -50
		numTests           = 100
	)

	// tolerance := bignumber.NewPowerOfTwo(log2tolerance)
	maxRowSwapCount := make(map[int]int)
	closeEnoughButNotExactCount := 0
	for testNbr := 0; testNbr < numTests; testNbr++ {
		var actualRowOp *IntOperation
		h, expected, err := createRandomH(numRows, numPossibleEntries, numCols-1, []int{}, "TestGetBestDiagonalRowOp")
		require.NoError(t, err)
		actualRowOp, err = bestDiagonalRowOpInH(h, "GetBestDiagonalRowOp")
		require.NoError(t, err)
		maxRowSwapCount[expected.maxRowSwaps]++

		// If expected.bestDiagonalRowOp == nil, actualRowOp should be nil, and vice versa.
		if (expected.bestDiagonalRowOp == nil) || (actualRowOp == nil) {
			require.Nil(t, expected.bestDiagonalRowOp)
			require.Nil(t, actualRowOp)
		} else {
			// Get expectedT, expectedU and expectedV
			var expectedT, expectedU, expectedV *bignumber.BigNumber
			require.Equal(t, 2, len(expected.bestDiagonalRowOp.Indices))
			require.Equal(
				t, 1, expected.bestDiagonalRowOp.Indices[1]-expected.bestDiagonalRowOp.Indices[0],
			)
			expectedT, err = h.Get(
				expected.bestDiagonalRowOp.Indices[0], expected.bestDiagonalRowOp.Indices[0],
			)
			require.NoError(t, err)
			expectedU, err = h.Get(
				expected.bestDiagonalRowOp.Indices[1], expected.bestDiagonalRowOp.Indices[0],
			)
			require.NoError(t, err)
			expectedV, err = h.Get(
				expected.bestDiagonalRowOp.Indices[1], expected.bestDiagonalRowOp.Indices[1],
			)

			// Get actualT, actualU and actualV
			var actualT, actualU, actualV *bignumber.BigNumber
			require.Equal(t, 2, len(actualRowOp.Indices))
			require.Equal(t, 1, actualRowOp.Indices[1]-actualRowOp.Indices[0])
			actualT, err = h.Get(actualRowOp.Indices[0], actualRowOp.Indices[0])
			require.NoError(t, err)
			actualU, err = h.Get(actualRowOp.Indices[1], actualRowOp.Indices[0])
			require.NoError(t, err)
			actualV, err = h.Get(actualRowOp.Indices[1], actualRowOp.Indices[1])
			require.NoError(t, err)

			// Compare the results for expected and actual
			closeEnoughButNotExact := false
			closeEnoughButNotExact, err = compareRowOps(
				expectedT, actualT, expectedU, actualU, expectedV, actualV,
				expected.bestDiagonalRowOp, actualRowOp,
				log2tolerance, "GetBestDiagonalRowOp",
			)
			require.NoError(t, err)
			if closeEnoughButNotExact {
				closeEnoughButNotExactCount++
			}
		}

	}
	t.Logf(
		"Counts of the maximum number of row swaps per test: %s\n",
		strings.Replace(fmt.Sprintf("%v", maxRowSwapCount), `map`, "", 1),
	)
	t.Logf(
		"Count of columnsTested where results were close enough but not exact: %d\n",
		closeEnoughButNotExactCount,
	)
}

func TestColumnSortingMethods(t *testing.T) {
	const (
		// A narrow range of possible diagonal elements to put in M makes column swaps less
		// likely to place a short column on the right-hand side of M.
		minDiagonalElementSize = 95
		maxDiagonalElementSize = 100
		numRows                = 17
		numTests               = 100
		log2tolerance          = -50

		// Function being tested
		sortDiagonalIndex      = 0
		sortColumnsByNormIndex = 1
		sortDiagonalLabel      = "sort-diagonal"
		sortColumnsByNormLabel = "sort-columns-by-norm"
	)

	tolerance := math.Pow(2, log2tolerance)
	nilColumnOpCount := []int{0, 0}
	dimensionCounts := []map[int]int{make(map[int]int), make(map[int]int)}
	for testNbr := 0; testNbr < numTests; testNbr++ {
		// Get M and expected results
		m, _, err := createRandomM(
			numRows, minDiagonalElementSize, maxDiagonalElementSize, []int{}, "TestBestSwapInM",
		)

		// Parallel structs to hold actual results
		actual := []struct {
			intOp         *IntOperation
			numIndices    int
			mAsFloat64Sq  []float64
			sortedColumns []int
			skipTest      bool
		}{
			{
				intOp:         nil,
				numIndices:    0,
				mAsFloat64Sq:  nil,
				sortedColumns: make([]int, numRows),
				skipTest:      false,
			},
			{
				intOp:         nil,
				numIndices:    0,
				mAsFloat64Sq:  nil,
				sortedColumns: make([]int, numRows),
				skipTest:      false,
			},
		}

		// Create separate instances of M as float64 for each sorting technique, since that
		// can be modified in place. Also make an unchanging instance to be used for comparison.
		var unchangingMAsFloat64 []float64
		actual[sortDiagonalIndex].mAsFloat64Sq, err = m.AsFloat64()
		require.NoError(t, err)
		actual[sortColumnsByNormIndex].mAsFloat64Sq, err = m.AsFloat64()
		require.NoError(t, err)
		unchangingMAsFloat64, err = m.AsFloat64()

		// Create column operations using the sort-diagonal technique
		// Square entries of M in-place
		for i := 0; i < numRows*numRows; i++ {
			entry := actual[sortDiagonalIndex].mAsFloat64Sq[i]
			actual[sortDiagonalIndex].mAsFloat64Sq[i] = entry * entry
		}
		actual[sortDiagonalIndex].intOp, err = sortDiagonal(
			actual[sortDiagonalIndex].mAsFloat64Sq, numRows, "TestBestColumnOpInM",
		)
		require.NoError(t, err)
		if actual[sortDiagonalIndex].intOp == nil {
			nilColumnOpCount[sortDiagonalIndex]++
			actual[sortDiagonalIndex].skipTest = true
		}

		// Create column operations using the sort-columns-by-norm technique
		for i := 0; i < numRows*numRows; i++ {
			entry := actual[sortColumnsByNormIndex].mAsFloat64Sq[i]
			actual[sortColumnsByNormIndex].mAsFloat64Sq[i] = entry * entry
		}
		actual[sortColumnsByNormIndex].intOp, err = sortColumnsByNorm(
			actual[sortColumnsByNormIndex].mAsFloat64Sq, numRows, "TestBestColumnOpInM",
		)
		require.NoError(t, err)
		if actual[sortColumnsByNormIndex].intOp == nil {
			nilColumnOpCount[sortColumnsByNormIndex]++
			actual[sortColumnsByNormIndex].skipTest = true
		}

		// Set numIndices and increment the count of this sub-matrix dimension
		for _, k := range []int{sortDiagonalIndex, sortColumnsByNormIndex} {
			if !actual[k].skipTest {
				actual[k].numIndices = len(actual[k].intOp.Indices)
				dimensionCounts[k][actual[k].numIndices]++
			}
		}

		// Verify that operations on A and B are inverses
		// The operations on B and A should be inverses of each other
		var areInverses bool
		for _, k := range []int{sortDiagonalIndex, sortColumnsByNormIndex} {
			if actual[k].skipTest {
				continue
			}
			operationOnB := util.CopyIntToInt64(actual[k].intOp.OperationOnB)
			operationOnA := util.CopyIntToInt64(actual[k].intOp.OperationOnA)
			areInverses, err = util.IsInversePair(operationOnB, operationOnA, actual[k].numIndices)
			require.NoError(t, err)
			sortingMethod := "diagonal"
			if k == sortColumnsByNormIndex {
				sortingMethod = "norm index"
			}
			require.True(
				t, areInverses, "sorting method: %s\nintOp:%#v\nm:\n%v\n",
				sortingMethod, actual[k].intOp, m,
			)
		}

		// Initialize numIndices and sortedColumns
		for _, k := range []int{sortDiagonalIndex, sortColumnsByNormIndex} {
			if actual[k].skipTest {
				continue
			}
			actual[k].numIndices = len(actual[k].intOp.Indices)
			for j := 0; j < numRows; j++ {
				actual[k].sortedColumns[j] = j
			}
		}

		// Calculate the column permutations from the intOp fields, and verify that they
		// are permutation matrices.
		for _, k := range []int{sortDiagonalIndex, sortColumnsByNormIndex} {
			if actual[k].skipTest {
				continue
			}

			// Compute the column permutation
			newSortedColumns := make([]int, numRows)
			for j := 0; j < numRows; j++ {
				newSortedColumns[j] = j
			}
			for j := 0; j < actual[k].numIndices; j++ {
				for i := 0; i < actual[k].numIndices; i++ {
					if actual[k].intOp.OperationOnB[i*actual[k].numIndices+j] == 1 {
						// During multiplication on the right by permutation matrix
						// actual[k].intOp.OperationOnB, column results[k].intOp.Indices[i]
						// (the source index), is copied to column results[k].intOp.Indices[j]
						// (the destination index).
						srcColumn := actual[k].intOp.Indices[i]
						destColumn := actual[k].intOp.Indices[j]
						newSortedColumns[destColumn] = actual[k].sortedColumns[srcColumn]
					}
				}
			}
			for j := 0; j < numRows; j++ {
				actual[k].sortedColumns[j] = newSortedColumns[j]
			}

			// Verify that sortedColumns is a permutation
			counts := make([]int, numRows)
			for j := 0; j < numRows; j++ {
				counts[actual[k].sortedColumns[j]]++
			}
			for j := 0; j < numRows; j++ {
				require.Equal(
					t, 1, counts[actual[k].sortedColumns[j]],
					"k: %d, OperationOnB: %v, sortedColumns: %v",
					k, actual[k].intOp.OperationOnB, actual[k].sortedColumns,
				)
			}
		}

		// In the permutation that sorts diagonal elements,
		// - All swaps come in pairs
		// - Partial norms decrease: For each pair (j0, j1) swapped with j0 < j1, the norm of
		//   the entries in rows j0 through j1 is less for column j0 than for column j1
		if !actual[sortDiagonalIndex].skipTest {
			for destColumn := 0; destColumn < numRows; destColumn++ {
				srcColumn := actual[sortDiagonalIndex].sortedColumns[destColumn]
				require.Equal( // all swaps come in pairs
					t, destColumn, actual[sortDiagonalIndex].sortedColumns[srcColumn],
					"destColumn: %d\nsrcColumn:%d\nsortedColumns:%v\nintOp:%#v\nm:\n%v\n",
					destColumn, srcColumn, actual[sortDiagonalIndex].sortedColumns,
					actual[sortColumnsByNormIndex].intOp, m,
				)
				if srcColumn == destColumn {
					// There is nothing to check for columns that remain fixed under the permutation
					continue
				}

				var leftNorm, rightNorm float64
				leftColumn, rightColumn := srcColumn, destColumn
				if destColumn < srcColumn {
					leftColumn = destColumn
					rightColumn = srcColumn
				}
				for i := leftColumn; i <= rightColumn; i++ {
					entry := unchangingMAsFloat64[i*numRows+leftColumn]
					leftNorm += entry * entry
					entry = unchangingMAsFloat64[i*numRows+rightColumn]
					rightNorm += entry * entry
				}
				require.True( // partial norms decrease
					t, leftNorm < rightNorm+tolerance,
					"columns: [%d,%d]\nnorms: [%f,%f]\nsortedColumns:%v\nintOp:%#v\nm:\n%v\n",
					leftColumn, rightColumn, leftNorm, rightNorm,
					actual[sortDiagonalIndex].sortedColumns, actual[sortDiagonalIndex].intOp, m,
				)
			}
		}

		// In the permutation that sorts columns by norm, the norms decrease or remain
		// the same from left to right.
		if !actual[sortColumnsByNormIndex].skipTest {
			lastNorm := math.MaxFloat64
			for j := 0; j < numRows; j++ {
				var norm float64
				for i := 0; i < numRows; i++ {
					srcColumn := actual[sortColumnsByNormIndex].sortedColumns[j]
					entry := unchangingMAsFloat64[i*numRows+srcColumn]
					norm += entry * entry
				}
				require.True(
					t, norm <= lastNorm+tolerance,
					"column: %d\nnorms: [%f,%f]\nsortedColumns:%v\nintOp:%#v\nm:\n%v\n",
					j, lastNorm, norm, actual[sortColumnsByNormIndex].sortedColumns,
					actual[sortColumnsByNormIndex].intOp, m,
				)
			}
		}
	}
	t.Logf(
		"Tests: %d\n"+
			"Nil IntOperations for [%s %s]: [%d %d]\n"+
			"Sub-matrix dimensions for %s: %s\n"+
			"Sub-matrix dimensions for %s: %s\n",
		numTests,
		sortDiagonalLabel, sortColumnsByNormLabel,
		nilColumnOpCount[sortDiagonalIndex], nilColumnOpCount[sortColumnsByNormIndex],
		sortDiagonalLabel, strings.Replace(fmt.Sprintf(
			"%v", dimensionCounts[sortDiagonalIndex]), "map", "", 1,
		),
		sortColumnsByNormLabel, strings.Replace(fmt.Sprintf(
			"%v", dimensionCounts[sortColumnsByNormIndex]), "map", "", 1,
		),
	)
}

func TestPerformRowOp(t *testing.T) {
	const (
		numPossibleEntries = 100
		numRows            = 17
		numCols            = 16
		log2tolerance      = -50
		numTests           = 100
	)

	tolerance := bignumber.NewPowerOfTwo(log2tolerance)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for _, rowOpType := range []string{intPermutation, generalIntOp} {
			for _, involveLastRow := range []bool{true, false} {
				var lastNonZeroEntryInLastRow int
				var actualH, expectedH *bigmatrix.BigMatrix
				var equals bool
				rowOpAsIntOperation, rowOpAsMatrix, _, err := getRandomIntOperation(
					numRows, numCols, rowOpType, involveLastRow, "TestPerformRowOp",
				)
				require.NoError(t, err)
				if involveLastRow {
					require.Equal(t, 2, len(rowOpAsIntOperation.Indices))
					require.Equal(t, numRows-1, rowOpAsIntOperation.Indices[1])
					lastNonZeroEntryInLastRow = rowOpAsIntOperation.Indices[0]
				} else {
					require.LessOrEqual(t, len(rowOpAsIntOperation.Indices), numCols)
					require.GreaterOrEqual(t, len(rowOpAsIntOperation.Indices), 2)
					lastNonZeroEntryInLastRow = -1
				}
				actualH, _, err = createRandomH(numRows, numPossibleEntries, lastNonZeroEntryInLastRow, []int{}, "TestPerformRowOp")
				require.NoError(t, err)
				expectedH, err = bigmatrix.NewEmpty(numRows, numCols).Mul(rowOpAsMatrix, actualH)
				err = rowOpAsIntOperation.performRowOp(actualH, "TestPerformRowOp")
				require.NoError(t, err)
				equals, err = expectedH.Equals(actualH, tolerance)
				require.NoError(t, err)
				require.True(t, equals)
			}
		}
	}
}

func TestPerformColumnOp(t *testing.T) {
	const (
		minDiagonalElementSize = 2
		maxDiagonalElementSize = 100
		numRows                = 17
		log2tolerance          = -50
		numTests               = 100
	)

	tolerance := bignumber.NewPowerOfTwo(log2tolerance)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for _, columnOpType := range []string{intPermutation, generalIntOp} {
			var actualM, expectedM *bigmatrix.BigMatrix
			var equals bool
			columnOpAsIntOperation, _, columnOpAsMatrix, err := getRandomIntOperation(
				numRows, numRows, columnOpType, false, "TestPerformColumnOp",
			)
			require.NoError(t, err)
			require.LessOrEqual(t, len(columnOpAsIntOperation.Indices), numRows)
			require.GreaterOrEqual(t, len(columnOpAsIntOperation.Indices), 2)
			actualM, _, err = createRandomM(
				numRows, minDiagonalElementSize, maxDiagonalElementSize, []int{}, "TestPerformColumnOp",
			)
			expectedM, err = bigmatrix.NewEmpty(numRows, numRows).Mul(actualM, columnOpAsMatrix)
			err = columnOpAsIntOperation.performColumnOp(actualM, "TestPerformColumnOp")
			require.NoError(t, err)
			equals, err = expectedM.Equals(actualM, tolerance)
			require.NoError(t, err)
			require.True(t, equals)
		}
	}
}
