package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
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
		"Count of tests where results were close enough but not exact: %d\n",
		closeEnoughButNotExactCount,
	)
}

func TestBestSwapInM(t *testing.T) {
	const (
		// A narrow range of possible diagonal elements to put in M makes column swaps less
		// likely to place a short column on the right-hand side of M.
		minDiagonalElementSize = 95
		maxDiagonalElementSize = 100
		numRows                = 17
		numTests               = 100

		// Edge cases to count
		columnOpWasNil             = "column op was nil"
		leftmostColumnsDiffered    = "leftmost columns differed"
		leftmostColumnsWereTheSame = "leftmost columns were the same"
	)

	edgeCaseCounts := make(map[string]int)
	rightmostColumnCounts := make(map[int]int)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		// Get M and expected results
		m, expected, err := createRandomM(
			numRows, minDiagonalElementSize, maxDiagonalElementSize, []int{}, "TestBestSwapInM",
		)
		require.NoError(t, err)
		if expected.bestColumnOp != nil {
			require.True(t, expected.bestColumnOp.isPermutation())
			require.Equal(t, 2, len(expected.bestColumnOp.Indices))
		}

		// Get actual results
		var actualColumnOp *IntOperation
		actualColumnOp, err = bestSwapInM(m, "TestBestSwapInM")
		require.NoError(t, err)
		if actualColumnOp != nil {
			require.True(t, actualColumnOp.isPermutation())
			require.Equal(t, 2, len(actualColumnOp.Indices))
		}

		// The expected and actual right-most columns in the column operations should match
		if (expected.bestColumnOp == nil) || (actualColumnOp == nil) {
			edgeCaseCounts[columnOpWasNil]++
			require.Nil(t, expected.bestColumnOp)
			require.Nil(t, actualColumnOp)
		} else {
			require.Equal(t, expected.bestColumnOp.Indices[1], actualColumnOp.Indices[1])
			rightmostColumnCounts[expected.bestColumnOp.Indices[1]]++
			if expected.bestColumnOp.Indices[0] != actualColumnOp.Indices[0] {
				edgeCaseCounts[leftmostColumnsDiffered]++
			} else {
				edgeCaseCounts[leftmostColumnsWereTheSame]++
			}
		}
	}
	t.Logf(
		"Edge cases: %s\n",
		strings.Replace(fmt.Sprintf("%v", edgeCaseCounts), "map", "", 1),
	)
	t.Logf(
		"Rightmost columns: %s\n",
		strings.Replace(fmt.Sprintf("%v", rightmostColumnCounts), "map", "", 1),
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
