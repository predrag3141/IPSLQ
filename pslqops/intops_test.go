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
		"Count of tests where results were close enough but not exact: %d\n",
		closeEnoughButNotExactCount,
	)
}

// intOpResults holds information needed in TestBestColumnOpInM for expected and actual results
type intOpResults struct {
	numIndices  int
	intOp       *IntOperation
	sortedNorms []float64
}

func TestBestColumnOpInM(t *testing.T) {
	const (
		// A narrow range of possible diagonal elements to put in M makes column swaps less
		// likely to place a short column on the right-hand side of M.
		minDiagonalElementSize = 95
		maxDiagonalElementSize = 100
		numRows                = 17
		numTests               = 100
		expectedIndex          = 0
		actualIndex            = 1
		log2tolerance          = -50

		// Edge cases to count
		columnOpWasNil      = "column op was nil"
		numIndicesDiffer    = "number of indices in column operations differ"
		indicesDiffer       = "indices in column operations differ"
		operationsOnBDiffer = "operations on B differ"
	)

	edgeCaseCounts := make(map[string]int)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		// Initializations
		tolerance := math.Pow(2, log2tolerance)
		results := []intOpResults{
			{numIndices: 0, intOp: nil, sortedNorms: nil}, // expected
			{numIndices: 0, intOp: nil, sortedNorms: nil}, // actual
		}

		// Get M and expected results
		m, expected, err := createRandomM(
			numRows, minDiagonalElementSize, maxDiagonalElementSize, []int{}, "TestBestSwapInM",
		)
		require.NoError(t, err)
		results[expectedIndex].intOp = expected.bestColumnOp
		if results[expectedIndex].intOp != nil {
			require.False(t, results[expectedIndex].intOp.isPermutation())
			results[expectedIndex].numIndices = len(results[expectedIndex].intOp.Indices)
		}

		// Get actual results
		var mAsFloat64 []float64
		mAsFloat64, err = m.AsFloat64()
		require.NoError(t, err)
		results[actualIndex].intOp, err = permutationOfM(mAsFloat64, m.NumRows(), "TestBestSwapInM")
		require.NoError(t, err)
		if results[actualIndex].intOp != nil {
			require.False(t, results[actualIndex].intOp.isPermutation())
			results[actualIndex].numIndices = len(results[actualIndex].intOp.Indices)
		}

		// Compute column norms
		columnNorms := make([]float64, numRows)
		for j := 0; j < numRows; j++ {
			for i := 0; i < numRows; i++ {
				columnNorms[j] += mAsFloat64[i*numRows+j] * mAsFloat64[i*numRows+j]
			}
		}

		// Handle nil column operations
		if (results[expectedIndex].intOp == nil) || (results[actualIndex].intOp == nil) {
			edgeCaseCounts[columnOpWasNil]++
			require.Nil(t, results[expectedIndex].intOp)
			require.Nil(t, results[actualIndex].intOp)

			// If the column operations are nil, the column norms should be sorted
			for j := 1; j < numRows; j++ {
				require.LessOrEqual(
					t, columnNorms[j], columnNorms[j-1]+tolerance,
					"columnNorms[%d] = %f > %f = columnNorms[%d]",
					j, columnNorms[j], columnNorms[j-1], j-1,
				)
			}
			continue
		}

		// Check that the expected and actual integer operations sort the column norms.
		for _, k := range []int{expectedIndex, actualIndex} {
			// Initializations
			expectedOrActual := "expected"
			if k == actualIndex {
				expectedOrActual = "actual"
			}

			// Create results[k].sortedNorms
			results[k].sortedNorms = make([]float64, numRows)
			for j := 0; j < numRows; j++ {
				// If j is a fixed point of the permutation, results[k].sortedNorms[j]
				// will remain untouched by the next block.
				results[k].sortedNorms[j] = columnNorms[j]
			}
			for j := 0; j < results[k].numIndices; j++ {
				for i := 0; i < results[k].numIndices; i++ {
					if results[k].intOp.OperationOnB[i*results[k].numIndices+j] == 1 {
						// Column results[k].intOp.Indices[i] is copied to column
						// results[k].intOp.Indices[j].
						src := results[k].intOp.Indices[i]
						dest := results[k].intOp.Indices[j]
						results[k].sortedNorms[dest] = columnNorms[src]
					}
				}
			}

			// Check that results[k].sortedNorms is sorted
			for j := 1; j < numRows; j++ {
				require.LessOrEqual(
					t, results[k].sortedNorms[j], results[k].sortedNorms[j-1]+tolerance,
					"results[%s].sortedNorms[%d] = %f > %f = results[%s].sortedNorms[%d]",
					expectedOrActual, j, results[k].sortedNorms[j],
					results[k].sortedNorms[j-1], expectedOrActual, j-1,
				)
			}

			// The operations on B and A should be inverses of each other
			var areInverses bool
			operationOnB := util.CopyIntToInt64(results[k].intOp.OperationOnB)
			operationOnA := util.CopyIntToInt64(results[k].intOp.OperationOnA)
			areInverses, err = util.IsInversePair(operationOnB, operationOnA, results[k].numIndices)
			require.NoError(t, err)
			require.True(
				t, areInverses, "%s operationOnB = %v %s operationOnA = %v",
				expectedOrActual, operationOnB, expectedOrActual, operationOnA,
			)
		}

		// In most cases, the expected and actual integer operations are actually equal.
		// But it is not a bug for them to be different, provided that they both sort the
		// column norms. Count these edge cases.
		columnOpsDiffer := false
		var numIndices int
		if results[expectedIndex].numIndices != results[actualIndex].numIndices {
			columnOpsDiffer = true
		}
		if columnOpsDiffer {
			edgeCaseCounts[numIndicesDiffer]++
		} else {
			numIndices = results[expectedIndex].numIndices
			for i := 0; i < numIndices; i++ {
				if results[expectedIndex].intOp.Indices[i] != results[actualIndex].intOp.Indices[i] {
					columnOpsDiffer = true
					break
				}
			}
		}
		if columnOpsDiffer {
			edgeCaseCounts[indicesDiffer]++
		} else {
			for i := 0; i < numIndices*numIndices; i++ {
				if results[expectedIndex].intOp.OperationOnB[i] != results[actualIndex].intOp.OperationOnB[i] {
					columnOpsDiffer = true
					break
				}
			}
		}
		if columnOpsDiffer {
			edgeCaseCounts[operationsOnBDiffer]++
		}

	}
	t.Logf(
		"Edge cases: %s\n",
		strings.Replace(fmt.Sprintf("%v", edgeCaseCounts), "map", "", 1),
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
