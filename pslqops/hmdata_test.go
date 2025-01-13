package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"math"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/predrag3141/IPSLQ/bignumber"
)

func TestNewDiagonalStatisticsOnH(t *testing.T) {
	const (
		numPossibleEntries = 100
		numRows            = 10
		numCols            = 9
		log2tolerance      = -50
		numTests           = 10
	)

	toleranceAsBigNumber := bignumber.NewPowerOfTwo(log2tolerance)
	toleranceAsFloat64 := math.Pow(2.0, log2tolerance)

	for testNbr := 0; testNbr < numTests; testNbr++ {
		// Initializations
		h, expected, err := createRandomH(
			numRows, numPossibleEntries, numCols-1, []int{}, "TestNewDiagonalStatisticsOnH",
		)
		require.NoError(t, err)
		actual := struct {
			diagonalStatistics *DiagonalStatistics
		}{
			diagonalStatistics: nil,
		}

		actual.diagonalStatistics, err = NewDiagonalStatistics(h)
		for i := 0; i < numCols; i++ {
			var equals bool
			equals = expected.diagonalStatistics.Diagonal[i].Equals(
				actual.diagonalStatistics.Diagonal[i], toleranceAsBigNumber,
			)
			require.Truef(t, equals, "absDiff > tolerance")
		}
		require.NotNil(t, expected.diagonalStatistics.Ratio)
		require.NotNil(t, actual.diagonalStatistics.Ratio)
		require.Less(t,
			math.Abs(*expected.diagonalStatistics.Ratio-*actual.diagonalStatistics.Ratio),
			toleranceAsFloat64,
		)
	}
}

func TestNewDiagonalStatisticsOnM(t *testing.T) {
	const (
		minDiagonalEntry = 2
		maxDiagonalEntry = 100
		numRows          = 10
		log2tolerance    = -50
		numTests         = 10
	)

	toleranceAsBigNumber := bignumber.NewPowerOfTwo(log2tolerance)
	toleranceAsFloat64 := math.Pow(2.0, log2tolerance)

	for testNbr := 0; testNbr < numTests; testNbr++ {
		// Initializations
		m, expected, err := createRandomM(
			numRows, minDiagonalEntry, maxDiagonalEntry, []int{}, "TestNewDiagonalStatisticsOnM",
		)
		require.NoError(t, err)
		actual := struct {
			diagonalStatistics *DiagonalStatistics
		}{
			diagonalStatistics: nil,
		}

		actual.diagonalStatistics, err = NewDiagonalStatistics(m)
		for i := 0; i < numRows; i++ {
			var equals bool
			equals = expected.diagonalStatistics.Diagonal[i].Equals(
				actual.diagonalStatistics.Diagonal[i], toleranceAsBigNumber,
			)
			require.Truef(t, equals, "absDiff > tolerance")
		}
		require.NotNil(t, expected.diagonalStatistics.Ratio)
		require.NotNil(t, actual.diagonalStatistics.Ratio)
		require.Less(t,
			math.Abs(*expected.diagonalStatistics.Ratio-*actual.diagonalStatistics.Ratio),
			toleranceAsFloat64,
		)
	}
}

func TestBottomRightOfH(t *testing.T) {
	const (
		numPossibleEntries = 100
		numRows            = 18
		numCols            = 17
		log2tolerance      = -50
		numTests           = 10
		log2threshold      = -20
	)

	toleranceAsBigNumber := bignumber.NewPowerOfTwo(log2tolerance)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		// Initializations
		var lastNonZeroColumn int
		lastNonZeroColumnEqualsMinusOne := rand.Intn(3)
		if lastNonZeroColumnEqualsMinusOne == 1 {
			lastNonZeroColumn = -1
		} else {
			lastNonZeroColumn = rand.Intn(numCols)
		}
		h, expected, err := createRandomH(
			numRows, numPossibleEntries, lastNonZeroColumn, []int{}, "TestBottomRightOfH",
		)
		require.NoError(t, err)
		actual := struct {
			bottomRightOfH *BottomRightOfH
			bestLastRowOp  *IntOperation
		}{
			bottomRightOfH: nil,
			bestLastRowOp:  nil,
		}

		// Get the actual BottomRightOfH
		actual.bottomRightOfH, err = getBottomRightOfH(h, "TestBottomRightOfH")

		// Check Found flag
		require.Equal(t, expected.bottomRightOfH.Found, actual.bottomRightOfH.Found)

		// Check RowNumberOfT and RowNumberOfU
		require.Equal(t, expected.bottomRightOfH.RowNumberOfT, actual.bottomRightOfH.RowNumberOfT)
		require.Equal(t, expected.bottomRightOfH.RowNumberOfU, actual.bottomRightOfH.RowNumberOfU)

		// Check T and U
		if !expected.bottomRightOfH.Found {
			require.Nil(t, expected.bottomRightOfH.T)
			require.Nil(t, expected.bottomRightOfH.U)
			require.Nil(t, actual.bottomRightOfH.T)
			require.Nil(t, actual.bottomRightOfH.U)
			continue
		}

		// Check original T and U
		require.True(t, expected.bottomRightOfH.T.Equals(
			actual.bottomRightOfH.T, toleranceAsBigNumber,
		))
		require.True(t, expected.bottomRightOfH.U.Equals(
			actual.bottomRightOfH.U, toleranceAsBigNumber,
		))

		// Check reduction of T and U. Calling reduce() modifies T and U in
		// actual.bottomRightOfH, so these two BigNumbers are saved off for future testing.
		// compareRowOps returns a "close enough flag" for diagonal row op testing,
		// which this test ignores.
		actualUnreducedT := bignumber.NewFromBigNumber(actual.bottomRightOfH.T)
		actualUnreducedU := bignumber.NewFromBigNumber(actual.bottomRightOfH.U)
		actual.bestLastRowOp, err = actual.bottomRightOfH.reduce(
			1<<-log2threshold, log2threshold, "TestBottomRightOfH",
		)
		_, err = compareRowOps(
			expected.bottomRightOfH.T, actualUnreducedT,
			expected.bottomRightOfH.U, actualUnreducedU,
			bignumber.NewFromInt64(0), bignumber.NewFromInt64(0), // v is zero!
			expected.bestLastRowOp, actual.bestLastRowOp,
			log2tolerance, "TestBottomRightOfH",
		)
		require.NoError(t, err)
	}
}
