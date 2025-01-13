package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bignumber"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/predrag3141/IPSLQ/bigmatrix"
)

func TestGetD(t *testing.T) {
	const (
		numTests           = 100
		numPossibleEntries = 100
		numRows            = 17
		numCols            = 16
	)

	numAllZeroRowsCalculated := 0
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedRowCount := 0; unreducedRowCount < numRows; unreducedRowCount++ {
			// Initializations
			var h *bigmatrix.BigMatrix
			var expected *RandomHInfo
			unreducedRows, err := getRandomUnreducedRows(numRows, unreducedRowCount, "TestGetD")
			require.NoError(t, err)
			require.Len(t, unreducedRows, unreducedRowCount)
			h, expected, err = createRandomH(
				numRows, numPossibleEntries, numCols-1, unreducedRows, "TestGetD",
			)
			require.NoError(t, err)
			d := bigmatrix.NewEmpty(numRows, numRows)

			// Get actual value of D
			var actual struct {
				isIdentity           bool
				calculatedAllZeroRow bool
				hIsReduced           bool
				hUnreducedRow        int
				hUnreducedColumn     int
				dhIsReduced          bool
				dhUnreducedRow       int
				dhUnreducedColumn    int
			}
			actual.isIdentity, actual.calculatedAllZeroRow, err = getD(h, d, "TestGetD")
			require.NoError(t, err)
			if actual.calculatedAllZeroRow {
				numAllZeroRowsCalculated++
			}

			// Check whether H is reduced before being left-multiplied by D
			actual.hIsReduced, actual.hUnreducedRow, actual.hUnreducedColumn, err = isRowReduced(
				h, bigNumberBitTolerance, "TestGetD",
			)
			require.NoError(t, err)
			if unreducedRowCount == 0 {
				require.True(t, actual.hIsReduced)
			} else {
				require.False(t, actual.hIsReduced)
			}
			require.Equal(t, expected.unreducedRow, actual.hUnreducedRow)
			require.Equal(t, expected.unreducedColumn, actual.hUnreducedColumn)

			// Check isIdentity and the known entries of D on and above the diagonal
			zero := bignumber.NewFromInt64(0)
			one := bignumber.NewFromInt64(1)
			if unreducedRowCount == 0 {
				require.True(t, actual.isIdentity)
			} else {
				require.False(t, actual.isIdentity)
			}
			for i := 0; i < numRows; i++ {
				var diagonalElement *bignumber.BigNumber
				diagonalElement, err = d.Get(i, i)
				require.NoError(t, err)
				equals := diagonalElement.Equals(one, zero)
				require.True(t, equals)
				err = isLowerQuadrangular(d, "TestGetD")
				require.NoError(t, err)
			}

			// Check that D reduces H
			var dh *bigmatrix.BigMatrix
			dh, err = bigmatrix.NewEmpty(numRows, numRows).Mul(d, h)
			require.NoError(t, err)
			err = isLowerQuadrangular(dh, "TestGetD")
			require.NoError(t, err)
			actual.dhIsReduced, actual.dhUnreducedRow, actual.dhUnreducedColumn, err = isRowReduced(
				dh, bigNumberBitTolerance, "TestGetD",
			)
			require.NoError(t, err)
			require.True(t, actual.dhIsReduced)
			require.Equal(t, -1, actual.dhUnreducedRow)
			require.Equal(t, -1, actual.dhUnreducedColumn)
		}
	}
	fmt.Printf("Number of all zero rows calculated: %d / %d\n", numAllZeroRowsCalculated, numTests)
}

func TestGetE(t *testing.T) {
	const (
		numTests               = 100
		maxDiagonalElementSize = 100
		eNumRows               = 17
		mNumRows               = 16
	)

	numAllZeroColumnsCalculated := 0
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedColumnCount := 0; unreducedColumnCount < mNumRows; unreducedColumnCount++ {
			// Initializations
			var m *bigmatrix.BigMatrix
			var expected *RandomMInfo
			unreducedColumns, err := getRandomUnreducedColumns(mNumRows, unreducedColumnCount, "TestGetE")
			require.Len(t, unreducedColumns, unreducedColumnCount)
			m, expected, err = createRandomM(
				mNumRows, 2, maxDiagonalElementSize, unreducedColumns, "TestGetE",
			)
			require.NoError(t, err)
			e := bigmatrix.NewEmpty(eNumRows, eNumRows)

			// Get actual value of D
			var actual struct {
				isIdentity              bool
				calculatedAllZeroColumn bool
				mIsReduced              bool
				mUnreducedRow           int
				mUnreducedColumn        int
				meIsReduced             bool
				meUnreducedRow          int
				meUnreducedColumn       int
			}
			actual.isIdentity, actual.calculatedAllZeroColumn, err = getE(m, e, "TestGetE")
			require.NoError(t, err)
			if actual.calculatedAllZeroColumn {
				numAllZeroColumnsCalculated++
			}

			// Check whether H is reduced before being left-multiplied by D
			actual.mIsReduced, actual.mUnreducedRow, actual.mUnreducedColumn, err = isColumnReduced(
				m, bigNumberBitTolerance, "TestGetE",
			)
			require.NoError(t, err)
			if unreducedColumnCount == 0 {
				require.True(t, actual.mIsReduced)
			} else {
				require.False(t, actual.mIsReduced)
			}
			require.Equal(t, expected.unreducedRow, actual.mUnreducedRow)
			require.Equal(t, expected.unreducedColumn, actual.mUnreducedColumn)

			// Check isIdentity and the known entries of D on and above the diagonal
			zero := bignumber.NewFromInt64(0)
			one := bignumber.NewFromInt64(1)
			if unreducedColumnCount == 0 {
				require.True(t, actual.isIdentity)
			} else {
				require.False(t, actual.isIdentity)
			}
			for i := 0; i < eNumRows; i++ {
				var diagonalElement *bignumber.BigNumber
				diagonalElement, err = e.Get(i, i)
				require.NoError(t, err)
				equals := diagonalElement.Equals(one, zero)
				require.True(t, equals)
				err = isLowerQuadrangular(e, "TestGetE")
				require.NoError(t, err)
			}

			// Check that D reduces H
			var me *bigmatrix.BigMatrix
			me, err = bigmatrix.NewEmpty(mNumRows, mNumRows).MulUpperLeft(m, e)
			require.NoError(t, err)
			err = isLowerQuadrangular(me, "TestGetE")
			require.NoError(t, err)
			actual.meIsReduced, actual.meUnreducedRow, actual.meUnreducedColumn, err = isColumnReduced(
				me, bigNumberBitTolerance, "TestGetE",
			)
			require.NoError(t, err)
			require.True(t, actual.meIsReduced)
			require.Equal(t, -1, actual.meUnreducedRow)
			require.Equal(t, -1, actual.meUnreducedColumn)
		}
	}
	fmt.Printf("Number of all zero columns calculated: %d / %d\n", numAllZeroColumnsCalculated, numTests)
}

func TestIsRowReduced(t *testing.T) {
	const (
		numTests           = 100
		numPossibleEntries = 100
		numRows            = 17
		numCols            = 16
	)

	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedRowCount := 0; unreducedRowCount < numRows; unreducedRowCount++ {
			// Initializations
			var h *bigmatrix.BigMatrix
			var expected *RandomHInfo
			unreducedRows, err := getRandomUnreducedRows(numRows, unreducedRowCount, "TestIsRowReduced")
			require.NoError(t, err)
			require.Len(t, unreducedRows, unreducedRowCount)
			h, expected, err = createRandomH(
				numRows, numPossibleEntries, numCols-1, unreducedRows, "TestIsRowReduced",
			)
			require.NoError(t, err)

			// Get actual values
			var actual struct {
				isReduced       bool
				unreducedRow    int
				unreducedColumn int
			}
			actual.isReduced, actual.unreducedRow, actual.unreducedColumn, err = isRowReduced(
				h, -bigNumberBitTolerance, "TestIsRowReduced",
			)
			require.NoError(t, err)

			// Compare expected to actual
			require.Equal(t, expected.unreducedRow, actual.unreducedRow)
			require.Equal(t, expected.unreducedColumn, actual.unreducedColumn)
			if unreducedRowCount == 0 {
				require.True(t, actual.isReduced)
			} else {
				require.False(t, actual.isReduced)
			}
		}
	}
}

func TestIsColumnReduced(t *testing.T) {
	const (
		numTests         = 100
		maxDiagonalEntry = 100
		numRows          = 17
	)

	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedColumnCount := 0; unreducedColumnCount < numRows; unreducedColumnCount++ {
			// Initializations
			var m *bigmatrix.BigMatrix
			var expected *RandomMInfo
			unreducedColumns, err := getRandomUnreducedColumns(
				numRows, unreducedColumnCount, " TestIsColumnReduced",
			)
			require.NoError(t, err)
			require.Len(t, unreducedColumns, unreducedColumnCount)
			m, expected, err = createRandomM(
				numRows, 2, maxDiagonalEntry, unreducedColumns, "TestIsColumnReduced",
			)
			require.NoError(t, err)

			// Get actual values
			var actual struct {
				isReduced       bool
				unreducedRow    int
				unreducedColumn int
			}
			actual.isReduced, actual.unreducedRow, actual.unreducedColumn, err = isColumnReduced(
				m, -bigNumberBitTolerance, "TestIsColumnReduced",
			)
			require.NoError(t, err)

			// Compare expected to actual
			require.Equal(t, expected.unreducedRow, actual.unreducedRow)
			require.Equal(t, expected.unreducedColumn, actual.unreducedColumn)
			if unreducedColumnCount == 0 {
				require.True(t, actual.isReduced)
			} else {
				require.False(t, actual.isReduced)
			}
		}
	}
}

func getRandomUnreducedRows(numRows, numUnreducedRows int, caller string) ([]int, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getRandomUnreducedRows", caller)

	// Check input
	if numRows <= numUnreducedRows {
		return nil, fmt.Errorf(
			"%s: numRows = %d <= %d = numUnreducedRows", caller, numRows, numUnreducedRows,
		)
	}

	// Get and return unreduced rows array
	var unreducedRows []int
	switch numUnreducedRows {
	case 0:
		unreducedRows = []int{}
	case 1:
		// Avoid choosing row 0, which has no entries below the diagonal
		unreducedRows = []int{1 + rand.Intn(numRows-1)}
	default:
		// As in the case of unreducedRowCount being 2, avoid choosing row 0
		unreducedRows = getRandomIndices(numUnreducedRows, numRows-1)
		for i := 0; i < numUnreducedRows; i++ {
			unreducedRows[i]++
		}
	}
	return unreducedRows, nil
}

func getRandomUnreducedColumns(numRows, numUnreducedColumns int, caller string) ([]int, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getRandomUnreducedColumns", caller)

	// Check input
	if numRows <= numUnreducedColumns {
		return nil, fmt.Errorf(
			"%s: numRows = %d <= %d = numUnreducedRows", caller, numRows, numUnreducedColumns,
		)
	}

	// Get and return unreduced columns array
	var unreducedColumns []int
	switch numUnreducedColumns {
	case 0:
		unreducedColumns = []int{}
	case 1:
		// Avoid choosing column numRows-1, which has no entries below the diagonal
		unreducedColumns = []int{rand.Intn(numRows - 1)}
	default:
		// As in the case of unreducedColumnCount being 2, avoid choosing row 0
		unreducedColumns = getRandomIndices(numUnreducedColumns, numRows-1)
	}
	return unreducedColumns, nil
}

func isLowerQuadrangular(x *bigmatrix.BigMatrix, caller string) error {
	caller = fmt.Sprintf("%s-isLowerQuadrangular", caller)
	numRows, numCols := x.Dimensions()
	var err error
	for i := 0; i < numRows-1; i++ {
		for j := i + 1; j < numCols; j++ {
			var upperElement *bignumber.BigNumber
			upperElement, err = x.Get(i, j)
			if err != nil {
				return fmt.Errorf(
					"%s: could not get X[%d][%d]: %q", caller, i, j, err.Error(),
				)
			}
			if !upperElement.IsZero() {
				_, upperElementAsStr := upperElement.String()
				return fmt.Errorf(
					"%s: X[%d][%d] = %s != 0", caller, i, j, upperElementAsStr,
				)
			}
		}
	}
	return nil
}
