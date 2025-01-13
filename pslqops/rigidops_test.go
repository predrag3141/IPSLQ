package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"math/rand"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
)

func TestGivensRotationOnH(t *testing.T) {
	// The first three rows and columns of H are the same as in an example
	// from https://en.wikipedia.org/wiki/Givens_rotation as of Dec 25, 2024. The
	// result of the two Givens transformations should match the equivalent result
	// in the article (except rows there correspond to columns here).
	//
	// The starting matrix in the example from Wikepedia (transposing) is
	//   _         _
	//  |  6  5  0  |
	//  |  5  1  4  |
	//  |_ 0  4  3 _|
	//
	// The output, in that 3x3 sub-matrix, after the second transformation,
	// should be approximately the following, copied (albeit transposed) from
	// Wikipedia:
	//  _                         _
	// |  7.8102   0       0       |
	// |  4.4813   4.6817  0       |
	// |_ 2.5607   0.9665 -4.1843 _|.

	// Initializations
	var rowLengths0 *bigmatrix.BigMatrix
	tolerance := bignumber.NewPowerOfTwo(-bigNumberBitTolerance)
	hEntries := []int64{6, 5, 0, 5, 1, 4, 0, 4, 3, 6, -3, 2}
	h, err := bigmatrix.NewFromInt64Array(hEntries, 4, 3)
	require.NoError(t, err)
	rowLengths0, err = getRowLengths(h, "TestGivensRotationOnH")

	// First Givens rotation, erasing H[0][1]
	var equals bool
	var rowLengths1 *bigmatrix.BigMatrix
	err = givensRotationOnH(h, 0, 1, "TestGivensRotationOnH")
	require.NoError(t, err)
	rowLengths1, err = getRowLengths(h, "TestGivensRotationOnH")
	require.NoError(t, err)
	equals, err = rowLengths0.Equals(rowLengths1, tolerance)
	require.NoError(t, err)
	require.True(t, equals)

	// Second Givens rotation, erasing H[1][2]
	var rowLengths2 *bigmatrix.BigMatrix
	err = givensRotationOnH(h, 1, 2, "TestGivensRotationOnH")
	require.NoError(t, err)
	rowLengths2, err = getRowLengths(h, "TestGivensRotationOnH")
	equals, err = rowLengths0.Equals(rowLengths2, tolerance)
	require.NoError(t, err)
	require.True(t, equals)

	// Compare determinants
	var dp *bignumber.BigNumber
	detAsInt := hEntries[0] * (hEntries[4]*hEntries[8] - hEntries[5]*hEntries[7])
	detAsInt -= hEntries[3] * (hEntries[1]*hEntries[8] - hEntries[2]*hEntries[7])
	detAsInt += hEntries[6] * (hEntries[1]*hEntries[5] - hEntries[2]*hEntries[4])
	dp, err = diagonalProduct(h, "TestGivensRotationOnH")
	require.True(
		t, dp.Equals(bignumber.NewFromInt64(detAsInt), tolerance),
	)
}

func TestRemoveCornerOfH(t *testing.T) {
	// This test creates a somewhat realistic scenario in which RH has just been
	// computed with a corner that needs to be removed, and verifies that the corner is
	// removed without changing certain invariants. Here,
	// - R is a column operation
	// - S is the inverse of R
	// - H is a lower quadrangular matrix
	//
	// Part of the setup is that x is a 1 by numRows matrix for which xH = 0, just as in
	// the actual PSLQ algorithm. The invariants that are maintained are:
	// - The determinant of H (excluding its bottom row) remains the same throughout
	// - All of the row norms of H remain the same
	// - The fact that xH = 0 is maintained at each of three stages:
	//   - before updating H with H <- RH
	//   - after updating H with H <- RH and x with x <- xS (using the variables rh and xs)
	//   - after removing the corner from H (still using the variables, xs and rh)
	//
	// The invariants xSRH = 0, etc. lend credence to the expectation that only matrix multiplies
	// are being performed throughout.
	const (
		numRows            = 7
		numCols            = 6
		numPossibleEntries = 100
		numTests           = 10
		minSeed            = 12955
		seedIncrement      = 1000
	)

	tolerance := bignumber.NewPowerOfTwo(-bigNumberBitTolerance)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		rand.Seed(int64(minSeed + testNbr*seedIncrement))
		for _, rowOpType := range []string{intPermutation, generalIntOp} {
			// Generate X and H with <X,H> = 0
			x, h, err := getRandomXandH(
				numRows, numCols, numPossibleEntries, tolerance, "TestRemoveCornerOfH",
			)
			require.NoError(t, err)

			// Compute the expected determinant of H
			var expectedDeterminant *bignumber.BigNumber
			expectedDeterminant, err = diagonalProduct(h, "TestRemoveCornerOfH")
			if rowOpType == intPermutation {
				expectedDeterminant.Sub(bignumber.NewFromInt64(0), expectedDeterminant)
			}

			// Generate an IntOperation and the equivalent R and S matrices
			var io *IntOperation
			var r, s *bigmatrix.BigMatrix
			io, r, s, err = getRandomIntOperation(
				numRows, numCols, rowOpType, false, "TestRemoveCornerOfH",
			)

			// Apply the row operation, R, to H to get RH
			rh := bigmatrix.NewEmpty(numRows, numCols)
			_, err = rh.Mul(r, h)
			require.NoError(t, err)

			// xSRH should be zero, since SR = I and xIH = xH = 0
			var equals bool
			xs := bigmatrix.NewEmpty(1, numRows)
			_, err = xs.Mul(x, s)
			xsrh := bigmatrix.NewEmpty(numRows, numCols)
			_, err = xsrh.Mul(xs, rh)
			require.NoError(t, err)
			equals, err = bigmatrix.NewEmpty(1, numCols).Equals(xsrh, tolerance)
			require.NoError(t, err)
			require.True(t, equals)

			// Remove the corner from RH
			var rowLengthsBefore *bigmatrix.BigMatrix
			rowLengthsBefore, err = getRowLengths(rh, "TestRemoveCornerOfH")
			err = removeCornerOfH(rh, io.Indices, "TestRemoveCornerOfH")
			require.NoError(t, err)

			// rh (the variable) is actually RHQ now, where Q is the orthogonal matrix
			// we never get to see, which just removed the corner from RH. Check that
			// RHQ has zeroes above the diagonal
			var zeroesAreAboveDiagonal bool
			zeroesAreAboveDiagonal, err = hasZeroesAboveTheDiagonal(
				rh, tolerance, "TestRemoveCornerOfH",
			)
			require.NoError(t, err)
			require.True(t, zeroesAreAboveDiagonal)

			// Since RHQ has zeroes above the diagonal, the original determinant should
			// now be the product of diagonal elements in RHQ.
			var actualDeterminant *bignumber.BigNumber
			actualDeterminant, err = diagonalProduct(rh, "TestRemoveCornerOfH")
			require.NoError(t, err)
			equals = expectedDeterminant.Equals(actualDeterminant, tolerance)
			require.True(t, equals, "TestRemoveCornerOfH")

			// Removing the corner from RH should not have changed the Euclidean lengths of its rows
			var rowLengthsAfter *bigmatrix.BigMatrix
			rowLengthsAfter, err = getRowLengths(rh, "TestRemoveCornerOfH")
			require.NoError(t, err)
			equals, err = rowLengthsBefore.Equals(rowLengthsAfter, tolerance)
			require.NoError(t, err)
			require.True(t, equals)

			// Like xSRH, xSRHQ should be zero, since SR = I, xIH = xH = 0, and corner removal is
			// a matrix multiply by (the unseen) Q.
			xsrhq := bigmatrix.NewEmpty(numRows, numCols)
			_, err = xsrhq.Mul(xs, rh)
			require.NoError(t, err)
			equals, err = bigmatrix.NewEmpty(1, numCols).Equals(xsrhq, tolerance)
			require.NoError(t, err)
			require.True(t, equals)
		}
	}
}

func TestGivensRotationOnM(t *testing.T) {
	// M is the same as in an example from https://en.wikipedia.org/wiki/Givens_rotation
	// as of Dec 25, 2024. The starting matrix in the example from Wikepedia is
	//   _         _
	//  |  6  5  0  |
	//  |  5  1  4  |
	//  |_ 0  4  3 _|
	//
	// This example can be used to compare what happens in the other Givens test,
	// involving H, but not necessarily what happens in this test. The reason is that
	// the row operations in GivensRotationOnM zero out the entries above the diagonal,
	// whereas in the Wikipedia article they zero out the entries below the diagonal.

	// Initializations
	var columnLengths0 *bigmatrix.BigMatrix
	tolerance := bignumber.NewPowerOfTwo(-bigNumberBitTolerance)
	mEntries := []int64{6, 5, 0, 5, 1, 4, 0, 4, 3}
	m, err := bigmatrix.NewFromInt64Array(mEntries, 3, 3)
	require.NoError(t, err)
	columnLengths0, err = getColumnLengths(m, "TestGivensRotationOnM")

	// First Givens rotation, erasing M[1][2]
	var equals bool
	var columnLengths1 *bigmatrix.BigMatrix
	err = givensRotationOnM(m, 1, 2, "TestGivensRotationOnM")
	require.NoError(t, err)
	columnLengths1, err = getColumnLengths(m, "TestGivensRotationOnM")
	require.NoError(t, err)
	equals, err = columnLengths0.Equals(columnLengths1, tolerance)
	require.NoError(t, err)
	require.True(t, equals)

	// Second Givens rotation, erasing M[0][1]
	var columnLengths2 *bigmatrix.BigMatrix
	err = givensRotationOnM(m, 0, 1, "TestGivensRotationOnM")
	require.NoError(t, err)
	columnLengths2, err = getColumnLengths(m, "TestGivensRotationOnM")
	equals, err = columnLengths0.Equals(columnLengths2, tolerance)
	require.NoError(t, err)
	require.True(t, equals)

	// Compare determinants before and after
	var dp *bignumber.BigNumber
	detAsInt := mEntries[0] * (mEntries[4]*mEntries[8] - mEntries[5]*mEntries[7])
	detAsInt -= mEntries[3] * (mEntries[1]*mEntries[8] - mEntries[2]*mEntries[7])
	detAsInt += mEntries[6] * (mEntries[1]*mEntries[5] - mEntries[2]*mEntries[4])
	dp, err = diagonalProduct(m, "TestGivensRotationOnM")
	require.NoError(t, err)
	require.True(
		t, dp.Equals(bignumber.NewFromInt64(detAsInt), tolerance),
	)
}

func TestRemoveCornerOfM(t *testing.T) {
	// This test creates a somewhat realistic scenario in which MS has just been
	// computed with a corner that needs to be removed, and verifies that the corner is
	// removed without changing certain invariants. Here,
	// - S is a column operation
	// - R is the inverse of S
	// - M is a lower triangular matrix
	//
	// Part of the setup, implemented in getRandomMandX(), is that x is a numRows by 1 matrix
	// for which Mx = [1,0,...,0]-transpose. There is a test at the end that exploits identities,
	// explained in comments with this test, to confirm that corner removal is a left-multiply by
	// a matrix with a norm-1 left-most column (consistent with an orthogonal matrix).
	//
	// Invariants that are maintained are:
	// - The determinant of M remains the same throughout
	// - All of the column norms of M remain the same.
	//
	// The invariants and the test exploiting Mx = [1,0,...,0]-transpose lend credence to the
	// expectation that only matrix multiplies are being performed throughout.
	const (
		numRows            = 6
		numPossibleEntries = 100
		numTests           = 10
		minSeed            = 266443
		seedIncrement      = 1000
	)

	tolerance := bignumber.NewPowerOfTwo(-bigNumberBitTolerance)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		rand.Seed(int64(minSeed + testNbr*seedIncrement))
		for _, columnOpType := range []string{intPermutation, generalIntOp} {
			// Generate X and M with <X,H> = 0
			m, x, err := getRandomMandX(
				numRows, numPossibleEntries, tolerance, "TestRemoveCornerOfH",
			)
			require.NoError(t, err)

			// Compute the expected determinant of M
			var expectedDeterminant *bignumber.BigNumber
			expectedDeterminant, err = diagonalProduct(m, "TestRemoveCornerOfH")
			if columnOpType == intPermutation {
				expectedDeterminant.Sub(bignumber.NewFromInt64(0), expectedDeterminant)
			}

			// Generate an IntOperation and the equivalent R and S matrices
			var io *IntOperation
			var r, s *bigmatrix.BigMatrix
			io, r, s, err = getRandomIntOperation(
				numRows, numRows, columnOpType, false, "TestRemoveCornerOfH",
			)

			// Apply the column operation, S, to M to get MS
			ms := bigmatrix.NewEmpty(numRows, numRows)
			_, err = ms.Mul(m, s)
			require.NoError(t, err)

			// Remove the corner from MS
			var columnLengthsBefore *bigmatrix.BigMatrix
			columnLengthsBefore, err = getColumnLengths(ms, "TestRemoveCornerOfM")
			require.NoError(t, err)
			err = removeCornerOfM(ms, io.Indices, "TestRemoveCornerOfM")
			require.NoError(t, err)

			// ms (the variable) is actually QMS now, where Q is the orthogonal matrix
			// we never get to see, which just removed the corner from MS. Check that
			// QMS has zeroes above the diagonal
			var zeroesAreAboveDiagonal bool
			zeroesAreAboveDiagonal, err = hasZeroesAboveTheDiagonal(
				ms, tolerance, "TestRemoveCornerOfM",
			)
			require.NoError(t, err)
			require.True(t, zeroesAreAboveDiagonal)

			// Since QMS has zeroes above the diagonal, the original determinant should
			// now be the product of diagonal elements in QMS.
			var actualDeterminant *bignumber.BigNumber
			var equals bool
			actualDeterminant, err = diagonalProduct(ms, "TestRemoveCornerOfM")
			require.NoError(t, err)
			equals = expectedDeterminant.Equals(actualDeterminant, tolerance)
			require.True(t, equals, "TestRemoveCornerOfM")

			// Removing the corner from MS should not have changed the Euclidean lengths of its columns
			var columnLengthsAfter *bigmatrix.BigMatrix
			columnLengthsAfter, err = getColumnLengths(ms, "TestRemoveCornerOfM")
			equals, err = columnLengthsBefore.Equals(columnLengthsAfter, tolerance)
			require.NoError(t, err)
			require.True(t, equals)

			// M and x were selected so that Mx = [x[0],0,0,...,0]-transpose and x[0] = 1.
			// Therefore,
			// - MSRx = (M)(I)(x) = Mx = [x[0],0,0,...,0]-transpose = [1,0,0,...,0]-transpose.
			// - QMSRx = (x[0])(column 0 of Q) = column 0 of Q.
			//
			// Since Q is an orthogonal matrix, each column of Q has norm 1, so
			// |QMSRx| = |column 0 of Q| = 1.
			//
			// To confirm that |QMSRx| = 1, multiply Like |MSRx|, |QMSRx| should be |x[0]|, since Q preserves row len|MSRx| = , since SR = I, MIx = Mx = 0, and corner removal is
			// a matrix multiply by an (unseen) orthogonal matrix, Q.
			var qmsrx, qmsrxTranspose, qmsrxSq *bigmatrix.BigMatrix
			var qmsrxSq00 *bignumber.BigNumber
			rx := bigmatrix.NewEmpty(numRows, 1)
			_, err = rx.Mul(r, x)
			require.NoError(t, err)
			qmsrx, err = bigmatrix.NewEmpty(numRows, 1).Mul(ms, rx)
			_, err = qmsrx.Mul(ms, rx)
			qmsrxTranspose, err = bigmatrix.NewEmpty(1, numRows).Transpose(qmsrx)
			require.NoError(t, err)
			qmsrxSq, err = bigmatrix.NewEmpty(1, 1).Mul(qmsrxTranspose, qmsrx)
			qmsrxSq00, err = qmsrxSq.Get(0, 0)
			require.NoError(t, err)
			equals = qmsrxSq00.Equals(bignumber.NewFromInt64(1), tolerance)
			require.NoError(t, err)
			require.True(t, equals)
		}
	}
}

// getRowLengths returns the lengths of rows in H
func getRowLengths(h *bigmatrix.BigMatrix, caller string) (*bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getRowLengths", caller)
	retVal := bigmatrix.NewEmpty(1, h.NumRows())
	for i := 0; i < h.NumRows(); i++ {
		retValISq := bignumber.NewFromInt64(0)
		var err error
		var retValI *bignumber.BigNumber
		for j := 0; j < h.NumCols(); j++ {
			var hIJ *bignumber.BigNumber
			hIJ, err = h.Get(i, j)
			if err != nil {
				return nil, fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, i, j, err.Error())
			}
			retValISq.MulAdd(hIJ, hIJ)
		}
		retValI, err = bignumber.NewFromInt64(0).Sqrt(retValISq)
		if err != nil {
			_, retValISqAsStr := retValISq.String()
			return nil, fmt.Errorf(
				"%s: could not compute sqrt(retVal[%d]^2=%s): %q",
				caller, i, retValISqAsStr, err.Error())
		}
		err = retVal.Set(0, i, retValI)
		if err != nil {
			return nil, fmt.Errorf("%s: could not set retVal[%d]: %q", caller, i, err.Error())
		}
	}
	return retVal, nil
}

func getColumnLengths(m *bigmatrix.BigMatrix, caller string) (*bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getColumnLengths", caller)
	retVal := bigmatrix.NewEmpty(1, m.NumCols())
	for j := 0; j < m.NumCols(); j++ {
		retValJSq := bignumber.NewFromInt64(0)
		var err error
		var retValI *bignumber.BigNumber
		for i := 0; i < m.NumRows(); i++ {
			var mIJ *bignumber.BigNumber
			mIJ, err = m.Get(i, j)
			if err != nil {
				return nil, fmt.Errorf("%s: could not get M[%d][%d]: %q", caller, i, j, err.Error())
			}
			retValJSq.MulAdd(mIJ, mIJ)
		}
		retValI, err = bignumber.NewFromInt64(0).Sqrt(retValJSq)
		if err != nil {
			_, retValJSqAsStr := retValJSq.String()
			return nil, fmt.Errorf(
				"%s: could not compute sqrt(retVal[%d]^2=%s): %q",
				caller, j, retValJSqAsStr, err.Error(),
			)
		}
		err = retVal.Set(0, j, retValI)
		if err != nil {
			return nil, fmt.Errorf("%s: could not set retVal[%d]: %q", caller, j, err.Error())
		}
	}
	return retVal, nil
}

// diagonalProduct assumes x is lower quadrangualar and its "determinant" (stretching this
// concept to non-square matrices) is therefore the product of its diagonal elements
func diagonalProduct(x *bigmatrix.BigMatrix, caller string) (*bignumber.BigNumber, error) {
	caller = fmt.Sprintf("%s-diagonalProduct", caller)
	det := bignumber.NewFromInt64(1)
	for i := 0; i < x.NumCols(); i++ {
		xII, err := x.Get(i, i)
		if err != nil {
			return nil, fmt.Errorf("%s: could not get X[%d][%d]: %q", caller, i, i, err.Error())
		}
		det.Mul(det, xII)
	}
	return det, nil
}

// getRandomXandH returns a pseudo-random x and H for which xH = 0 and H has
// - non-zero entries on the diagonal
// - zeroes above the diagonal
func getRandomXandH(
	numRows, numCols, numPossibleEntries int, tolerance *bignumber.BigNumber, caller string,
) (*bigmatrix.BigMatrix, *bigmatrix.BigMatrix, error) {
	// Update caller
	caller = fmt.Sprintf("%s-getRandomXandH", caller)

	// Create a random vector, xEntries, ending in a 1 so that H can be
	// generated with each column orthogonal to xEntries
	xEntries := make([]int64, numRows)
	for i := 0; i < numRows-1; i++ {
		xEntries[i] = int64(rand.Intn(numPossibleEntries) - (numPossibleEntries / 2))
	}
	xEntries[numRows-1] = 1

	// Create a matrix, hEntries, such that
	// - xH = 0
	// - H[i][i] != 0 for i = 0, ..., numCols-1
	hEntries := make([]int64, numRows*numCols)
	for j := 0; j < numCols; j++ {
		// First, create a vector orthogonal to x to put in the current row or column of hEntries
		dotProduct := int64(0)
		for i := j; i < numCols; i++ {
			hEntries[i*numCols+j] = int64(rand.Intn(numPossibleEntries) - (numPossibleEntries / 2))
			if (i == j) && (hEntries[i*numCols+j] == 0) {
				// An entry of zero on the diagonal of H might have made the columns of H dependent
				hEntries[i*numCols+j] = 1
			}
			dotProduct += xEntries[i] * hEntries[i*numCols+j]
		}
		hEntries[(numRows-1)*numCols+j] = -dotProduct
	}

	// Create H and x
	var x, h *bigmatrix.BigMatrix
	var err error
	h, err = bigmatrix.NewFromInt64Array(hEntries, numRows, numCols)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: error in NewFromInt64Array(hEntries=%v): %q",
			caller, hEntries, err.Error())
	}
	x, err = bigmatrix.NewFromInt64Array(xEntries, 1, numRows)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: error in NewFromInt64Array(xEntries=%v, 1, numRows=%d): %q",
			caller, xEntries, numRows, err.Error())
	}

	// Check orthogonality and return x, H
	xH := bigmatrix.NewEmpty(1, numRows)
	_, err = xH.Mul(x, h)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: could not compute\n%v\n\ntimes\n\n%v: %q", caller, x, h, err.Error(),
		)
	}
	var isOrthogonal bool
	isOrthogonal, err = bigmatrix.NewEmpty(1, numCols).Equals(xH, tolerance)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: error in zero.Equals(xH=%v): %q", caller, xH, err.Error(),
		)
	}
	if !isOrthogonal {
		return nil, nil, fmt.Errorf("%s: xH != 0 with\nx =\n%v\nH=%v\n", caller, x, h)
	}
	return x, h, nil
}

// getRandomMandX returns a pseudo-random M and x for which Mx = [0,0,...,x[numRows-1]]
// and M has
// - non-zero entries on the diagonal
// - zeroes above the diagonal
func getRandomMandX(
	numRows, numPossibleEntries int, tolerance *bignumber.BigNumber, caller string,
) (*bigmatrix.BigMatrix, *bigmatrix.BigMatrix, error) {
	// Update caller
	caller = fmt.Sprintf("%s-getRandomXandM", caller)

	// Create a random vector, xEntries, ending in a 1 so that M can be
	// generated with each row before the last row orthogonal to xEntries
	xEntries := make([]int64, numRows)
	xEntries[0] = 1
	for i := 1; i < numRows; i++ {
		xEntries[i] = int64(rand.Intn(numPossibleEntries) - (numPossibleEntries / 2))
	}

	// Create a matrix, mEntries, such that
	// - M[i][i] != 0 for i = 0,...,numRows-1
	// - M[0][0] = 1 (just to make things simple; any non-zero entry would do)
	// - Mx[i] = 0 for i = 1,...,numRows-1
	mEntries := make([]int64, numRows*numRows)
	mEntries[0] = 1
	for i := 1; i < numRows; i++ {
		// Generate entries 1,2,...,i of the current row, and use their dot product with x
		// to set entry 0. Entry 0 is multiplied by x[0] = 1 in the computation, Mx, so it
		// should be set to -<(x[1],x[2],...,x[i]),(M[i][1],M[i][2],...,M[i][i])>
		dotProduct := int64(0)
		for j := 1; j <= i; j++ {
			mEntries[i*numRows+j] = int64(
				rand.Intn(numPossibleEntries) - (numPossibleEntries / 2),
			)
			dotProduct += xEntries[j] * mEntries[i*numRows+j]
		}
		mEntries[i*numRows] = -dotProduct
	}

	// Create x and M
	var x, m *bigmatrix.BigMatrix
	var err error
	m, err = bigmatrix.NewFromInt64Array(mEntries, numRows, numRows)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: error in NewFromInt64Array(mEntries=%v): %q",
			caller, mEntries, err.Error())
	}
	x, err = bigmatrix.NewFromInt64Array(xEntries, numRows, 1)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: error in NewFromInt64Array(xEntries=%v, 1, numRows=%d): %q",
			caller, xEntries, numRows, err.Error())
	}

	// Check orthogonality of all but the last row of M, and return M, x
	var isOrthogonal bool
	mx := bigmatrix.NewEmpty(1, numRows)
	_, err = mx.Mul(m, x)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: could not compute\n%v\n\ntimes\n\n%v: %q", caller, m, x, err.Error(),
		)
	}
	expectedMX := bigmatrix.NewEmpty(numRows, 1)
	err = expectedMX.Set(0, 0, bignumber.NewFromInt64(xEntries[0]))
	isOrthogonal, err = expectedMX.Equals(mx, tolerance)
	if err != nil {
		return nil, nil, fmt.Errorf(
			"%s: error in expectedMX.Equals(mx=%v): %q", caller, mx, err.Error(),
		)
	}
	if !isOrthogonal {
		return nil, nil, fmt.Errorf("%s: mx =\n%v\n is not [%d,0,0,...,0]", caller, mx, xEntries[0])
	}
	return m, x, nil
}

func hasZeroesAboveTheDiagonal(
	x *bigmatrix.BigMatrix, tolerance *bignumber.BigNumber, caller string,
) (bool, error) {
	caller = fmt.Sprintf("%s-hasZeroesAboveTheDiagonal", caller)
	for i := 0; i < x.NumRows(); i++ {
		for j := i + 1; j < x.NumCols(); j++ {
			xIJ, err := x.Get(i, j)
			if err != nil {
				return false, fmt.Errorf(
					"%s: could not get X[%d][%d]: %q", caller, i, j, err.Error(),
				)
			}
			equals := bignumber.NewFromInt64(0).Equals(xIJ, tolerance)
			if !equals {
				return false, nil
			}
		}
	}
	return true, nil
}
