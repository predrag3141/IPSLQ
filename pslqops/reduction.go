package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
)

const (
	makeUSmallerThanT = iota
	makeTSmallerThanU
)

// getD returns a row operation matrix, D, that reduces a lower quadrangular matrix
// like the matrix H in the original PSLQ paper. It returns
//
// - Whether the reduction matrix is the identity
//
//   - Whether GetD calculated an all-zero row (left of the diagonal element),
//     in spite of detecting the need to change the row. This is wasteful but not an error,
//     because it can happen due to round-off.
//
// - Any error encountered.
//
// Entries below the diagonal of DH are bounded by the absolute value of the diagonal
// element above them, multiplied by .5+[gentle reduction mode].
func getD(h *bigmatrix.BigMatrix, d *bigmatrix.BigMatrix, caller string) (bool, bool, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getD", caller)
	numRows, numCols := h.Dimensions()
	one := bignumber.NewFromInt64(1)
	zero := bignumber.NewFromInt64(0)
	half := bignumber.NewPowerOfTwo(-1)
	isIdentity := true
	calculatedAllZeroRow := false
	columnThresh, err := getColumnThresholds(h, caller)

	// Get 1/H[i][i] for all i
	reciprocalDiagonal := make([]*bignumber.BigNumber, numCols)
	for j := 0; j < numCols; j++ {
		var hJJ *bignumber.BigNumber
		hJJ, err = h.Get(j, j)
		if err != nil {
			return true, false,
				fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j, j, err.Error())
		}
		reciprocalDiagonal[j], err = bignumber.NewFromInt64(0).Quo(one, hJJ)
	}

	// Compute D
	for i := 0; i < numRows; i++ {
		// Set the diagonal element to 1.
		err = d.Set(i, i, one)
		if err != nil {
			return isIdentity, calculatedAllZeroRow,
				fmt.Errorf("%s: could not set D[%d][%d]: %q", caller, i, i, err.Error())
		}

		// Skip rows that do not need reduction
		reduceRow := false
		if i > 0 {
			reduceRow, _, err = rowNeedsReduction(h, columnThresh, i, 0, caller)
			if err != nil {
				return isIdentity, calculatedAllZeroRow, fmt.Errorf(
					"%s: could not determine whether row %d needs reduction: %q",
					caller, i, err.Error(),
				)
			}
		}
		if !reduceRow {
			// Row i of D is all zero, except for its diagonal element. Elements of row i to
			// the right of the diagonal are already set to zero.
			for j := 0; j < i; j++ {
				err = d.Set(i, j, zero)
				if err != nil {
					return isIdentity, calculatedAllZeroRow,
						fmt.Errorf("%s: could not set D[%d][%d]: %q", caller, i, j, err.Error())
				}
			}
			continue
		}

		// Row i needs reduction
		rowIsAllZero := true
		for j := i - 1; 0 <= j; j-- {
			// Initially, compute the entry in D, except not rounded to the nearest integer
			// The formula for this is equation 6 in the 1992 PSLQ paper. The range for k
			// is from j+1 to i, including both j+1 and i. Note that the counterpart of this
			// formula for column reduction uses a range k = j, j+1, ..., i-1 instead of
			// j+1, j+2, ..., i.
			dEntry := bignumber.NewFromInt64(0)
			for k := j + 1; k <= i; k++ {
				var dIK, hKJ *bignumber.BigNumber
				dIK, err = d.Get(i, k)
				if err != nil {
					return isIdentity, calculatedAllZeroRow,
						fmt.Errorf("%s: could not get D[%d][%d]: %q", caller, i, k, err.Error())
				}
				hKJ, err = h.Get(k, j)
				if err != nil {
					return isIdentity, calculatedAllZeroRow,
						fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, k, j, err.Error())
				}
				dIKhKJ := bignumber.NewFromInt64(0).Mul(dIK, hKJ)
				dEntry.Sub(dEntry, dIKhKJ)
			}
			dEntry.Mul(dEntry, reciprocalDiagonal[j])

			// Next, round the entry to the nearest integer.
			if dEntry.Cmp(zero) > 0 {
				dEntry.Add(dEntry, half)
			} else {
				dEntry.Sub(dEntry, half)
			}
			dEntry.RoundTowardsZero()
			if !dEntry.IsZero() {
				isIdentity = false
				rowIsAllZero = false
			}

			// Set D[i][j]
			err = d.Set(i, j, dEntry)
			if err != nil {
				return isIdentity, calculatedAllZeroRow,
					fmt.Errorf("%s: could not set D[%d][%d]: %q", caller, i, j, err.Error())
			}
		}
		if rowIsAllZero {
			// Since this row of H needed reduction, it should be reported that this
			// row of D has only zeroes (other than the diagonal element, of course).
			// This is not an error, since it can happen due to round-off.
			calculatedAllZeroRow = true
		}
	}
	return isIdentity, calculatedAllZeroRow, nil
}

// getE returns a row operation matrix, E, that reduces a lower quadrangular
// matrix like the matrix H in the original PSLQ paper. It returns
//
// - Whether the reduction matrix is the identity
//
//   - Whether getE calculated an all-zero column (below the diagonal element),
//     in spite of detecting the need to change the row. This is wasteful but not an error,
//     because it can happen due to round-off.
//
// - Any error encountered.
//
// Entries to the left of the diagonal of ME are bounded by the absolute value of the diagonal
// element to their right, multiplied by .5.
func getE(m *bigmatrix.BigMatrix, e *bigmatrix.BigMatrix, caller string) (bool, bool, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getE", caller)
	mNumRows := m.NumRows()
	eNumRows := e.NumRows()
	one := bignumber.NewFromInt64(1)
	zero := bignumber.NewFromInt64(0)
	half := bignumber.NewPowerOfTwo(-1)
	isIdentity := true
	calculatedAllZeroColumn := false
	rowThresh, err := getRowThresholds(m, caller)

	// M and E are square matrices, with one more row and column in E than in M.
	// The bottom row and right-most column of E have zeroes off the diagonal and one
	// on the diagonal.
	err = e.Set(eNumRows-1, eNumRows-1, one)
	if err != nil {
		return true, false,
			fmt.Errorf("%s: could not set E[%d][%d]: %q", caller, eNumRows-1, eNumRows-1, err.Error())
	}

	// Get 1/M[i][i] for all i
	reciprocalDiagonal := make([]*bignumber.BigNumber, mNumRows)
	for i := 0; i < mNumRows; i++ {
		var mII *bignumber.BigNumber
		mII, err = m.Get(i, i)
		if err != nil {
			return true, false,
				fmt.Errorf("%s: could not get M[%d][%d]: %q", caller, i, i, err.Error())
		}
		reciprocalDiagonal[i], err = bignumber.NewFromInt64(0).Quo(one, mII)
	}

	// Compute the E[i][j] for i, j < mNumRows (not the bottom row and right-most column of E)
	for j := mNumRows - 1; 0 <= j; j-- {
		// Set the diagonal element to 1. Also, zero out the entries above the diagonal, because
		// there is no guarantee that E comes into this function as a lower-triangular matrix.
		err = e.Set(j, j, one)
		if err != nil {
			return isIdentity, calculatedAllZeroColumn,
				fmt.Errorf("%s: could not set E[%d][%d]: %q", caller, j, j, err.Error())
		}
		for i := 0; i < j; i++ {
			err = e.Set(i, j, zero)
			if err != nil {
				return isIdentity, calculatedAllZeroColumn,
					fmt.Errorf("%s: could not set E[%d][%d]: %q", caller, i, j, err.Error())
			}
		}

		// Skip columns that do not need reduction.
		reduceColumn := false
		if j < (mNumRows - 1) {
			reduceColumn, _, err = columnNeedsReduction(m, rowThresh, j, 0, caller)
			if err != nil {
				return isIdentity, calculatedAllZeroColumn, fmt.Errorf(
					"%s: could not determine whether column %d needs reduction: %q",
					caller, j, err.Error(),
				)
			}
		}
		if !reduceColumn {
			// Column j of E is all zero, except for its diagonal element. Elements of column
			// i above the diagonal are already set to zero.
			for i := j + 1; i < mNumRows; i++ {
				err = e.Set(i, j, zero)
				if err != nil {
					return isIdentity, calculatedAllZeroColumn,
						fmt.Errorf("%s: could not set E[%d][%d]: %q", caller, i, j, err.Error())
				}
			}
			continue
		}

		// Column j needs reduction
		columnIsAllZero := true
		for i := j + 1; i < mNumRows; i++ {
			// Initially, compute the entry in E, except not rounded to the nearest integer.
			//
			// See comments in the equivalent section of getD comparing the formula
			// used here to the formula used to define D.
			eEntry := bignumber.NewFromInt64(0)
			for k := j; k < i; k++ {
				var mIK, eKJ *bignumber.BigNumber
				mIK, err = m.Get(i, k)
				if err != nil {
					return isIdentity, calculatedAllZeroColumn,
						fmt.Errorf("%s: could not get M[%d][%d]: %q", caller, i, k, err.Error())
				}
				eKJ, err = e.Get(k, j)
				if err != nil {
					return isIdentity, calculatedAllZeroColumn,
						fmt.Errorf("%s: could not get E[%d][%d]: %q", caller, k, j, err.Error())
				}
				mIKeKJ := bignumber.NewFromInt64(0).Mul(mIK, eKJ)
				eEntry.Sub(eEntry, mIKeKJ)
			}
			eEntry.Mul(eEntry, reciprocalDiagonal[i])

			// Next, round the entry to the nearest integer.
			if eEntry.Cmp(zero) > 0 {
				eEntry.Add(eEntry, half)
			} else {
				eEntry.Sub(eEntry, half)
			}
			eEntry.RoundTowardsZero()
			if !eEntry.IsZero() {
				isIdentity = false
				columnIsAllZero = false
			}

			// Set E[i][j]
			err = e.Set(i, j, eEntry)
			if err != nil {
				return isIdentity, calculatedAllZeroColumn,
					fmt.Errorf("%s: could not set E[%d][%d]: %q", caller, i, j, err.Error())
			}
		}
		if columnIsAllZero {
			// Since this column of M needed reduction, it should be reported that this
			// column has only zeroes (other than the diagonal element, of course).
			// This is not an error, since it can happen due to round-off.
			calculatedAllZeroColumn = true
		}
	}
	return isIdentity, calculatedAllZeroColumn, nil
}

// getColumnThresholds returns the maximum absolute value in each column of H
// after reduction, and any error.
func getColumnThresholds(
	h *bigmatrix.BigMatrix, caller string,
) ([]*bignumber.BigNumber, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getColumnThresholds", caller)
	numCols := h.NumCols()
	columnThresh := make([]*bignumber.BigNumber, numCols)
	half := bignumber.NewPowerOfTwo(-1)

	// Set thresholds
	for j := 0; j < numCols; j++ {
		hJJ, err := h.Get(j, j)
		if err != nil {
			return nil, fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j, j, err.Error())
		}
		absHJJ := bignumber.NewFromInt64(0).Abs(hJJ)
		columnThresh[j] = bignumber.NewFromInt64(0).Mul(half, absHJJ)
	}
	return columnThresh, nil
}

// getRowThresholds returns the maximum absolute value in each column of H
// after reduction, and any error.
func getRowThresholds(
	m *bigmatrix.BigMatrix, caller string,
) ([]*bignumber.BigNumber, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getRowThresholds", caller)
	numRows := m.NumRows()
	rowThresh := make([]*bignumber.BigNumber, numRows)
	half := bignumber.NewPowerOfTwo(-1)

	// Set thresholds
	for i := 0; i < numRows; i++ {
		mII, err := m.Get(i, i)
		if err != nil {
			return nil, fmt.Errorf("%s: could not get M[%d][%d]: %q", caller, i, i, err.Error())
		}
		absMII := bignumber.NewFromInt64(0).Abs(mII)
		rowThresh[i] = bignumber.NewFromInt64(0).Mul(half, absMII)
	}
	return rowThresh, nil
}

// rowNeedsReduction returns whether rowNumber of H needs reduction, what column exceeds
// the threshold, and any error.  The threshold is increased by 2^log2tolerance
// if log2tolerance is negative, else by zero.
func rowNeedsReduction(
	h *bigmatrix.BigMatrix, columnThresh []*bignumber.BigNumber,
	rowNumber, log2tolerance int, caller string,
) (bool, int, error) {
	caller = fmt.Sprintf("%s-rowNeedsReduction", caller)
	var tolerance *bignumber.BigNumber
	if log2tolerance < 0 {
		tolerance = bignumber.NewPowerOfTwo(int64(log2tolerance))
	}

	// Check whether all elements in the row are bounded by columnThresh.
	for j := 0; j < rowNumber; j++ {
		hIJ, err := h.Get(rowNumber, j)
		if err != nil {
			return false, -1, fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, rowNumber, j, err.Error())
		}
		absHIJ := bignumber.NewFromInt64(0).Abs(hIJ)
		if tolerance != nil {
			// Decrease absHIJ before comparing it to the threshold. This avoids false alarms.
			absHIJ = bignumber.NewFromInt64(0).Sub(absHIJ, tolerance)
		}
		if absHIJ.Cmp(columnThresh[j]) > 0 {
			return true, j, nil
		}
	}

	// No elements in row rowNumber need reducing
	return false, -1, nil
}

// columnNeedsReduction returns whether column columnNumber of M needs reduction, what row
// exceeds the threshold, and any error. The threshold is increased by 2^log2tolerance
// if log2tolerance is negative, else by zero.
func columnNeedsReduction(
	m *bigmatrix.BigMatrix, rowThresh []*bignumber.BigNumber,
	columnNumber, log2tolerance int, caller string,
) (bool, int, error) {
	caller = fmt.Sprintf("%s-columnNeedsReduction", caller)
	var tolerance *bignumber.BigNumber
	if log2tolerance < 0 {
		tolerance = bignumber.NewPowerOfTwo(int64(log2tolerance))
	}

	// Check whether all elements in the row are bounded by rowThresh.
	for i := columnNumber + 1; i < m.NumRows(); i++ {
		mIJ, err := m.Get(i, columnNumber)
		if err != nil {
			return false, -1, fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, i, columnNumber, err.Error())
		}
		absMIJ := bignumber.NewFromInt64(0).Abs(mIJ)
		if tolerance != nil {
			// Decrease absHIJ before comparing it to the threshold. This avoids false alarms.
			absMIJ = bignumber.NewFromInt64(0).Sub(absMIJ, tolerance)
		}
		if absMIJ.Cmp(rowThresh[i]) > 0 {
			return true, i, nil
		}
	}

	// No elements in row rowNumber need reducing
	return false, -1, nil
}

// isRowReduced is a convenience function that wraps a series of tests of the rows of H that use
// rowNeedsReduction. Returned values:
//
// - whether H is row reduced
//
// - a row and column that exceeds the threshold for H being reduced, or (-1, -1) if H is reduced
//
// - any error encountered
func isRowReduced(
	h *bigmatrix.BigMatrix, log2tolerance int, caller string,
) (bool, int, int, error) {
	caller = fmt.Sprintf("%s-isRowReduced", caller)
	numRows := h.NumRows()
	columnThresh, err := getColumnThresholds(h, caller)
	if err != nil {
		return true, -1, -1, err
	}
	for i := 1; i < numRows; i++ {
		var needsReduction bool
		var column int
		needsReduction, column, err = rowNeedsReduction(
			h, columnThresh, i, log2tolerance, caller,
		)
		if err != nil {
			return true, -1, -1, err
		}
		if needsReduction {
			return false, i, column, nil
		}
	}
	return true, -1, -1, nil
}

// isColumnReduced is a convenience function that wraps a series of tests of the columns of M
// that use columnNeedsReduction. Returned values:
//
// - whether M is column reduced
//
// - a row and column that exceeds the threshold for M being reduced, or (-1, -1) if M is reduced
//
// - any error encountered
func isColumnReduced(
	m *bigmatrix.BigMatrix, log2tolerance int, caller string,
) (bool, int, int, error) {
	caller = fmt.Sprintf("%s-isColumnReduced", caller)
	numCols := m.NumCols()
	rowThresh, err := getRowThresholds(m, caller)
	if err != nil {
		return true, -1, -1, err
	}
	for j := 0; j < numCols-1; j++ {
		var needsReduction bool
		var row int
		needsReduction, row, err = columnNeedsReduction(
			m, rowThresh, j, log2tolerance, caller,
		)
		if err != nil {
			return true, -1, -1, err
		}
		if needsReduction {
			return false, row, j, nil
		}
	}
	return true, -1, -1, nil
}

// reducePair calls a provided function, reportRowOp, with a parameter that reportRowOp
// should interpret as a 2x2 matrix R that reduces t and u to t' and u', i.e.
//
// R [t u]-transpose = [t' u']-transpose
//
// Reduction ends when either it cannot continue, or reportRowOp returns true. Every time
// reportRowOp is called, reducePair has set t to t' and u to u'. Termination conditions are:
//
//   - t' or u' is essentially 0. If this happens when t' is essentially zero, t' and u' are
//     swapped and R is updated to reflect that. This is because t is considered to be a diagonal
//     element of H, which can never be zero while the algorithm is running.
//
//   - An entry in R would exceed maxMatrixEntry in absolute value if reduction were to continue.
//
//   - As noted above, when reportRowOp returns true.
func reducePair(
	t, u *bignumber.BigNumber, maxMatrixEntry int, caller string, reportRowOp func([]int) bool,
) error {
	// Initializations
	caller = fmt.Sprintf("%s-reducePair", caller)
	oneHalf := bignumber.NewPowerOfTwo(-1)
	rowOpMatrix := []int{1, 0, 0, 1}
	coefficientAsBigNumber := bignumber.NewFromInt64(0)
	loopIterations := 0

	for (!t.IsSmall()) && (!u.IsSmall()) {
		// At the start of each loop, force |u| <= |t|
		reorderRows(t, u, rowOpMatrix, makeUSmallerThanT) // since this loop assumes |u| <= |t|

		// Since |u| <= |t|, R' forces |t| down with rows [-nint(t/u), 1] [1,0], where
		// "nint" means nearest int
		_, err := coefficientAsBigNumber.Quo(t, u) // to be rounded down and negated
		if err != nil {
			_, tStr := t.String()
			_, uStr := u.String()
			reorderRows(t, u, rowOpMatrix, makeUSmallerThanT) // otherwise t could be set to essentially-zero
			reportRowOp(rowOpMatrix)
			return fmt.Errorf(
				"%s: could not compute %q/%q: %q", caller, tStr, uStr, err.Error(),
			)
		}

		// The result of adding or subtracting 1/2, depending on the sign of coefficientAsBigNumber,
		// then rounding towards zero, is nint(coefficientAsBigNumber)
		if coefficientAsBigNumber.IsNegative() {
			coefficientAsBigNumber.Sub(coefficientAsBigNumber, oneHalf)
		} else {
			coefficientAsBigNumber.Add(coefficientAsBigNumber, oneHalf)
		}
		coefficientAsInt64Ptr := coefficientAsBigNumber.Int64RoundTowardsZero()

		// Handle edge cases where coefficientAsInt64Ptr is nil, is de-referenced to zero,
		// or is de-referenced into such a large integer that it cannot be incorporated
		// into rowOpMatrix with entries below maxMatrixEntry.
		//
		// The second of these edge cases, where *coefficientAsInt64Ptr == 0, should never
		// happen. The other two should be extremely rare.
		if coefficientAsInt64Ptr == nil {
			// Edge case: u ~ 0 and coefficientAsBigNumber = t/u. This is a sign of a very
			// successful reduction, I guess. It is not possible to update rowOpMatrix, so return.
			// Rows should not be re-ordered since u is essentially zero and t should be non-zero.
			reportRowOp(rowOpMatrix)
			return nil // rowOpMatrix entries are guaranteed <= maxMatrixEntry
		}
		coefficientAsInt64 := *coefficientAsInt64Ptr
		if coefficientAsInt64 == 0 {
			// Since |u| <= |t|, |coefficientAsInt64| = |nint(t/u)| > 0. There is no way
			// coefficientAsInt64 can be zero, other than a catastrophic error.
			// In view of the catastrophic error, rows are not re-ordered (why bother?).
			_, tAsStr := t.String()
			_, uAsStr := u.String()
			reportRowOp(rowOpMatrix)
			return fmt.Errorf("%s: computed nearest integer to  %q/%q as zero", caller, tAsStr, uAsStr)
		}
		if (coefficientAsInt64 > int64(maxMatrixEntry)) || (-coefficientAsInt64 > int64(maxMatrixEntry)) {
			// The coefficient is so large that it cannot be converted to an int that is within the
			// allowed range. Such a large coefficient cannot be incorporated into rowOpMatrix, except.
			// on the first time through the main loop of this function (hence the "if" test below). Since
			// u and t should not have been essentially zero entering this loop, the smaller of the two
			// belongs on the diagonal (where t is) when re-ordering rows.
			if loopIterations > 0 {
				reorderRows(t, u, rowOpMatrix, makeTSmallerThanU) // Since neither t nor u is essentially zero
				reportRowOp(rowOpMatrix)
				return nil // rowOpMatrix entries are guaranteed <= maxMatrixEntry
			}
		}

		// The edge cases have been dispensed with. R' has rows [1, -nint(t/u)] and [0,1].
		// nint(t/u) = coefficientAsInt64
		a := rowOpMatrix[0] - int(coefficientAsInt64)*rowOpMatrix[2]
		b := rowOpMatrix[1] - int(coefficientAsInt64)*rowOpMatrix[3]
		if (a > maxMatrixEntry) || (a < -maxMatrixEntry) || (b > maxMatrixEntry) || (b < -maxMatrixEntry) {
			// Since u and t should not have been essentially zero entering this loop, the
			// smaller of the two belongs on the diagonal (where t is) when re-ordering rows.
			if loopIterations > 0 {
				reorderRows(t, u, rowOpMatrix, makeTSmallerThanU) // Since neither t nor u is essentially zero
				reportRowOp(rowOpMatrix)
				return nil // rowOpMatrix entries are guaranteed <= maxMatrixEntry
			}
		}
		rowOpMatrix[0] = a // rowOpMatrix entries are guaranteed <= maxMatrixEntry
		rowOpMatrix[1] = b // rowOpMatrix entries are guaranteed <= maxMatrixEntry
		addThisToT := bignumber.NewFromInt64(0).Int64Mul(-coefficientAsInt64, u)
		t.Set(bignumber.NewFromInt64(0).Add(t, addThisToT))

		// Check termination conditions. There is no need to check whether u is
		// essentially zero, because the loop condition did that.
		if t.IsSmall() {
			reorderRows(t, u, rowOpMatrix, makeUSmallerThanT) // makes u essentially 0, instead of t
			reportRowOp(rowOpMatrix)
			return nil
		}
		if reportRowOp(rowOpMatrix) {
			// reorderRows to make |t| <= |u| would be a no-op, since |t| has just been reduced.
			return nil // rowOpMatrix entries are guaranteed <= maxMatrixEntry
		}
		loopIterations++
	}

	// The last iteration of the main loop, if any, made |t| smaller than |u|. The only time
	// this is undesirable is when t is essentially zero, since t is typically a diagonal element.
	if t.IsSmall() && (!u.IsSmall()) {
		reorderRows(t, u, rowOpMatrix, makeUSmallerThanT) // makes u essentially 0, instead of t
	}
	reportRowOp(rowOpMatrix)
	return nil // rowOpMatrix entries are guaranteed <= maxMatrixEntry
}

// reorderRows makes |u| <= |t| or |t| <= |u| according to the parameter, how, by swapping
// rows in r and swapping t and u, if required.
func reorderRows(t, u *bignumber.BigNumber, r []int, how int) {
	// Handle t and u being essentially zero by returning early or ignoring the parameter, how.
	if u.IsSmall() {
		// Row order should not be changed, since this could put essentially-zero on the diagonal
		// (which is where t normally comes from).
		return
	}
	if !t.IsSmall() {
		// Since t is not essentially zero, it is OK to follow the wishes of the calling function
		absT := bignumber.NewFromInt64(0).Abs(t)
		absU := bignumber.NewFromInt64(0).Abs(u)
		tCmpU := absT.Cmp(absU)
		if (how == makeUSmallerThanT) && tCmpU > -1 {
			// No action is required
			return
		}
		if (how == makeTSmallerThanU) && tCmpU == -1 {
			// No action is required
			return
		}
	}

	// All possible reasons not to swap rows have been checked
	a := r[0]
	b := r[1]
	r[0] = r[2]
	r[1] = r[3]
	r[2] = a
	r[3] = b
	oldT := bignumber.NewFromBigNumber(t)
	t.Set(u)
	u.Set(oldT)
}
