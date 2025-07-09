package pslqops

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
	"sort"
	"strings"
)

const (
	log2boundednessTolerance = -20
)

// getE populates a row operation matrix, E, that reduces a lower triangular matrix like
// the matrix M, the inverse of the matrix H from the original PSLQ paper. It returns
//
//   - the number of columns of ME that were reduced compared to M, i.e. the corresponding
//     column of E is not an identity column
//
//   - the number of columns of M that were reduced compared to M *and* each element in
//     the corresponding column of ME is bounded in absolute value by half the diagonal
//     element to its right in M and in ME (which are the same, since M and ME have the
//     same diagonal).
//
// - Any error encountered.
func getE(
	m *bigmatrix.BigMatrix, e *bigmatrix.BigMatrix, maxTreeDepth int, caller string,
) (int, int, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getE", caller)
	boundednessTolerance := bignumber.NewPowerOfTwo(log2boundednessTolerance)
	mNumRows := m.NumRows()
	eNumRows := e.NumRows()
	one := bignumber.NewFromInt64(1)
	columnsReduced, columnsBounded := 0, 0
	var err error

	// Get 1/M[i][i] for all i
	reciprocalDiagonal, err := getReciprocalDiagonal(m, caller)
	if err != nil {
		return columnsReduced, columnsBounded, fmt.Errorf(
			"%s: could not get the negative reciprocal of the diagonal of M: %q",
			caller, err.Error(),
		)
	}
	zero := bignumber.NewFromInt64(0)
	for i := 0; i < mNumRows; i++ {
		var mII, oneOverMII *bignumber.BigNumber
		mII, err = m.Get(i, i)
		if err != nil {
			return columnsReduced, columnsBounded,
				fmt.Errorf(
					"%s: could not get M[%d][%d]: %q", caller, i, i, err.Error(),
				)
		}
		oneOverMII, err = bignumber.NewFromInt64(0).Quo(one, mII)
		reciprocalDiagonal[i] = bignumber.NewFromInt64(0).Sub(zero, oneOverMII)
	}

	// Set E, column by column. This can be done in parallel, one column per thread,
	// though it is currently sequential.
	for j := eNumRows - 1; 0 <= j; j-- {
		var bestColumn *EColumn // This nil value is updated except for the two rightmost columns
		if j < mNumRows-1 {
			var eColumns *EColumnArray
			eColumns, err = newEColumnArray(m, reciprocalDiagonal, j, maxTreeDepth, caller)
			bestColumn, err = eColumns.bestColumn(caller)
			if err != nil {
				return columnsReduced, columnsBounded, fmt.Errorf(
					"%s: could not get best column: %q", caller, err.Error(),
				)
			}
		}
		if bestColumn == nil {
			// Either this is the last column, or no column improving on the identity column
			// was found for column j. Set column j to the identity column.
			for i := 0; i < eNumRows; i++ {
				if i == j {
					err = e.Set(i, j, one)
				} else {
					err = e.Set(i, j, zero)
				}
				if err != nil {
					return columnsReduced, columnsBounded, fmt.Errorf(
						"%s: could not set E[%d][%d]: %q", caller, i, j, err.Error(),
					)
				}
			}
			continue
		}

		// Collect statistics
		if !bestColumn.equalsIdentityColumn {
			var boundedByDiagonal bool
			columnsReduced++
			boundedByDiagonal, err = bestColumn.isBoundedByDiagonal(boundednessTolerance, "getE")
			if boundedByDiagonal {
				columnsBounded++
			}
		}

		// Set column j of E to bestColumn
		for i := 0; i < eNumRows; i++ {
			err = e.Set(i, j, bestColumn.columnOfE[i])
			if err != nil {
				return columnsReduced, columnsBounded, err
			}
		}
	}
	return columnsReduced, columnsBounded, nil
}

func getReciprocalDiagonal(m *bigmatrix.BigMatrix, caller string) ([]*bignumber.BigNumber, error) {
	caller = fmt.Sprintf("%s-getReciprocalDiagonal", caller)
	zero := bignumber.NewFromInt64(0)
	one := bignumber.NewFromInt64(1)
	mNumRows := m.NumRows()
	reciprocalDiagonal := make([]*bignumber.BigNumber, mNumRows)
	for i := 0; i < mNumRows; i++ {
		var oneOverMII *bignumber.BigNumber
		mII, err := m.Get(i, i)
		if err != nil {
			return nil, fmt.Errorf("%s: could not get M[%d][%d]: %q", caller, i, i, err.Error())
		}
		oneOverMII, err = bignumber.NewFromInt64(0).Quo(one, mII)
		reciprocalDiagonal[i] = bignumber.NewFromInt64(0).Sub(zero, oneOverMII)
	}
	return reciprocalDiagonal, nil
}

type EColumn struct {
	m                    *bigmatrix.BigMatrix   // matrix M being reduced
	reciprocalDiagonal   []*bignumber.BigNumber // copy of -1/m[i][i] for i = 0,...,mNumRows-1
	mNumRows             int                    // convenience variable with number of rows in M
	rowNbr               int                    // number of row just updated or being updated
	colNbr               int                    // number of column of E being computed
	columnOfE            []*bignumber.BigNumber // contains entries of E for rows colNbr,...,rowNbr-1
	columnOfME           []*bignumber.BigNumber // ME[i][ec.colNbr] for i = 0,...,mNumRows-1
	squaredNorm          float64                // Squared norm of column colNbr of ME so far
	equalsIdentityColumn bool
}

// newEColumn
func newEColumn(
	m *bigmatrix.BigMatrix, reciprocalDiagonal []*bignumber.BigNumber, colNbr int, caller string,
) (*EColumn, error) {
	caller = fmt.Sprintf("%s-newEColumn", caller)
	mNumRows := m.NumRows()
	eNumRows := mNumRows + 1
	diagonalEntryAsBigNumber, err := m.Get(colNbr, colNbr)
	if err != nil {
		return nil, fmt.Errorf(
			"%s: could not get M[%d][%d]: %q", caller, colNbr, colNbr, err.Error(),
		)
	}
	diagonalEntryAsFloat64, _ := diagonalEntryAsBigNumber.AsFloat().Float64()
	retVal := &EColumn{
		m:                    m,                  // shallow copy
		reciprocalDiagonal:   reciprocalDiagonal, // shallow copy
		mNumRows:             mNumRows,
		rowNbr:               colNbr,
		colNbr:               colNbr,
		columnOfE:            make([]*bignumber.BigNumber, eNumRows), // computed for row colNbr only
		columnOfME:           make([]*bignumber.BigNumber, eNumRows), // computed for row colNbr only
		squaredNorm:          diagonalEntryAsFloat64 * diagonalEntryAsFloat64,
		equalsIdentityColumn: true, // becomes false if a non-identity entry of columnOfE is added
	}

	// Initialize
	// - columnOfE to all zero except 1 on the diagonal. Though the entries above
	//   the diagonal are unused, including them makes indexing simpler.
	// - columnOfME to M[i][colNbr] E[colNbr][colNbr], which is just M[i][colNbr]
	//   since diagonal elements of E are 1.
	for i := 0; i < colNbr; i++ {
		retVal.columnOfE[i] = bignumber.NewFromInt64(0)
		retVal.columnOfME[i] = bignumber.NewFromInt64(0)
	}
	for i := colNbr; i < mNumRows; i++ {
		var mIJ *bignumber.BigNumber
		mIJ, err = m.Get(i, colNbr)
		if err != nil {
			return nil, fmt.Errorf(
				"%s: could not get M[%d][%d]: %q", caller, i, colNbr, err.Error(),
			)
		}
		if i == colNbr {
			retVal.columnOfE[i] = bignumber.NewFromInt64(1)
		} else {
			retVal.columnOfE[i] = bignumber.NewFromInt64(0)
		}
		retVal.columnOfME[i] = bignumber.NewFromBigNumber(mIJ) // Since E[colNbr][colNbr] = 1
	}
	retVal.columnOfE[eNumRows-1] = bignumber.NewFromInt64(0)
	retVal.columnOfME[eNumRows-1] = bignumber.NewFromInt64(0)

	// Return the new EColumn
	return retVal, nil
}

func (ec *EColumn) clone() *EColumn {
	columnLen := len(ec.columnOfE)
	columnOfE := make([]*bignumber.BigNumber, columnLen)
	columnOfME := make([]*bignumber.BigNumber, columnLen)
	for i := 0; i < columnLen; i++ {
		columnOfE[i] = bignumber.NewFromBigNumber(ec.columnOfE[i])
		columnOfME[i] = bignumber.NewFromBigNumber(ec.columnOfME[i])
	}
	return &EColumn{
		m:                    ec.m,                  // shallow copy
		reciprocalDiagonal:   ec.reciprocalDiagonal, // shallow copy
		mNumRows:             ec.mNumRows,
		rowNbr:               ec.rowNbr,
		colNbr:               ec.colNbr,
		columnOfE:            columnOfE,
		columnOfME:           columnOfME,
		squaredNorm:          ec.squaredNorm,
		equalsIdentityColumn: ec.equalsIdentityColumn, // becomes false if a non-identity entry of columnOfE is added
	}
}

func (ec *EColumn) update(caller string) (*EColumn, error) {
	caller = fmt.Sprintf("%s-update", caller)
	ec.rowNbr++
	if ec.rowNbr == ec.mNumRows {
		return nil, nil
	}

	// To update, clone ec and
	// - Compute ec.columnOfE[ec.rowNbr] and clone.columnOfE[ec.rowNbr]
	// - Compute the final value of ec.columnOfME[ec.rowNbr] and clone.columnOfME[ec.rowNbr]
	// - Update the provisional values of ec.columnOfME[i] and clone.columnOfME for
	//   i in { ec.rowNbr+1, ..., ec.mNumRows-1 }
	// - Update ec.squaredNorm and clone.squaredNorm with the sum of squares of columnOfME[i]
	//   for i in { ec.colNbr, ... ec.rowNbr }
	//
	// Start with creating the clone
	clone := ec.clone()

	// Compute columnOfE[ec.rowNbr]
	//
	// For all i in { ec.colNbr+1, ..., mNumRows-1 }, the current state of ec.columnOfME[i] is
	//
	//                        ec.rowNbr-1
	//     ec.columnOfME[i] =     sum      M[i][k] E[k][ec.colNbr]
	//                        k=ec.colNbr
	//
	// The approximate formula for E[i][ec.colNbr], before rounding to one of the nearest
	// integers, called "nextEntryOfEAsFloat" in the code below, is
	//
	//                        i-1
	//                        sum     M[i][k] E[k][ec.colNbr]
	//                     k=ec.colNbr
	// E[i][ec.colNbr] ~ - __________________________________
	//                                 M[i][i]
	//
	// The formula for E[ec.rowNbr][ec.colNbr] can be simplified by substituting in
	// ec.columnOfME[ec.rowNbr] using its formula above:
	//
	// E[ec.rowNbr][ec.colNbr] ~ nextEntryOfEAsFloat
	//
	//                             ec.rowNbr-1
	//                                sum      M[ec.rowNbr][k] E[k][ec.colNbr]
	//                             k=ec.colNbr
	//                         = - ___________________________________________
	//                                       M[ec.rowNbr][ec.rowNbr]
	//
	//                             ec.columnOfME[ec.rowNbr]
	//                         = - ________________________
	//                              M[ec.rowNbr][ec.rowNbr]
	//
	// This expression nextEntryOfEAsFloat uses only currently available information, so
	// it is computed now. The two possible round-offs of nextEntryOfEAsFloat to integers
	// are copied to ec.columnOfE[ec.rowNbr] and  clone.columnOfE[ec.rowNbr].
	ec.columnOfE[ec.rowNbr] = bignumber.NewFromInt64(0).Mul(
		ec.columnOfME[ec.rowNbr], ec.reciprocalDiagonal[ec.rowNbr],
	) // floating point for now
	entryOfEAsFloatIsNegative := ec.columnOfE[ec.rowNbr].IsNegative()
	ec.columnOfE[ec.rowNbr].RoundTowardsZero() // converted to an integer
	if entryOfEAsFloatIsNegative {
		clone.columnOfE[ec.rowNbr] = bignumber.NewFromInt64(0).Add(
			ec.columnOfE[ec.rowNbr], bignumber.NewFromInt64(-1),
		)
	} else {
		clone.columnOfE[ec.rowNbr] = bignumber.NewFromInt64(0).Add(
			ec.columnOfE[ec.rowNbr], bignumber.NewFromInt64(1),
		)
	}

	// Update equalsIdentityColumn
	if !ec.columnOfE[ec.rowNbr].IsZero() {
		ec.equalsIdentityColumn = false
	}
	if !clone.columnOfE[ec.rowNbr].IsZero() {
		clone.equalsIdentityColumn = false
	}

	// Perform the following updates
	// - Compute the final ec.columnOfME[ec.rowNbr] and clone.columnOfME[ec.rowNbr]
	// - Update the provisional values of ec.columnOfME[i] and clone.columnOfME for
	//   i in { ec.rowNbr+1, ..., ec.mNumRows-1 }
	//
	// For either c = ec or c = clone, the formula for ME[ec.rowNbr][ec.colNbr], based on
	// c.columnOfE, is
	//
	//                                ec.rowNbr
	//     c.columnOfME[ec.rowNbr] =     sum      M[ec.rowNbr][k] c.columnOfE[k]
	//                                k=ec.colNbr
	//
	// The current provisional value of c.columnOfME[ec.rowNbr] is
	//
	//                               ec.rowNbr-1
	//     c.columnOfME[ec.rowNbr] =     sum     M[ec.rowNbr][k] c.columnOfE[k]
	//                               k=ec.colNbr
	//
	// Therefore, adding M[ec.rowNbr][ec.rowNbr] c.columnOfE[ec.rowNbr] to c.columnOfME[ec.rowNbr]
	// sets it to its final value.
	//
	// Also, for i > ec.rowNbr, adding M[i][ec.rowNbr] c.columnOfE[ec.rowNbr] to c.columnOfME[i]
	// updates c.columnOfME[i] to the provisional value expected in the next call to this update
	// function.
	for i := ec.rowNbr; i < ec.mNumRows; i++ {
		mIJ, err := ec.m.Get(i, ec.rowNbr)
		if err != nil {
			return clone, fmt.Errorf("%s: could not get M[%d][%d]", caller, i, ec.rowNbr)
		}

		ec.columnOfME[i].MulAdd(mIJ, ec.columnOfE[ec.rowNbr])
		clone.columnOfME[i].MulAdd(mIJ, clone.columnOfE[ec.rowNbr])
	}

	// Update ec.squaredNorm and clone.squaredNorm with the sum of squares of columnOfME[i]
	// for i in { ec.colNbr, ... ec.rowNbr }. The current value of each squaredNorm is the sum
	// of squares of columnOfME for i in { ec.colNbr, ..., ec.rowNbr-1 }.
	//
	// After this update, squaredNorm incorporates final values of ME[i][ec.colNbr] for
	// i in { ec.colNbr, ..., ec.rowNbr } and no contribution from ME[i][ec.colNbr] for
	// i in { ec.rowNbr+1, ..., ec.mNumCols }.
	//
	meIJAsFloat64, _ := ec.columnOfME[ec.rowNbr].AsFloat().Float64()
	ec.squaredNorm += meIJAsFloat64 * meIJAsFloat64
	meIJAsFloat64, _ = clone.columnOfME[ec.rowNbr].AsFloat().Float64()
	clone.squaredNorm += meIJAsFloat64 * meIJAsFloat64
	return clone, nil
}

// isBoundedByDiagonal returns whether every entry of ec.columnOfME below the diagonal, up
// to and including ec.rowNbr, is bounded in absolute value by half the diagonal element
// of M to its right.
func (ec *EColumn) isBoundedByDiagonal(tolerance *bignumber.BigNumber, caller string) (bool, error) {
	caller = fmt.Sprintf("%s-isBoundedByDiagonal", caller)
	half := bignumber.NewPowerOfTwo(-1)
	for i := ec.colNbr + 1; i <= ec.rowNbr; i++ {
		mII, err := ec.m.Get(i, i)
		if err != nil {
			return true, fmt.Errorf(
				"%s: could not get M[%d][%d]: %q", caller, i, i, err.Error(),
			)
		}
		absMII := bignumber.NewFromInt64(0).Abs(mII)
		halfAbsMII := bignumber.NewFromInt64(0).Mul(half, absMII)
		absMEIJ := bignumber.NewFromInt64(0).Abs(ec.columnOfME[i])
		absMEIJMinusTolerance := bignumber.NewFromInt64(0).Sub(absMEIJ, tolerance)
		if absMEIJMinusTolerance.Cmp(halfAbsMII) > 0 {
			// Even after subtracting a small tolerance from |ME[i][colNbr]|,
			// |ME[i][colNbr]| > |M[i][i]|/2. Therefore, column colNbr is considered
			// to be unbounded by the diagonal.
			return false, nil
		}
	}
	return true, nil
}

// printEColumn prints eColumn to the returned string. If printOps are not provided,
// all fields are printed. I printOps is provided, it should be one string -- ideally
// with commas separating any two or more of the following strings -- to indicate that
// the corresponding field(s) should be printed.
//
// - "address": print the address of eColumn
//
// - "rowNbr": print eColumn.rowNbr
//
// - "colNbr": print eColumn.colNbr
//
// - "squaredNorm": print eColumn.squaredNorm
//
// - "columnOfE": print eColumn.columnOfE
//
// - "columnOfME": print eColumn.columnOfME
//
// - "matrixM": print eColumn.m
func printEColumn(eColumn *EColumn, context string, printOpts ...string) string {
	doPrint := struct {
		address     bool
		rowNbr      bool
		colNbr      bool
		squaredNorm bool
		columnOfE   bool
		columnOfME  bool
		matrixM     bool
	}{
		address:     (printOpts == nil) || strings.Contains(printOpts[0], "address"),
		rowNbr:      (printOpts == nil) || strings.Contains(printOpts[0], "rowNbr"),
		colNbr:      (printOpts == nil) || strings.Contains(printOpts[0], "colNbr"),
		squaredNorm: (printOpts == nil) || strings.Contains(printOpts[0], "squaredNorm"),
		columnOfE:   (printOpts == nil) || strings.Contains(printOpts[0], "columnOfE"),
		columnOfME:  (printOpts == nil) || strings.Contains(printOpts[0], "columnOfME"),
		matrixM:     (printOpts == nil) || strings.Contains(printOpts[0], "matrixM"),
	}

	retVal := fmt.Sprintf("%s\n", context)
	if doPrint.address {
		retVal += fmt.Sprintf("address     : %p\n", eColumn)
	}
	if doPrint.rowNbr {
		retVal += fmt.Sprintf("rowNbr      : %d\n", eColumn.rowNbr)
	}
	if doPrint.colNbr {
		retVal += fmt.Sprintf("colNbr      : %d\n", eColumn.colNbr)
	}
	if doPrint.squaredNorm {
		retVal += fmt.Sprintf("squaredNorm : %f\n", eColumn.squaredNorm)
	}
	if doPrint.columnOfE {
		retVal += fmt.Sprintf("columnOfE   : [ ")
		for i := 0; i <= eColumn.rowNbr; i++ {
			entryAsStr, _ := eColumn.columnOfE[i].String()
			retVal += fmt.Sprintf("%s ", entryAsStr)
		}
		retVal += "]\n"
	}
	if doPrint.columnOfME {
		retVal += fmt.Sprintf("columnOfME      : [ ")
		for i := 0; i <= eColumn.rowNbr; i++ {
			entryAsStr, _ := eColumn.columnOfME[i].String()
			retVal += fmt.Sprintf("%s ", entryAsStr)
		}
		retVal += "]\n"
	}
	if doPrint.matrixM {
		retVal += fmt.Sprintf("m:\n%v\n", eColumn.m)
	}
	return retVal
}

type EColumnArray struct {
	colNbr          int
	halfLength      int
	fullLength      int
	maxUsableSqNorm float64 // Any usable EColumn C satisfies |MC|^2 < maxUsableSqNorm
	candidateCol    []*EColumn
}

// newEColumnArray returns a new EColumnArray for column colNbr of M with up to
// 2^maxTreeDepth EColumns containing the top candidate EColumns populated with rows
// colNbr, colNbr+1, ..., min(colNbr+maxTreeDepth, m.NumRows()-1).
//
// If there is an error, the error is returned and the EColumnArray returned may be invalid
// or nil.
func newEColumnArray(
	m *bigmatrix.BigMatrix, reciprocalDiagonal []*bignumber.BigNumber,
	colNbr, maxTreeDepth int, caller string,
) (*EColumnArray, error) {
	// Error checks
	caller = fmt.Sprintf("%s-newEColumnArray", caller)
	mNumRows := m.NumRows()
	if maxTreeDepth < 0 {
		return nil, fmt.Errorf("%s: maxTreeDepth = %d < 0", caller, maxTreeDepth)
	}
	if (colNbr < 0) || (mNumRows <= colNbr) {
		return nil, fmt.Errorf(
			"%s: colNbr = %d is not in {0,...,%d}", caller, colNbr, mNumRows-1,
		)
	}
	if m.NumRows() <= maxTreeDepth {
		// No column of M has enough elements below the diagonal to fill a tree of
		// EColumns of depth maxTreeDepth.
		return nil, fmt.Errorf(
			"%s: maxTreeDepth = %d > %d", caller, maxTreeDepth, m.NumRows()-1,
		)
	}

	// Compute the squared norm of a column of ME above which the corresponding column
	// of E cannot be used in getE().
	var maxSqNorm float64
	for i := 0; i < mNumRows; i++ {
		mIJ, err := m.Get(i, colNbr)
		if err != nil {
			return nil, fmt.Errorf("%s: could not get M[%d][%d]", caller, colNbr, colNbr)
		}

		// Compute the final squared norm
		mIJSq := bignumber.NewFromInt64(0).Mul(mIJ, mIJ)
		mIJSqAsFloat64, _ := mIJSq.AsFloat().Float64()
		maxSqNorm += mIJSqAsFloat64
	}

	// Create the EColumnArray.
	treeDepth := maxTreeDepth
	if treeDepth > (mNumRows-1)-colNbr {
		treeDepth = (mNumRows - 1) - colNbr
	}
	if treeDepth == 0 {
		// treeDepth is allowed to be zero, to indicate no attempt to reduce column
		// colNbr of E. Calling update on such an instance is a no-op that returns false.
		return &EColumnArray{
			colNbr:          colNbr,
			halfLength:      0,
			fullLength:      1,
			maxUsableSqNorm: maxSqNorm,
			candidateCol:    []*EColumn{nil},
		}, nil
	}
	retVal := &EColumnArray{
		colNbr:          colNbr,
		halfLength:      1 << (treeDepth - 1),
		fullLength:      1 << treeDepth,
		maxUsableSqNorm: maxSqNorm,
		candidateCol:    make([]*EColumn, 1<<treeDepth),
	}

	// Populate the first half of retVal.candidateCol. The second half is reserved for the
	// first call to EColumnArray.update(). Note that even if treeDepth == 1, one entry
	// of candidateCol is populated.
	var err error
	lastPopulatedIndex := 0
	retVal.candidateCol[lastPopulatedIndex], err = newEColumn(m, reciprocalDiagonal, colNbr, caller)
	for partialTreeDepth := 0; partialTreeDepth < treeDepth-1; partialTreeDepth++ {
		// Update high indices of candidateCol using lower indices as inputs.
		for i := lastPopulatedIndex; 0 <= i; i-- {
			retVal.candidateCol[2*i+1], err = retVal.candidateCol[i].update(caller)
			if err != nil {
				return nil, err
			}
			if i > 0 {
				retVal.candidateCol[2*i] = retVal.candidateCol[i] // never a no-op since i != 0
			}
		}
		lastPopulatedIndex = 2*lastPopulatedIndex + 1 // max of 2*i+1 from previous loop
	}

	// Sort candidate columns by squared norm to cover edge cases where EColumnArray.update() is
	// called just once and exits before sorting the candidate columns.
	if retVal.halfLength > 1 {
		sort.Slice(retVal.candidateCol[0:retVal.halfLength], func(i, j int) bool {
			return retVal.candidateCol[i].squaredNorm < retVal.candidateCol[j].squaredNorm
		})
	}
	return retVal, nil
}

// update computes the last eca.halfLength entries in eca.candidateCol, then sorts all
// instances by increasing squaredNorm, and removes the last (largest) eca.halfLength
// instances.
//
// update returns true if eca.candidateCol was updated, otherwise false. A return value of false
// indicates that each eca.candidateCol[i].columnOfE was fully populated before update was called.
//
// update does nothing and returns false on an instance with halfLength == 0 (i.e., one
// created with maxTreeDepth == 0).
func (eca *EColumnArray) update(caller string) (bool, error) {
	// Handle overt errors
	var err error
	caller = fmt.Sprintf("%s-update", caller)
	if eca.candidateCol[0] == nil {
		return false, nil
	}
	if eca.candidateCol[0].mNumRows-1 <= eca.candidateCol[0].rowNbr {
		// The current row is the last one in M. No further updates of candidate columns
		// are possible.
		return false, nil
	}

	// Increment the current row in each element of candidateCol. Each candidate column
	// is bifurcated into a pair and placed further towards the end of the array.
	for i := eca.halfLength - 1; 0 <= i; i-- {
		if eca.candidateCol[i] == nil {
			return false, fmt.Errorf("%s: called without candidate column %d", caller, i)
		}
		eca.candidateCol[2*i+1], err = eca.candidateCol[i].update(caller)
		if err != nil {
			return false, err
		}
		if i > 0 {
			eca.candidateCol[2*i] = eca.candidateCol[i] // not a no-op since i != 0
		}
	}

	// Sort by squared norm as of row rowNbr
	if eca.fullLength > 1 {
		sort.Slice(eca.candidateCol, func(i, j int) bool {
			return eca.candidateCol[i].squaredNorm < eca.candidateCol[j].squaredNorm
		})
	}

	// Delete the high-norm half of the candidate columns
	for i := eca.halfLength; i < eca.fullLength; i++ {
		eca.candidateCol[i] = nil
	}
	return true, nil
}

// bestColumn returns the best column of E computed by a greedy algorithm. This could be
// the identity column.
func (eca *EColumnArray) bestColumn(caller string) (*EColumn, error) {
	const tolerance = 1.e-5
	caller = fmt.Sprintf("%s-bestColumn", caller)

	// Return nil if this is the last column
	if eca.halfLength == 0 {
		return nil, nil
	}

	// Compute the best eca.halfLength columns of column colNbr E, in terms of the
	// squared norm of the same column of ME.
	keepGoing := true
	for keepGoing {
		var err error
		keepGoing, err = eca.update(caller)
		if err != nil {
			return nil, fmt.Errorf(
				"%s: could update of EColumnArray failed: %q", caller, err.Error(),
			)
		}
	}

	// If the best candidate equals the identity column or has squaredNorm below
	// that induced by the identity column (which is eca.maxUsableSqNorm), return the
	// best candidate.
	//
	// To prevent churn in the overall algorithm caused by non-identity columns being
	// returned that merely match the performance of the identity column but do not
	// exceed it, the comparison below uses a slightly reduce eca.maxUsableSqNorm.
	// This reduction weeds out this case even if there is round-off error.
	retVal := eca.candidateCol[0]
	if retVal == nil {
		return nil, fmt.Errorf(
			"%s: no candidate columns found though halfLength = %d", caller, eca.halfLength,
		)
	}
	if (retVal.equalsIdentityColumn) || (retVal.squaredNorm < eca.maxUsableSqNorm-tolerance) {
		return retVal, nil
	}

	// The best candidate is not the identity column and does not beat the identity column,
	// at least not by any more than (approximately) the tolerance.
	return nil, nil
}
