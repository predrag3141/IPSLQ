// Package pslqops performs operations specific to the PSLQ algorithm
package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
)

const (
	// Round-off is a concern in the PSLQ algorithm, so it is tracked with a
	// rudimentary infinite impulse response filter. The input to this filter
	// is the observed error in the calculation of an invariant of PSLQ. The
	// filtered, observed round-off error is updated to
	//
	// 0.95 s.observedRoundOffError + 0.05 currentRoundOffError
	//
	// As the above formula indicates, the weight given the current round-off
	// error is 0.05. This is controlled by:
	roundOffCurrentWeight = "0.05"
)

// State holds the state of a running PSLQ algorithm
type State struct {
	rawX                  *bigmatrix.BigMatrix
	rawXB                 *bigmatrix.BigMatrix
	h                     *bigmatrix.BigMatrix
	m                     *bigmatrix.BigMatrix
	a                     *bigmatrix.BigMatrix
	b                     *bigmatrix.BigMatrix
	d                     *bigmatrix.BigMatrix
	e                     *bigmatrix.BigMatrix
	numRows               int // number of rows in H
	numCols               int // number of columns in H, and the dimension of M
	observedRoundOffError *bignumber.BigNumber
	roundOffCurrentWeight *bignumber.BigNumber
	roundOffHistoryWeight *bignumber.BigNumber
	solutionCount         int
	hReductionCount       int // How many times H was row reduced by D
	mReductionCount       int // How many times M was column reduced by E
	hIterationCount       int // How many times H was iterated on, with or without reduction
	mIterationCount       int // How many times M was iterated on, with or without reduction
	allZeroRowsCalculated int64
}

// NewState returns a new State from a provided decimal string array
func NewState(input []string) (*State, error) {
	// var rawX *bigmatrix.BigMatrix = GetRawX(input)
	rawX, err := getRawX(input, "NewState")
	if err != nil {
		return nil, err
	}
	nr := len(input)
	retVal := &State{
		rawX:                  rawX,
		rawXB:                 bigmatrix.NewEmpty(1, nr),
		h:                     nil,
		m:                     nil,
		a:                     bigmatrix.NewEmpty(nr, nr),
		b:                     bigmatrix.NewEmpty(nr, nr),
		d:                     bigmatrix.NewEmpty(nr, nr),
		e:                     bigmatrix.NewEmpty(nr, nr),
		numRows:               nr,
		numCols:               nr - 1,
		observedRoundOffError: bignumber.NewFromInt64(0),
		roundOffCurrentWeight: nil,
		solutionCount:         0,
		allZeroRowsCalculated: 0,
	}

	// Initialize rawXB
	for i := 0; i < retVal.numRows; i++ {
		var initialValue *bignumber.BigNumber
		initialValue, err = retVal.rawX.Get(0, i)
		if err != nil {
			return nil, fmt.Errorf("NewState: could not get rawX[0][%d]: %q", i, err.Error())
		}
		err = retVal.rawXB.Set(0, i, bignumber.NewFromBigNumber(initialValue))
		if err != nil {
			return nil, fmt.Errorf("NewState: could not set updatedRawX[0][%d]: %q", i, err.Error())
		}
	}

	// Initialize A and B
	for i := 0; i < retVal.numRows; i++ {
		for j := 0; j < retVal.numRows; j++ {
			var initialValue int64
			if i == j {
				initialValue = 1
			}
			err = retVal.a.Set(i, j, bignumber.NewFromInt64(initialValue))
			if err != nil {
				return nil, fmt.Errorf("NewState: could not set A[%d][%d]: %q", i, j, err.Error())
			}
			err = retVal.b.Set(i, j, bignumber.NewFromInt64(initialValue))
			if err != nil {
				return nil, fmt.Errorf("NewState: could not set B[%d][%d]: %q", i, j, err.Error())
			}
		}
	}

	// Initialize H
	normalizedX, err := getNormalizedX(retVal.rawX, "NewState")
	if err != nil {
		return nil, err
	}
	s, err := getS(normalizedX, "NewState")
	if err != nil {
		return nil, err
	}
	retVal.h, err = getH(normalizedX, s, "NewState")
	if err != nil {
		return nil, err
	}

	// Initialize round-off input and output weights
	retVal.roundOffCurrentWeight, err = bignumber.NewFromDecimalString(roundOffCurrentWeight)
	if err != nil {
		return nil, fmt.Errorf(
			"NewState: could not parse round-off input weight string %q", roundOffCurrentWeight,
		)
	}
	retVal.roundOffHistoryWeight = bignumber.NewFromInt64(0).Sub(
		bignumber.NewFromInt64(1), retVal.roundOffCurrentWeight,
	)
	err = retVal.updateRoundOffError(true, "NewState")
	if err != nil {
		return nil, err
	}

	// retVal is fully defined
	return retVal, nil
}

// OneIteration performs one iteration of the PSLQ algorithm, updating either
//
//   - H, A and B the way the original 1992 PSLQ paper says to in section 5, page 5;
//     except R is from the client-provided getR; or
//
// - M, A and B using column operations
//
// which is not necessarily consistent with the paper. OneIteration returns a bool
// indicating whether to terminate.
//
// getR() will be passed H or M. The behavior of getR depends on whether
// the matrix it is passed has one more row than column -- indicating it is
// H -- or is square, which indicates that it is H.
func (s *State) OneIteration(
	getR func(*bigmatrix.BigMatrix) (*IntOperation, error),
	checkInvariantsOptional ...bool,
) (bool, error) {
	fmt.Printf("==========\nB:\n%v\nH:\n%v\n", s.b, s.h) // debug
	// Initializations
	checkInvariants := false
	if len(checkInvariantsOptional) > 0 {
		checkInvariants = checkInvariantsOptional[0]
	}

	// Step 1 of this PSLQ iteration
	err := s.step1("OneIteration")
	if err != nil {
		return false, err
	}

	// Check invariants expected after step 1.
	//
	// Note: The place for invariants to hold -- after step 1 instead of after step 3
	//       (or, equivalently, before step 1) -- follows from the original 1992 paper.
	if checkInvariants {
		err = s.checkInvariants("OneIteration")
		if err != nil {
			return false, err
		}
		err = s.updateRoundOffError(false, "OneIteration")
		if err != nil {
			return false, err
		}
	}

	// Step 2 of this PSLQ iteration
	//
	// H was just row-reduced, or M was just column-reduced, depending on which
	// phase the PSLQ algorithm is in.
	var intOperation *IntOperation
	if s.m == nil {
		intOperation, err = getR(s.h)
		if (err == nil) && (intOperation == nil) {
			// It is here that the transition from H to M occurs. The function, transitionFromHtoM,
			// performs this transition and signals whether to terminate (which is unlikely).
			var terminationFlag bool
			intOperation, terminationFlag, err = s.transitionFromHtoM(getR, "OneIteration")
			if terminationFlag {
				return true, err
			}
		}
	} else {
		intOperation, err = getR(s.m)
		if (err == nil) && (intOperation == nil) {
			return true, nil
		}
	}
	if err != nil {
		return false, fmt.Errorf("OneIteration: error from getR: %q", err.Error())
	}
	err = intOperation.validateAll(s.numRows, "OneIteration")
	if err != nil {
		return false, err
	}

	// Step 3 of this PSLQ iteration
	err = s.step3(intOperation, "OneIteration")
	if err != nil {
		return false, err
	}

	// Update counts
	if s.m == nil {
		s.hIterationCount++
	} else {
		s.mIterationCount++
	}
	return false, nil
}

// IterationCounts returns the total number of iterations involving H and M, respectively. This does
// not count the time M was first reduced, after being created by inverting H.
func (s *State) IterationCounts() (int, int) {
	return s.hIterationCount, s.mIterationCount
}

// ReductionCounts returns the total number of iterations during which H and M were reduced,
// respectively.
func (s *State) ReductionCounts() (int, int) {
	return s.hReductionCount, s.mReductionCount
}

func (s *State) GetAllZeroRowsCalculated() int64 {
	return s.allZeroRowsCalculated
}

func (s *State) GetObservedRoundOffError() *bignumber.BigNumber {
	return s.observedRoundOffError
}

// GetXB returns the current matrix of reduced inputs. If rawX is
// the matrix of inputs and B is the matrix of the same name in the original
// 1992 PSLQ paper, updatedRawX is (rawX)(B).
func (s *State) GetXB() *bigmatrix.BigMatrix {
	return s.rawXB
}

// GetDiagonal returns an instance of DiagonalStatistics
//
// The elements of the returned diagonal are deep copies.
func (s *State) GetDiagonal() (*DiagonalStatistics, error) {
	if s.m != nil {
		return NewDiagonalStatistics(s.m)
	}
	return NewDiagonalStatistics(s.h)
}

// GetSolutions returns an array of []int64 arrays that are solutions. An error is returned
// if there is a failure.
func (s *State) GetSolutions() (map[int][]int64, error) {
	retVal := make(map[int][]int64, 0)
	for j := 0; j < s.numRows; j++ {
		columnOfB, err := s.GetColumnOfB(j)
		if err != nil {
			return map[int][]int64{}, fmt.Errorf(
				"GetSolutions: could not get column %d of H: %q", j, err.Error(),
			)
		}
		dotProduct := bignumber.NewFromInt64(0)
		for i := 0; i < s.numRows; i++ {
			var xi *bignumber.BigNumber
			xi, err = s.rawX.Get(0, i)
			dotProduct.Int64MulAdd(columnOfB[i], xi)
		}
		if dotProduct.IsSmall() {
			retVal[j] = columnOfB
		}
	}
	return retVal, nil
}

// GetRowOfA returns a row of A
func (s *State) GetRowOfA(row int) ([]int64, error) {
	retVal := make([]int64, s.numRows)
	for i := 0; i < s.numRows; i++ {
		aEntryAsBigNumber, err := s.a.Get(row, i)
		if err != nil {
			return nil, fmt.Errorf(
				"GetRowOfA: could not get A[%d][%d]: %q", row, i, err.Error(),
			)
		}
		var aEntryAsInt64 int64
		aEntryAsInt64, err = aEntryAsBigNumber.AsInt64()
		if err != nil {
			return nil, fmt.Errorf(
				"GetRowOfA: could not convert A[%d][%d] as an integer: %q",
				row, i, err.Error(),
			)
		}
		retVal[i] = aEntryAsInt64
	}
	return retVal, nil
}

// GetColumnOfB returns a column of B, which is an approximate or exact solution
// of <x,.> = 0, depending on how far the algorithm has progressed. If there are
// entries in the column of B that do not fit into an int64, the column being
// requested is presumably the one that does not represent a solution. In that
// case, the returned array is empty. If there is an error (which does not include
// the case where entries do not fit into an int64), an error is returned.
func (s *State) GetColumnOfB(col int) ([]int64, error) {
	retVal := make([]int64, s.numRows)
	for i := 0; i < s.numRows; i++ {
		bEntryAsBigNumber, err := s.b.Get(i, col)
		if err != nil {
			return []int64{}, fmt.Errorf(
				"GetColumnOfB: could not get B[%d][%d]: %q", i, col, err.Error(),
			)
		}
		var bEntryAsInt64 int64
		bEntryAsInt64, err = bEntryAsBigNumber.AsInt64()
		if err != nil {
			return []int64{}, nil
		}
		retVal[i] = bEntryAsInt64
	}
	return retVal, nil
}

// NumCols returns the number of columns in s.h
func (s *State) NumCols() int {
	return s.numCols
}

// NumRows returns the number of columns in s.h
func (s *State) NumRows() int {
	return s.numRows
}

// UsingInverse returns whether M, the inverse of H, is being used
func (s *State) UsingInverse() bool {
	return s.m != nil
}

func (s *State) checkInvariants(caller string) error {
	if s.m == nil {
		// Check that H has essentially-zero above the diagonal
		for i := 0; i < s.numRows; i++ {
			for j := i + 1; j < s.numCols; j++ {
				hij, err := s.h.Get(i, j)
				if err != nil {
					return fmt.Errorf(
						"%s: could not get H[%d][%d]: %q", caller, i, j, err.Error(),
					)
				}
				if !hij.IsSmall() {
					_, hijAsStr := hij.String()
					return fmt.Errorf("%s: H[%d][%d] = %q is not small", caller, i, j, hijAsStr)
				}
			}
		}
	} else {
		for i := 0; i < s.numCols; i++ {
			for j := i + 1; j < s.numCols; j++ {
				mij, err := s.m.Get(i, j)
				if err != nil {
					return fmt.Errorf(
						"%s: could not get M[%d][%d]: %q", caller, i, j, err.Error(),
					)
				}
				if !mij.IsSmall() {
					_, hijAsStr := mij.String()
					return fmt.Errorf("%s: M[%d][%d] = %q is not small", caller, i, j, hijAsStr)
				}
			}
		}
	}

	// Check that A and B are inverses of each other
	var ab *bigmatrix.BigMatrix
	var err error
	ab, err = bigmatrix.NewEmpty(s.numRows, s.numRows).Mul(
		s.a, s.b,
	)
	if err != nil {
		return fmt.Errorf(
			"%s: could not multiply %v by %v: %q",
			caller, s.a, s.b, err.Error(),
		)
	}
	for i := 0; i < s.numRows; i++ {
		for j := 0; j < s.numRows; j++ {
			expected := int64(0)
			if i == j {
				expected = 1
			}
			var abIJ *bignumber.BigNumber
			abIJ, err = ab.Get(i, j)
			if err != nil {
				return fmt.Errorf(
					"%s: could not get AB[%d][%d]: %q", caller, i, j, err.Error(),
				)
			}
			if abIJ.Cmp(bignumber.NewFromInt64(expected)) != 0 {
				_, abIJAsString := abIJ.String()
				return fmt.Errorf(
					"%s: AB[%d][%d] = %q != %d\nA = %v\nB = %v. Using big numbers: Yes",
					caller, i, j, abIJAsString, expected, s.a, s.b,
				)
			}
		}
	}

	// Check that H has been row reduced or M has been column reduced, whichever is applicable.
	var row, col int
	if s.m == nil {
		var hIsReduced bool
		hIsReduced, row, col, err = isRowReduced(
			s.h, -10, caller,
		)
		if err != nil {
			return fmt.Errorf(
				"%s: error determining whether H is row reduced: %q",
				caller, err.Error(),
			)
		}
		if !hIsReduced {
			return fmt.Errorf("%s: H[%d][%d] is not reduced. H=\n%v",
				caller, row, col, s.h,
			)
		}
	} else {
		var mIsReduced bool
		mIsReduced, row, col, err = isColumnReduced(s.m, -10, "OneIteration")
		if err != nil {
			return fmt.Errorf(
				"%s: error determining whether M is column reduced: %q",
				caller, err.Error(),
			)
		}
		if !mIsReduced {
			return fmt.Errorf("%s: M[%d][%d] is not reduced. M=\n%v",
				caller, row, col, s.m,
			)
		}
	}

	// Check that Lemma 10 of the 1999 analysis of PSLQ holds when the last element of H,
	// H[numRows-1][numCols-1], is zero. When that element is zero,
	// (norm of column numCols-1 of B)|H[numCols-1][numCols-1]| = 1
	//
	// Check the equivalent of the above if s.m != nil, i.e.
	// (norm of column numCols-1 of B) = |M[numCols-1][numCols-1]|
	var lastDiagonalElement, absLastDiagonalElement *bignumber.BigNumber
	if s.m == nil {
		var lastElementOfH *bignumber.BigNumber
		lastElementOfH, err = s.h.Get(s.numRows-1, s.numCols-1)
		if err != nil {
			return fmt.Errorf(
				"%s: could not get H[%d][%d]: %q", caller, s.numRows-1, s.numCols-1, err.Error(),
			)
		}
		if lastElementOfH.IsSmall() {
			lastDiagonalElement, err = s.h.Get(s.numCols-1, s.numCols-1)
		}
	} else {
		lastDiagonalElement, err = s.m.Get(s.numCols-1, s.numCols-1)
	}
	if lastDiagonalElement != nil {
		tolerance := bignumber.NewPowerOfTwo(-50)
		var norm, diff *bignumber.BigNumber
		var lastColumnOfB []int64
		var normSq int64
		var equals bool
		var errorMessage string
		absLastDiagonalElement = bignumber.NewFromInt64(0).Abs(lastDiagonalElement)
		lastColumnOfB, err = s.GetColumnOfB(s.numCols - 1)
		for i := 0; i < s.numRows; i++ {
			normSq += lastColumnOfB[i] * lastColumnOfB[i]
		}
		norm, err = bignumber.NewFromInt64(0).Sqrt(bignumber.NewFromInt64(normSq))
		_, absLastDiagonalElementAsStr := absLastDiagonalElement.String()
		_, normAsStr := norm.String()
		if s.m == nil {
			shouldBeOne := bignumber.NewFromInt64(0).Mul(norm, absLastDiagonalElement)
			diff = bignumber.NewFromInt64(0).Sub(shouldBeOne, bignumber.NewFromInt64(1))
			_, shouldBeOneAsStr := shouldBeOne.String()
			_, diffAsStr := diff.String()
			errorMessage = fmt.Sprintf(
				"|H[%d][%d]| = %s; |B[%d]| = %s; (%s)(%s) = %s != 1 (error = %s)",
				s.numCols-1, s.numCols-2, absLastDiagonalElementAsStr, s.numCols-1, normAsStr,
				absLastDiagonalElementAsStr, normAsStr, shouldBeOneAsStr, diffAsStr,
			)
		} else {
			diff = bignumber.NewFromInt64(0).Sub(absLastDiagonalElement, norm)
			_, diffAsStr := diff.String()
			errorMessage = fmt.Sprintf(
				"|M[%d][%d]| = %s != %s = |B[%d]| (error = %s)",
				s.numCols-1, s.numCols-2, absLastDiagonalElementAsStr,
				normAsStr, s.numCols-1, diffAsStr,
			)
		}
		equals = diff.Equals(bignumber.NewFromInt64(0), tolerance)
		if !equals {
			return fmt.Errorf("%s: %s", caller, errorMessage)
		}
	}

	// No errors were found
	return nil
}

// step1 performs step 1 of the PSLQ algorithm. This row-reduces H or column-reduces M,
// depending on whether s.m == nil.
func (s *State) step1(caller string) error {
	caller = fmt.Sprintf("%s-step1", caller)
	var isIdentity, calculatedAllZeroRow bool
	var err error

	// Compute D and E
	if s.m == nil {
		isIdentity, calculatedAllZeroRow, err = getD(s.h, s.d, caller)
		if err != nil {
			return err
		}
		if !isIdentity {
			s.hReductionCount++
		}
		_, err = s.e.InvertLowerTriangular(s.d)
		if err != nil {
			return fmt.Errorf("%s: could not invert D: %q", caller, err.Error())
		}
	} else {
		isIdentity, calculatedAllZeroRow, err = getE(s.m, s.e, caller)
		if err != nil {
			return err
		}
		if !isIdentity {
			s.mReductionCount++
		}
		_, err = s.d.InvertLowerTriangular(s.e)
		if err != nil {
			return fmt.Errorf("%s: could not invert E: %q", caller, err.Error())
		}
	}

	if calculatedAllZeroRow {
		s.allZeroRowsCalculated++
	}

	// Update H or M
	if s.m == nil {
		_, err = s.h.Mul(s.d, s.h)
		if err != nil {
			return fmt.Errorf("%s: could not compute DH: %q", caller, err.Error())
		}
	} else {
		_, err = s.m.MulUpperLeft(s.m, s.e)
		if err != nil {
			return fmt.Errorf("%s: could not compute ME: %q", caller, err.Error())
		}
	}

	// Update A, B and xB so that invariants still hold
	_, err = s.a.Mul(s.d, s.a)
	if err != nil {
		return fmt.Errorf("%s: could not compute DA: %q", caller, err.Error())
	}
	_, err = s.b.Mul(s.b, s.e)
	if err != nil {
		return fmt.Errorf("%s: could not compute BE: %q", caller, err.Error())
	}
	_, err = s.rawXB.Mul(s.rawXB, s.e)
	if err != nil {
		return fmt.Errorf("%s: could not compute xB: %q", caller, err.Error())
	}
	return nil
}

func (s *State) step3(intOperation *IntOperation, caller string) error {
	var err error
	caller = fmt.Sprintf("%s-step3", caller)

	// Update A: A <- RA
	err = intOperation.performRowOp(s.a, caller)
	if err != nil {
		return err
	}

	// Update B: B <- BR^-1
	err = intOperation.performColumnOp(s.b, caller)
	if err != nil {
		return err
	}

	// Update XB: XB <- XBR^-1
	err = intOperation.performColumnOp(s.rawXB, caller)
	if err != nil {
		return err
	}

	// Update H or M. These updates involve the rigid operator that removes corners, which
	// is not a part of the updates for A, B and xB.
	//
	// H <- RHG or M <- (G^-1)(M)(R^-1)
	if s.m == nil {
		err = intOperation.performRowOp(s.h, caller)
		if err != nil {
			return err
		}
		_, err = s.e.InvertLowerTriangular(s.d)
		if err != nil {
			return fmt.Errorf("%s: could not compute big matrix E: %q", caller, err.Error())
		}
		if intOperation.Indices[len(intOperation.Indices)-1] < s.numCols {
			// Since rowOperation does not just swap the last two rows, a corner needs removing
			err = removeCornerOfH(s.h, intOperation.Indices, caller)
			if err != nil {
				return err
			}
		}
	} else {
		err = intOperation.performColumnOp(s.m, caller)
		if err != nil {
			return err
		}
		_, err = s.d.InvertLowerTriangular(s.e)
		if err != nil {
			return fmt.Errorf("%s: could not compute D from E: %q", caller, err.Error())
		}
		err = removeCornerOfM(s.m, intOperation.Indices, caller)
		if err != nil {
			return err
		}
	}
	return nil
}

// transitionFromHtoM creates s.m as the inverse of H. Then it repeats step 1 and part of
// step 2 to enable OneIteration to pick up where it paused to call transitionFromHtoM.
func (s *State) transitionFromHtoM(
	getR func(matrix *bigmatrix.BigMatrix) (*IntOperation, error), caller string,
) (*IntOperation, bool, error) {
	caller = fmt.Sprintf("%s-transitionFromHtoM", caller)
	var err error
	s.m, err = bigmatrix.NewEmpty(s.h.NumCols(), s.h.NumCols()).InvertLowerTriangular(s.h)
	if err != nil {
		return nil, true, fmt.Errorf("%s: error inverting H: %q", caller, err.Error())
	}
	err = s.step1(caller)
	if err != nil {
		return nil, false, err
	}

	// M is now column-reduced.
	terminationFlag := false
	var intOperation *IntOperation
	intOperation, err = getR(s.m)
	if err != nil {
		return nil, true, fmt.Errorf("%s: error in getR: %q", caller, err.Error())
	}
	if intOperation == nil {
		// There should be plenty of iterations to come on M, so reaching here is possible but
		// not likely.
		terminationFlag = true
	}
	return intOperation, terminationFlag, err
}

func (s *State) updateRoundOffError(firstIteration bool, caller string) error {
	caller = fmt.Sprintf("%s-updateRoundOffError", caller)
	var err error
	var oneRoundOffError *bignumber.BigNumber
	maxRoundOffError := bignumber.NewFromInt64(0)
	for j := 0; j < s.numCols; j++ {
		oneRoundOffError, err = bigmatrix.DotProduct(
			s.rawXB, s.h, 0, j, 0, s.numRows, false,
		)
		if err != nil {
			return fmt.Errorf("%s: could not compute <raw X, column %d of h>", caller, j)
		}
		oneRoundOffError.Abs(oneRoundOffError)
		if maxRoundOffError.Cmp(oneRoundOffError) < 0 {
			maxRoundOffError.Set(oneRoundOffError)
		}
	}
	if firstIteration {
		s.observedRoundOffError.Set(maxRoundOffError)
		return nil
	}
	s.observedRoundOffError.Mul(s.observedRoundOffError, s.roundOffHistoryWeight)
	s.observedRoundOffError.MulAdd(maxRoundOffError, s.roundOffCurrentWeight)
	return nil
}
