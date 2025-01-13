package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"math"

	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
)

// BottomRightOfH holds rightmost non-zero element, U, of the last row of H, along
// with T, the element above U. W is the element to the left of T, and V is above W.
type BottomRightOfH struct {
	Found        bool
	T            *bignumber.BigNumber
	RowNumberOfT int
	U            *bignumber.BigNumber
	RowNumberOfU int
}

// getBottomRightOfH returns v, w, t, u in the bottom-right of H for which zeroes appear
// as shown below, along with the row number in which t appears; or if zeroes in this
// pattern do not exist, NewBottomRightOfH returns the bottom-right 3x2 sub-matrix of H,
// which has the same form as the first two columns shown below. In the latter case, the
// row number returned is numRows-3, because t appears in that row.
//
//	_             _
//
// |  v  0  0 ...  |
// |  w  t  0 ...  |
// |  ...          |
// |_ *  u  0 ... _|
//
// A strategy can use such t, u, v, w and row number to define a row operation that reduces
// w just enough to swap the largest diagonal element down and to the right in the 2x2 sub-
// matrix whose upper-left element is t.
func getBottomRightOfH(h *bigmatrix.BigMatrix, caller string) (*BottomRightOfH, error) {
	caller = fmt.Sprintf("%s-getBottomRightOfH", caller)
	numRows := h.NumRows()
	lastNonZero, err := getLastNonZero(h, caller)
	if err != nil {
		return nil, err
	}
	if lastNonZero < 0 {
		//  There is no bottom-right of H in need of reduction. This is indicated
		//	by the fact that Found == false in the returned value.
		return &BottomRightOfH{
			Found:        false,
			T:            nil,
			RowNumberOfT: 0,
			U:            nil,
			RowNumberOfU: h.NumRows() - 1,
		}, nil
	}

	// Since lastNonZero is non-negative, there is a bottom-right of H in need of reduction in column
	// lastNonZero.
	var t, u *bignumber.BigNumber
	t, err = h.Get(lastNonZero, lastNonZero)
	if err != nil {
		return nil, fmt.Errorf(
			"GetBottomRightOfH: could not get H[%d][%d]: %q", lastNonZero, lastNonZero, err.Error(),
		)
	}
	u, err = h.Get(numRows-1, lastNonZero)
	if err != nil {
		return nil, fmt.Errorf(
			"GetBottomRightOfH: could not get H[%d][%d]: %q", numRows-1, lastNonZero, err.Error(),
		)
	}
	return &BottomRightOfH{
		Found:        true,
		T:            bignumber.NewFromInt64(0).Set(t),
		RowNumberOfT: lastNonZero,
		U:            bignumber.NewFromInt64(0).Set(u),
		RowNumberOfU: numRows - 1,
	}, nil
}

// reduce returns a non-nil row operation and nil error, or vice versa. The row operation
// that reduce returns (if it is non-nil) reduces |brh.T| to the point that one of the
// following happens.
//   - brh.U = 0 (technically brh.U.IsSmall() is true)
//   - brh.T < 2^log2threshold
//   - an entry in the matrix retVal that reduced brh.T and brh.U would exceed
//     maxRowOpEntry if reduction were to continue.
//
// Notes:
//   - brh must come from getBottomRightOfH, which means that it contains a diagonal element
//     of H, brh.T and last-row element brh.U in the same column of H as brh.T, with zeroes
//     to the right of brh.U (and, as for all diagonal elements, zeroes to the right of brh.T)
//   - Some non-identity row operation is always returned, because brh is constructed by
//     getBottomRightOfH so that brh.T and brh.U are non-zero.
func (brh *BottomRightOfH) reduce(maxRowOpEntry, log2threshold int, caller string) (*IntOperation, error) {
	caller = fmt.Sprintf("%s-reduce", caller)
	if !brh.Found {
		return nil, fmt.Errorf("BottomRightOfH.Reduce: empty bottom-right of H")
	}
	threshold := bignumber.NewPowerOfTwo(int64(log2threshold))
	var operationOnH []int
	err := reducePair(
		brh.T, brh.U, maxRowOpEntry, caller, func(rowOpMatrix []int) bool {
			operationOnH = []int{rowOpMatrix[0], rowOpMatrix[1], rowOpMatrix[2], rowOpMatrix[3]}
			absT := bignumber.NewFromInt64(0).Abs(brh.T)
			return absT.Cmp(threshold) <= 0
		},
	)
	if err != nil {
		return nil, err
	}
	det := operationOnH[0]*operationOnH[3] - operationOnH[1]*operationOnH[2] // 1 or -1
	return &IntOperation{
		Indices:      []int{brh.RowNumberOfT, brh.RowNumberOfU},
		OperationOnA: operationOnH,
		OperationOnB: []int{
			det * operationOnH[3], -det * operationOnH[1], -det * operationOnH[2], det * operationOnH[0],
		},
		PermutationOfA: [][]int{},
		PermutationOfB: [][]int{},
	}, nil
}

// getLastNonZero returns the last column number in H, starting from the left, that contains
// a not-essentially-zero element in the last row (hence only essentially zero values to its right).
// If there are no such columns of H (i.e. H[numRows-1] is all essentially zero), getLastNonZero
// returns -1.
func getLastNonZero(h *bigmatrix.BigMatrix, caller string) (int, error) {
	caller = fmt.Sprintf("%s-getLastNonZero", caller)
	numRows, numCols := h.Dimensions()
	for colNbr := numCols - 1; 0 <= colNbr; colNbr-- {
		hEntry, err := h.Get(numRows-1, colNbr)
		if err != nil {
			return 0, fmt.Errorf(
				"%s: could not get H[%d][%d]: %q", caller, numRows-1, colNbr, err.Error(),
			)
		}
		if !hEntry.IsSmall() {
			return colNbr, nil
		}
	}

	// This line should be reached if the last row of H is all essentially zero
	return -1, nil
}

type DiagonalStatistics struct {
	Diagonal []*bignumber.BigNumber
	Ratio    *float64
}

// NewDiagonalStatistics returns an instance of DiagonalStatistics for the input hm,
// which can be either an H matrix or its inverse, M.
func NewDiagonalStatistics(hm *bigmatrix.BigMatrix) (*DiagonalStatistics, error) {
	// Initializations
	numCols := hm.NumCols()
	var err error
	matrixName := "M"
	hmIsH := hm.NumRows() != hm.NumCols()
	if hmIsH {
		matrixName = "H"
	}
	retVal := &DiagonalStatistics{
		Diagonal: make([]*bignumber.BigNumber, numCols),
		Ratio:    nil,
	}
	largestDiagonalElement := bignumber.NewFromInt64(0)              // Largest diagonal element of H
	smallestDiagonalElement := bignumber.NewFromInt64(math.MaxInt64) // Smallest diagonal element of M

	// Iterate through the diagonal of H, updating norm and largestDiagonalElement
	for i := 0; i < numCols; i++ {
		var retValI *bignumber.BigNumber
		retValI, err = hm.Get(i, i)
		if err != nil {
			return retVal, fmt.Errorf(
				"NewDiagonalStatistics: could not get %s[%d][%d]: %q",
				matrixName, i, i, err.Error(),
			)
		}
		retVal.Diagonal[i] = bignumber.NewFromInt64(0).Set(retValI)
		absRetVal := bignumber.NewFromInt64(0).Abs(retValI)
		if hmIsH && absRetVal.Cmp(largestDiagonalElement) > 0 {
			largestDiagonalElement.Set(absRetVal)
		}
		if !hmIsH && absRetVal.Cmp(smallestDiagonalElement) < 0 {
			smallestDiagonalElement.Set(absRetVal)
		}
	}

	// Put the largest-to-last ratio in the struct to be returned, if possible.
	var ratioAsBigNumber *bignumber.BigNumber
	if hmIsH {
		ratioAsBigNumber, err = bignumber.NewFromInt64(0).Quo(
			largestDiagonalElement, retVal.Diagonal[numCols-1],
		)
		if err != nil {
			_, l0 := largestDiagonalElement.String()
			_, r0 := retVal.Diagonal[numCols-1].String()
			return retVal, fmt.Errorf("GetDiagonal: could compute %q/%q: %q", l0, r0, err.Error())
		}
	} else {
		ratioAsBigNumber, err = bignumber.NewFromInt64(0).Quo(
			retVal.Diagonal[numCols-1], smallestDiagonalElement,
		)
		if err != nil {
			_, s0 := smallestDiagonalElement.String()
			_, r0 := retVal.Diagonal[numCols-1].String()
			return retVal, fmt.Errorf("GetDiagonal: could compute %q/%q: %q", r0, s0, err.Error())
		}
	}
	ratioAsFloat := ratioAsBigNumber.AsFloat()
	ratioAsFloat64, _ := ratioAsFloat.Float64()
	if !math.IsInf(ratioAsFloat64, 0) {
		ratioAsFloat64 = math.Abs(ratioAsFloat64)
		retVal.Ratio = &ratioAsFloat64
	}

	// Return the diagonal, and potentially the statistics that go with it
	return retVal, nil
}
