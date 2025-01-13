package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"

	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
)

func getRawX(input []string, caller string) (*bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getRawX", caller)
	n := len(input)
	if input == nil || n == 0 {
		return nil, fmt.Errorf("%s: empty input", caller)
	}
	retVal := bigmatrix.NewEmpty(1, n)
	for i := 0; i < n; i++ {
		retValI, err := bignumber.NewFromDecimalString(input[i])
		if err != nil {
			return nil, fmt.Errorf(
				"%s error from NewFromDecimalString: input = %q err = %q",
				caller, input[i], err.Error(),
			)
		}
		if retValI.IsZero() {
			return nil, fmt.Errorf("%s: input %d = 0", caller, i)
		}
		err = retVal.Set(0, i, retValI)
		if err != nil {
			return nil, fmt.Errorf(
				"%s: error from Set: i = %d err = %q", caller, i, err.Error(),
			)
		}
	}
	return retVal, nil
}

// getNormalizedX converts input to a 1 x len(input) BigMatrix scaled
// to have Euclidean length 1.
//
// If input is nil or empty, or contains a 0, or contains a number that
// cannot be parsed as a decimal, an error is returned.
func getNormalizedX(rawX *bigmatrix.BigMatrix, caller string) (*bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getNormalizedX", caller)
	numCols := rawX.NumCols()
	sumOfSquares := bignumber.NewFromInt64(0)
	sqrtSumOfSquares := bignumber.NewFromInt64(0)
	oneOverSqrtSumOfSquares := bignumber.NewFromInt64(1)
	retValAsArray := make([]*bignumber.BigNumber, numCols)
	for i := 0; i < numCols; i++ {
		retValI, err := rawX.Get(0, i)
		if err != nil {
			return nil, fmt.Errorf(
				"%s: could not get rawX[0][%d]: %q", caller, i, err.Error(),
			)
		}
		retValAsArray[i] = bignumber.NewFromInt64(0).Set(retValI)
		sumOfSquares.MulAdd(retValI, retValI)
	}
	_, err := sqrtSumOfSquares.Sqrt(sumOfSquares)
	if err != nil {
		return nil, fmt.Errorf(
			"%s: error taking the square root of the sum of squares: %q",
			caller, err.Error(),
		)
	}
	_, err = oneOverSqrtSumOfSquares.Quo(oneOverSqrtSumOfSquares, sqrtSumOfSquares)
	if err != nil {
		return nil, fmt.Errorf("GetNormalizedX: error dividing by sum of squares: %q", err.Error())
	}

	// retValAsArray contains unscaled deep copies of the elements of rawX.
	// oneOverSumOfSquares is scale factor that normalizes rawX to Euclidean length 1.
	for i := 0; i < numCols; i++ {
		retValAsArray[i].Mul(retValAsArray[i], oneOverSqrtSumOfSquares)
		retValAsArray[i].Normalize(0) // not the normalization to Euclidean length 1
	}
	retVal := bigmatrix.NewEmpty(1, numCols)
	for i := 0; i < numCols; i++ {
		err = retVal.Set(0, i, retValAsArray[i])
		if err != nil {
			return nil, fmt.Errorf(
				"%s: could not set retVal[0][%d]: %q", caller, i, err.Error(),
			)
		}
	}
	return retVal, nil
}

// getS gets, given 1 x n matrix x, s as defined in page 4 of the
// original PSLQ paper: "Define the partial sums of squares, sj , for x ...".
// The returned value is a 1 x n matrix. GetS does not modify x.
//
// If bm is not a 1 x n matrix, or contains 0s, an error is returned.
//
// Python code:
//
// return [
//
//	sqrt(sum([input[j] * input[j] for j in range(k, n_in)]))
//	for k in range(len(input))
//
// ]
func getS(x *bigmatrix.BigMatrix, caller string) (*bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getS", caller)
	numRows, n := x.Dimensions()
	if numRows != 1 {
		return nil, fmt.Errorf("%s: numRows = %d, but must be 1", caller, numRows)
	}
	retval := bigmatrix.NewEmpty(1, n)
	lastSquared := bignumber.NewFromInt64(0)
	for i := 0; i < n; i++ {
		tmp, err := x.Get(0, i)
		if err != nil {
			return nil, fmt.Errorf("%s: error from x.Get: %q", caller, err.Error())
		}
		retvalI := bignumber.NewFromBigNumber(tmp)
		if retvalI.IsZero() {
			return nil, fmt.Errorf("%s: input %d is zero", caller, i)
		}
		if i == n-1 {
			lastSquared.Mul(retvalI, retvalI)
			retvalI.Abs(retvalI)
		} else {
			retvalI.Mul(retvalI, retvalI)
		}
		err = retval.Set(0, i, retvalI)
		if err != nil {
			return nil, fmt.Errorf("GetS: error setting element %d: %q", i, err.Error())
		}
	}

	// retval contains squares of original elements passed in, except for
	// retval[n-1], which contains the absolute value of the last element
	// passed in.
	for i := n - 2; i >= 0; i-- {
		retvalI, err := retval.Get(0, i)
		if err != nil {
			return nil, fmt.Errorf("%s: could not retrieve element %d: %q", caller, i, err.Error())
		}
		if i == n-2 {
			retvalI.Add(retvalI, lastSquared)
		} else {
			tmp, err := retval.Get(0, i+1)
			if err != nil {
				return nil, fmt.Errorf("%s: could not retrieve element %d: %q", caller, i+1, err.Error())
			}
			retvalIPlusOne := bignumber.NewFromBigNumber(tmp)
			err = retval.Set(0, i, retvalI.Add(retvalI, retvalIPlusOne))
			if err != nil {
				return nil, fmt.Errorf("%s: error setting element %d: %q", caller, i, err.Error())
			}
		}
	}

	// retval now contains partial sums of squares of original elements passed
	// in, except for retval[n-1], which still contains the absolute value of the
	// last element passed in.
	for i := 0; i < n-1; i++ {
		retvalI, err := retval.Get(0, i)
		if err != nil {
			return nil, fmt.Errorf("%s: could not retrieve element %d: %q", caller, i, err.Error())
		}
		_, err = retvalI.Sqrt(retvalI)
		if err != nil {
			return nil, fmt.Errorf(
				"%s: could not take the square root of element %d: %q", caller, i, err.Error(),
			)
		}
	}

	// retval now contains square roots of partial sums of squares
	return retval, nil
}

//  Page 4: "Define the lower trapezoidal n x (n − 1) matrix H(x)"
//
//  Left of the diagonal:  -x[i]x[j]/s[j]s[j+1] (rows 1, ... n - 1)
//  On the diagonal:       s[i+1]/s[i]          (rows 0, ... n - 2)
//  Right of the diagonal: 0                    (rows 0, ... n - 3)
//
// Python code:
//
//  H = []
//  for i in range(n_in):
//    row = []       # Overwritten by entries to the left of the diagonal if i > 0
//    if 1 <= i:     # Write entries to the left of the diagonal
//      row = [-x_in[i] * x_in[j] / (s_in[j] * s_in[j+1]) for j in range(i)]
//    if i <= n_in - 2: # Write an entry on the diagonal
//      row.append(s_in[i + 1] / s_in[i])
//    if i <= n_in - 3: # Write 0s to the right of the diagonal
//      for j in range(i + 1, n_in - 1):
//      row.append(0)
//      H.append(row)
//  return H

// getH creates the initial H matrix defined on page 4 of the original PSLQ
// paper. Inputs x and s are 1 x n matrices.
func getH(x *bigmatrix.BigMatrix, s *bigmatrix.BigMatrix, caller string) (*bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getH", caller)
	numRows, n := x.Dimensions()
	if numRows != 1 {
		return nil, fmt.Errorf("%s: input x has %d > 1 rows", caller, numRows)
	}
	numRows, sn := s.Dimensions()
	if numRows != 1 {
		return nil, fmt.Errorf("%s: input x has %d > 1 rows", caller, numRows)
	}
	if sn != n {
		return nil, fmt.Errorf("%s: inputs x and s have unequal lengths %d and %d", caller, n, sn)
	}
	retval := bigmatrix.NewEmpty(n, n-1)
	zero := bignumber.NewFromInt64(0)
	minusXiXj := bignumber.NewFromInt64(0)
	sjSjplus1 := bignumber.NewFromInt64(0)
	hij := bignumber.NewFromInt64(0)
	for i := 0; i < n; i++ {
		for j := 0; j < n-1; j++ {
			if i < j {
				// Right of the diagonal, H[i][j] = 0
				// (rows 0, ... n - 3; on other rows j <= i)
				hij.Set(zero)
			} else {
				// j <= i; s[j] and s[j+1] are always needed
				sj, err := s.Get(0, j)
				if err != nil {
					return nil, fmt.Errorf(
						"GetH: could not get element %d from s: %q", j, err.Error(),
					)
				}
				sJplus1, err := s.Get(0, j+1)
				if err != nil {
					return nil, fmt.Errorf(
						"GetH: could not get element %d from s: %q", j+1, err.Error(),
					)
				}
				if i == j {
					// On the diagonal, H[j][j] = s[j+1]/s[j]
					// (rows 0, ... n - 2; on row n-1, j < i)
					_, err := hij.Quo(sJplus1, sj)
					if err != nil {
						return nil, fmt.Errorf(
							"GetH: could not divide by s[%d]: %q", j+1, err.Error(),
						)
					}
				} else {
					// Left of the diagonal (j < i), H[i][j] = -x[i]x[j]/s[j]s[j+1]
					// (rows 1, ... n - 1)
					xi, err := x.Get(0, i)
					if err != nil {
						return nil, fmt.Errorf(
							"GetH: could not get element %d from x: %q", i, err.Error(),
						)
					}
					xj, err := x.Get(0, j)
					if err != nil {
						return nil, fmt.Errorf(
							"GetH: could not get element %d from x: %q", j, err.Error(),
						)
					}
					minusXiXj.Mul(xi, xj)
					minusXiXj.Sub(zero, minusXiXj)
					sjSjplus1.Mul(sj, sJplus1)
					_, err = hij.Quo(minusXiXj, sjSjplus1)
					if err != nil {
						return nil, fmt.Errorf(
							"GetH: could not divide by s[%d]s[%d]: %q", j, j+1, err.Error(),
						)
					}
				}
			}

			// hij has been set in one of the if-blocks.
			err := retval.Set(i, j, hij)
			if err != nil {
				return nil, fmt.Errorf("GetH: could not set H[%d][%d]: %q", i, j, err.Error())
			}
		} // iterate over columns j
	} // iterate over rows i
	return retval, nil
}
