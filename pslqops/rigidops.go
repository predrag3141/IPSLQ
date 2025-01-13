package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"

	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
)

// givensRotationOnH right-multiplies H by an orthogonal matrix, G ,that is an identity matrix
// except for a 2x2 sub-matrix with entries c, -s, s, c at coordinates (j0,j0), (j0,j1),
// (j1,j0) and (j1,j1), respectively. c and s are defined so that (HG)[j0][j1] = ac - sb = 0,
// where a = H[j0][j0] and b = H[j0][j1] are the entries in the two target columns and the
// one target row, j0. The entries that contribute to ac - sb = 0 are in the same row since
// H is on the left side of the matrix multiplication.
//
// Since G is orthogonal, c^2 + s^2 = 1. This and the condition that (HG)[j0][j1] = 0 are
// satisfied by
//
// - c <- a/r
//
// - s <- b/r
//
// where r = sqrt(a^2 + b^2). Orthogonality of G follows from its form. The fact that
// (HG)[j0][j1] = 0 follows from:
//
// (HG)[j0][j1] = (a)(-s)+(b)(c) = (a)(-b/r)+(b)(a/r) = (-ab/r)+(ab/r) = 0.
func givensRotationOnH(h *bigmatrix.BigMatrix, j0, j1 int, caller string) error {
	// Initializations
	caller = fmt.Sprintf("%s-givensRotation", caller)
	if (j0 < 0) || (j1 <= j0) || (h.NumCols() <= j1) {
		return fmt.Errorf(
			"%s: parameters [j0, j1] = [%d, %d] violate 0 <= j0 < j1 <= %d",
			caller, j0, j1, h.NumCols()-1,
		)
	}

	// a = H[j0][j0] and b = H[j0][j1]
	a, err := h.Get(j0, j0)
	if err != nil {
		return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j0, j0, err.Error())
	}
	var b, r, oneOverR *bignumber.BigNumber
	b, err = h.Get(j0, j1)
	if err != nil {
		return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j0, j1, err.Error())
	}

	// c <- a/r and s <- b/r
	aSq := bignumber.NewFromInt64(0).Mul(a, a)
	bSq := bignumber.NewFromInt64(0).Mul(b, b)
	rSq := bignumber.NewFromInt64(0).Add(aSq, bSq)
	r, err = bignumber.NewFromInt64(0).Sqrt(rSq)
	if err != nil {
		_, rSqAsStr := rSq.String()
		return fmt.Errorf("%s: could not compute Sqrt(%s): %q", caller, rSqAsStr, err.Error())
	}
	oneOverR, err = bignumber.NewFromInt64(0).Quo(bignumber.NewFromInt64(1), r)
	if err != nil {
		_, rSqAsStr := rSq.String()
		return fmt.Errorf("%s: could not compute 1/%s: %q", caller, rSqAsStr, err.Error())
	}
	c := bignumber.NewFromInt64(0).Mul(a, oneOverR)
	s := bignumber.NewFromInt64(0).Mul(b, oneOverR)
	ms := bignumber.NewFromInt64(0).Sub(bignumber.NewFromInt64(0), s)

	// Column j0 of H <- (column j0 of H)( c) + (column j1 of H)(s)
	// Column j1 of H <- (column j0 of H)(-s) + (column j1 of H)(c)
	// For rows 0, ..., j0-1, there are zeroes in both columns j0 and j1.
	for k := 0; k < h.NumRows(); k++ {
		var hkj0, hkj1 *bignumber.BigNumber
		hkj0, err = h.Get(k, j0)
		if err != nil {
			return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, k, j0, err.Error())
		}
		hkj1, err = h.Get(k, j1)
		if err != nil {
			return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, k, j1, err.Error())
		}
		if hkj0.IsSmall() && hkj1.IsSmall() {
			continue
		}
		hkj0c := bignumber.NewFromInt64(0).Mul(hkj0, c)
		hkj0ms := bignumber.NewFromInt64(0).Mul(hkj0, ms)
		hkj1c := bignumber.NewFromInt64(0).Mul(hkj1, c)
		hkj1s := bignumber.NewFromInt64(0).Mul(hkj1, s)
		err = h.Set(k, j0, bignumber.NewFromInt64(0).Add(hkj0c, hkj1s))
		if err != nil {
			return fmt.Errorf("%s: could not set H[%d][%d]: %q", caller, k, j0, err.Error())
		}
		err = h.Set(k, j1, bignumber.NewFromInt64(0).Add(hkj0ms, hkj1c))
		if err != nil {
			return fmt.Errorf("%s: could not set H[%d][%d]: %q", caller, k, j1, err.Error())
		}
	}
	return nil
}

// givensRotationOnM left-multiplies M by an orthogonal matrix, G, that is an identity matrix
// except for a 2x2 sub-matrix with entries c, -s, s, c at coordinates (j0,j0), (j0,j1),
// (j1,j0) and (j1,j1), respectively. c and s are defined so that (GM)[j0][j1] = ca - sb = 0,
// where a = M[j0][j1] and b = M[j1][j1] are the entries in the two target rows and the
// one target column, j1. The entries that contribute to ca - sb = 0 are in the same column
// since M is on the right side of the matrix multiplication.
//
// Since G is orthogonal, c^2 + s^2 = 1. That and the condition that (GM)[j0][j1] = 0 are
// satisfied by
//
// - c <- b/r
//
// - s <- a/r
//
// where r = sqrt(a^2 + b^2). Orthogonality of G follows from its form. The fact that
// (GM)[j0][j1] = 0 follows from:
//
// (GM)[j0][j1] = (c)(a) + (-s)(b) = (b/r)(a)+(-a/r)(b) = (ab/r)+(-ab/r) = 0.
func givensRotationOnM(m *bigmatrix.BigMatrix, j0, j1 int, caller string) error {
	// Initializations
	caller = fmt.Sprintf("%s-givensRotation", caller)
	numCols := m.NumCols()
	if (j0 < 0) || (j1 <= j0) || (numCols <= j1) {
		return fmt.Errorf(
			"%s: parameters [j0, j1] = [%d, %d] violate 0 <= j0 < j1 <= %d",
			caller, j0, j1, numCols-1,
		)
	}

	// a = M[j0][j1] and b = M[j1][j1]
	a, err := m.Get(j0, j1)
	if err != nil {
		return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j0, j0, err.Error())
	}
	var b, r, oneOverR *bignumber.BigNumber
	b, err = m.Get(j1, j1)
	if err != nil {
		return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j1, j1, err.Error())
	}

	// c <- b/r and s <- a/r
	aSq := bignumber.NewFromInt64(0).Mul(a, a)
	bSq := bignumber.NewFromInt64(0).Mul(b, b)
	rSq := bignumber.NewFromInt64(0).Add(aSq, bSq)
	r, err = bignumber.NewFromInt64(0).Sqrt(rSq)
	if err != nil {
		_, rSqAsStr := rSq.String()
		return fmt.Errorf("%s: could not compute Sqrt(%s): %q", caller, rSqAsStr, err.Error())
	}
	oneOverR, err = bignumber.NewFromInt64(0).Quo(bignumber.NewFromInt64(1), r)
	if err != nil {
		_, rSqAsStr := rSq.String()
		return fmt.Errorf("%s: could not compute 1/%s: %q", caller, rSqAsStr, err.Error())
	}
	c := bignumber.NewFromInt64(0).Mul(b, oneOverR)
	s := bignumber.NewFromInt64(0).Mul(a, oneOverR)
	ms := bignumber.NewFromInt64(0).Sub(bignumber.NewFromInt64(0), s)

	// Row j0 of M <- (c)(row j0 of M) + (-s)(row j1 of M)
	// Row j1 of M <- (s)(row j0 of M) + ( c)(row j1 of M)
	// For columns j1+1, j1+2, ... there are zeroes in both rows j0 and j1.
	for k := 0; k < numCols; k++ {
		var mj0k, mj1k *bignumber.BigNumber
		mj0k, err = m.Get(j0, k)
		if err != nil {
			return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j0, k, err.Error())
		}
		mj1k, err = m.Get(j1, k)
		if err != nil {
			return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, j1, k, err.Error())
		}
		if mj0k.IsSmall() && mj1k.IsSmall() {
			continue
		}
		cmj0k := bignumber.NewFromInt64(0).Mul(c, mj0k)
		smj0k := bignumber.NewFromInt64(0).Mul(s, mj0k)
		cmj1k := bignumber.NewFromInt64(0).Mul(c, mj1k)
		msmj1k := bignumber.NewFromInt64(0).Mul(ms, mj1k)
		err = m.Set(j0, k, bignumber.NewFromInt64(0).Add(cmj0k, msmj1k))
		if err != nil {
			return fmt.Errorf("%s: could not set H[%d][%d]: %q", caller, k, j0, err.Error())
		}
		err = m.Set(j1, k, bignumber.NewFromInt64(0).Add(smj0k, cmj1k))
		if err != nil {
			return fmt.Errorf("%s: could not set H[%d][%d]: %q", caller, k, j1, err.Error())
		}
	}
	return nil
}

// removeCornerOfH uses Givens rotations to implicitly right-multiply H by an orthogonal
// matrix, Q (never created explicitly), for which HQ has zeroes above the diagonal.
func removeCornerOfH(h *bigmatrix.BigMatrix, indices []int, caller string) error {
	caller = fmt.Sprintf("%s-removeCornerOfH", caller)
	j0 := indices[0]
	j1 := indices[len(indices)-1]
	for i := j0; i <= j1; i++ {
		for j := j1; j > i; j-- {
			hij, err := h.Get(i, j)
			if err != nil {
				return fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, i, j, err.Error())
			}
			if hij.IsZero() {
				continue
			}
			err = givensRotationOnH(h, i, j, caller)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

// removeCornerOfM uses Givens rotations to implicitly left-multiply M by an orthogonal matrix,
// Q (never created explicitly), for which QM has zeroes above the diagonal.
func removeCornerOfM(m *bigmatrix.BigMatrix, indices []int, caller string) error {
	caller = fmt.Sprintf("%s-removeCornerOfM", caller)
	j0 := indices[0]
	j1 := indices[len(indices)-1]
	for j := j1; 0 <= j; j-- {
		for i := j0; i < j; i++ {
			mij, err := m.Get(i, j)
			if err != nil {
				return fmt.Errorf("%s: could not get M[%d][%d]: %q", caller, i, j, err.Error())
			}
			if mij.IsZero() {
				continue
			}
			err = givensRotationOnM(m, i, j, caller)
			if err != nil {
				return err
			}
		}
	}
	return nil
}
