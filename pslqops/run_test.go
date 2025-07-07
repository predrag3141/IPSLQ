package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"math/rand"
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/predrag3141/IPSLQ/bignumber"
)

type expectedState struct {
	rawX        []*bignumber.BigNumber
	normalizedX []*bignumber.BigNumber
	s           []*bignumber.BigNumber
	h           []*bignumber.BigNumber
}

func TestMain(m *testing.M) {
	err := bignumber.Init(binaryPrecision)
	if err != nil {
		fmt.Printf("Invalid input to Init: %q", err.Error())
		return
	}
	code := m.Run()
	os.Exit(code)
}

func TestNew(t *testing.T) {
	rawStrs00 := []string{"245.43", "3452.98", "-943.54", "-89.876234"}
	expected00, err := newExpectedState(rawStrs00, "TestNew")
	require.NoError(t, err)
	actual00, err := NewState(rawStrs00, 2)
	expected00.testEquality(t, actual00)

	var actual01 *State
	rawStrs01 := []string{"761.98", "2.8952", "-61.941", "-900.03", "241.3", "-53.473", "73.832", "-51.347", "91.827", "-1.5017"}
	expected01, err := newExpectedState(rawStrs01, "TestNew")
	require.NoError(t, err)
	actual01, err = NewState(rawStrs01, 2)
	require.NoError(t, err)
	expected01.testEquality(t, actual01)
}

func TestState_OneIteration(t *testing.T) {
	const (
		numRows        = 20
		maxIterations  = 1000
		digitsPerEntry = 10
		minSeed        = 13427
		numTests       = 1
	)

	rand.Seed(minSeed)
	for i := 0; i < numTests; i++ {
		input := make([]string, numRows)
		for j := 0; j < numRows; j++ {
			input[j] = getRandomDecimalStr(t, digitsPerEntry)
		}
		state, err := NewState(input, 2)
		var roundOffErrorAsString string
		numIterations := 0 // needed outside the loop below
		for ; numIterations < maxIterations; numIterations++ {
			var terminated bool
			terminated, err = state.OneIteration(NextIntOp, true)
			require.NoError(t, err)

			// Test round-off
			roundOffError := state.GetObservedRoundOffError()
			_, roundOffErrorAsString = roundOffError.String()
			require.Truef(
				t, roundOffError.IsSmall(), "round-off error is not small: %q", roundOffErrorAsString,
			)
			if terminated {
				break
			}
		}
		xB := state.GetXB()
		t.Logf(
			"\nAfter %d iterations:\n"+
				"- %d all-zero rows were calculated\n"+
				"- round-off error = %s\n"+
				"- xB =\n%v\n",
			numIterations, state.GetAllZeroRowsCalculated(), roundOffErrorAsString, xB,
		)
	}
}

func getRandomDecimalStr(t *testing.T, numDigits int) string {
	sgn := 1 - 2*rand.Intn(2)
	require.Truef(t, sgn == -1 || sgn == 1, "sgn = %d but must be -1 or 1", sgn)
	retVal := ""
	if sgn == -1 {
		retVal = "-"
	}
	decimalPos := rand.Intn(numDigits)
	require.Truef(
		t, (0 <= decimalPos) && (decimalPos < numDigits),
		"decimal position is not in {0,1,...,%d}", numDigits-1,
	)
	for i := 0; i < numDigits; i++ {
		digit := rand.Intn(10)
		require.Truef(t, (0 <= digit) && (digit < 10), "digit is not in {0,1,...,9}")
		if (digit == 0) && (i == 0) {
			digit = 1
		}
		if i == decimalPos {
			if i == 0 {
				retVal = retVal + "0."
			} else {
				retVal = retVal + "."
			}
		}
		retVal = retVal + fmt.Sprintf("%d", digit)
	}
	return retVal
}

func newExpectedState(rawStrs []string, caller string) (*expectedState, error) {
	caller = fmt.Sprintf("%s-newExpectedState", caller)
	var retVal expectedState
	err := retVal.setRawX(rawStrs, caller)
	if err != nil {
		return nil, err
	}
	err = retVal.setNormalizedX(caller)
	if err != nil {
		return nil, err
	}
	err = retVal.setS(caller)
	if err != nil {
		return nil, err
	}
	err = retVal.setH(caller)
	if err != nil {
		return nil, err
	}
	return &retVal, nil
}

func (es *expectedState) setRawX(rawStrs []string, caller string) error {
	caller = fmt.Sprintf("%s-setRawX", caller)
	inputLen := len(rawStrs)
	es.rawX = make([]*bignumber.BigNumber, inputLen)
	var err error
	for i := 0; i < inputLen; i++ {
		es.rawX[i], err = bignumber.NewFromDecimalString(rawStrs[i])
		if err != nil {
			return fmt.Errorf(
				"%s: could not parse rawStrs[%d] = %q as a decimal number", caller, i, rawStrs[i],
			)
		}
	}
	return nil
}

func (es *expectedState) setNormalizedX(caller string) error {
	caller = fmt.Sprintf("%s-setNormalizedX", caller)
	inputLen := len(es.rawX)
	normSq := bignumber.NewFromInt64(0)
	for i := 0; i < inputLen; i++ {
		normSq.MulAdd(es.rawX[i], es.rawX[i])
	}
	norm, err := bignumber.NewFromInt64(0).Sqrt(normSq)
	if err != nil {
		_, normSqAsStr := normSq.String()
		return fmt.Errorf("%s: could not take square root of %s: %q",
			caller, normSqAsStr, err.Error())
	}
	es.normalizedX = make([]*bignumber.BigNumber, inputLen)
	for i := 0; i < inputLen; i++ {
		es.normalizedX[i], err = bignumber.NewFromInt64(0).Quo(es.rawX[i], norm)
	}
	return nil
}

func (es *expectedState) setS(caller string) error {
	caller = fmt.Sprintf("%s-setS", caller)
	inputLen := len(es.normalizedX)
	es.s = make([]*bignumber.BigNumber, inputLen)
	for j := 0; j < inputLen; j++ {
		var err error
		sJSq := bignumber.NewFromInt64(0)
		for k := j; k < inputLen; k++ {
			sJSq.MulAdd(es.normalizedX[k], es.normalizedX[k])
		}
		es.s[j], err = bignumber.NewFromInt64(0).Sqrt(sJSq)
		if err != nil {
			_, sJSqAsStr := sJSq.String()
			return fmt.Errorf(
				"%s: could not take the square root of %s: %q",
				caller, sJSqAsStr, err.Error(),
			)
		}
	}
	return nil
}

func (es *expectedState) setH(caller string) error {
	caller = fmt.Sprintf("%s-setH", caller)
	numRows := len(es.rawX)
	numCols := numRows - 1
	zero := bignumber.NewFromInt64(0)
	es.h = make([]*bignumber.BigNumber, numRows*numCols)
	for i := 0; i < numRows; i++ {
		// Below the diagonal
		for j := 0; j < i; j++ {
			var err error
			xIxJ := bignumber.NewPowerOfTwo(0).Mul(es.normalizedX[i], es.normalizedX[j])
			minusxIxJ := bignumber.NewFromInt64(0).Sub(zero, xIxJ)
			sJsJplus1 := bignumber.NewFromInt64(0).Mul(es.s[j], es.s[j+1])
			es.h[i*numCols+j], err = bignumber.NewFromInt64(0).Quo(minusxIxJ, sJsJplus1)
			if err != nil {
				_, sJsJplus1AsStr := sJsJplus1.String()
				return fmt.Errorf(
					"%s: could not divide by %s: %q", caller, sJsJplus1AsStr, err.Error(),
				)
			}
		}

		// On the diagonal
		if i < numCols {
			var err error
			es.h[i*numCols+i], err = bignumber.NewFromInt64(0).Quo(es.s[i+1], es.s[i])
			if err != nil {
				_, siAsStr := es.s[i].String()
				return fmt.Errorf(
					"%s: could not divide by s[%d] = %s: %q", caller, i, siAsStr, err.Error(),
				)
			}
		}

		// Above the diagonal
		for j := i + 1; j < numCols; j++ {
			es.h[i*numCols+j] = bignumber.NewFromInt64(0)
		}
	}
	return nil
}

func (es *expectedState) testEquality(t *testing.T, actualState *State) {
	// Initializations
	tolerance := bignumber.NewPowerOfTwo(-bigNumberBitTolerance)

	// Compare expected to actual raw X
	for i := 0; i < actualState.rawX.NumCols(); i++ {
		actualRawXI, err := actualState.rawX.Get(0, i)
		require.NoError(t, err)
		eq := es.rawX[i].Equals(actualRawXI, tolerance)
		_, expectedRawXiAsStr := es.rawX[i].String()
		_, actualRawXiAsStr := actualRawXI.String()
		require.Truef(
			t, eq,
			"expectedState.rawX[%d] = %q != %q = actualState.rawX[0][%d]",
			i, expectedRawXiAsStr, actualRawXiAsStr, i,
		)
	}

	// Compare expected to actual H
	numRows := len(es.rawX)
	numCols := numRows - 1
	for i := 0; i < numRows; i++ {
		for j := 0; j < numCols; j++ {
			actualHij, err := actualState.h.Get(i, j)
			require.NoError(t, err)
			eq := actualHij.Equals(es.h[i*numCols+j], tolerance)
			_, expectedHijAsStr := es.h[i*numCols+j].String()
			_, actualHijAsStr := actualHij.String()
			require.Truef(
				t, eq, "expected vs. actual H[%d][%d] are %s != %s",
				i, j, expectedHijAsStr, actualHijAsStr,
			)
		}
	}
}
