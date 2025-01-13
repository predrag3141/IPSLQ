package pslqops

// Copyright (c) 2025 Colin McRae

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
)

func TestGetNormalizedX(t *testing.T) {
	// Known input and output from pslq.py
	input := []string{"1", "-3", "-5", "70"}
	expected, err := bigmatrix.NewFromDecimalStringArray(
		[]string{
			"0.014234965584343246", "-0.04270489675302974",
			"-0.07117482792171623", "0.9964475909040272",
		}, 1, 4,
	)
	assert.NoError(t, err)
	rawX, err := getRawX(input, "TestGetNormalizedX")
	assert.NoError(t, err)
	actual, err := getNormalizedX(rawX, "TestGetNormalizedX")
	assert.NoError(t, err)
	equals, err := expected.Equals(actual, bignumber.NewPowerOfTwo(-50))
	assert.True(t, equals)

	// Input cannot be parsed
	input = []string{"0.23402694307324226", "a", "0.7352167040894672"}
	shouldBeNil, err := getRawX(input, "TestGetNormalizedX")
	assert.Error(t, err)
	assert.Nil(t, shouldBeNil)

	// Input contains a zero
	input = []string{"0.23402694307324226", "0", "0.7352167040894672"}
	shouldBeNil, err = getRawX(input, "TestGetNormalizedX")
	assert.Error(t, err)
	assert.Nil(t, shouldBeNil)
}

func TestGetS(t *testing.T) {
	// Known input and output from pslq.py
	input := []string{"0.23402694307324226", "0.6361507588171331", "0.7352167040894672"}
	expected, err := bigmatrix.NewFromDecimalStringArray(
		[]string{"1", "0.972230111607223", "0.7352167040894672"}, 1, 3,
	)
	assert.NoError(t, err)
	actual, err := getSFromStrs(input, "TestGetS")
	assert.NoError(t, err)
	equals, err := expected.Equals(actual, bignumber.NewPowerOfTwo(-50))
	assert.True(t, equals)

	// Input cannot be parsed
	input = []string{"0.23402694307324226", "a", "0.7352167040894672"}
	shouldBeNil, err := getSFromStrs(input, "TestGetS")
	assert.Error(t, err)
	assert.Nil(t, shouldBeNil)

	// Input contains a zero
	input = []string{"0.23402694307324226", "0", "0.7352167040894672"}
	shouldBeNil, err = getSFromStrs(input, "TestGetS")
	assert.Error(t, err)
	assert.Nil(t, shouldBeNil)

	// Input is not a row vector
	shouldBeNil, err = getS(bigmatrix.NewEmpty(3, 1), "TestGetS")
	assert.Error(t, err)
	assert.Nil(t, shouldBeNil)
}

func TestGetH(t *testing.T) {
	// Known input and output from pslq.py
	normalizedX0, err := bigmatrix.NewFromDecimalStringArray(
		[]string{"0.23402694307324226", "0.6361507588171331", "0.7352167040894672"},
		1, 3,
	)
	assert.NoError(t, err)
	s0, err := bigmatrix.NewFromDecimalStringArray(
		[]string{"1", "0.972230111607223", "0.7352167040894672"}, 1, 3,
	)
	assert.NoError(t, err)
	expectedH0, err := bigmatrix.NewFromDecimalStringArray(
		[]string{
			"0.9722301116072231", "0",
			"-0.15312878673710792", "0.7562167590901482",
			"-0.17697509643062187", "-0.6543211851004007",
		},
		3, 2,
	)
	assert.NoError(t, err)
	actualH0, err := getH(normalizedX0, s0, "TestGetH")
	assert.NoError(t, err)
	equals, err := expectedH0.Equals(actualH0, bignumber.NewPowerOfTwo(-50))
	assert.True(t, equals)
	testPropertiesOfH(t, normalizedX0, actualH0)

	// Generate a new H and test its properties
	rawX1, err := getRawX([]string{"1", "2435", "3524", "23452", "674"}, "TestGetH")
	assert.NoError(t, err)
	normalizedX1, err := getNormalizedX(rawX1, "TestGetH")
	assert.NoError(t, err)
	s1, err := getS(normalizedX1, "TestGetH")
	assert.NoError(t, err)
	actualH1, err := getH(normalizedX1, s1, "TestGetH")
	testPropertiesOfH(t, rawX1, actualH1)
}

func testPropertiesOfH(t *testing.T, x *bigmatrix.BigMatrix, h *bigmatrix.BigMatrix) {
	// H-transpose H = I(n-1)
	ht, err := bigmatrix.NewEmpty(0, 0).Transpose(h)
	assert.NoError(t, err)
	shouldBeIdentity, err := bigmatrix.NewEmpty(0, 0).Mul(ht, h)
	assert.NoError(t, err)
	isTheIdentity, err := bigmatrix.NewIdentity(h.NumCols())
	assert.NoError(t, err)
	equals, err := isTheIdentity.Equals(shouldBeIdentity, bignumber.NewPowerOfTwo(-50))
	assert.True(t, equals)

	// x H = 0
	zero := bigmatrix.NewEmpty(1, h.NumCols())
	shouldBeZero, err := bigmatrix.NewEmpty(0, 0).Mul(x, h)
	assert.NoError(t, err)
	equals, err = zero.Equals(shouldBeZero, bignumber.NewPowerOfTwo(-50))
	assert.NoError(t, err)
	assert.True(t, equals)
}

// getSFromStrs creates a 1 x n matrix from input and passes it to
// GetS. It passes back to the caller the same matrix GetS returns.
//
// If input is nil or empty, or if any input element is a malformed
// decimal string, an error is returned. If the call to GetS fails,
// the error GetS reports is returned.
func getSFromStrs(input []string, caller string) (*bigmatrix.BigMatrix, error) {
	caller = fmt.Sprintf("%s-getSFromStrs", caller)
	n := len(input)
	if input == nil || n == 0 {
		return nil, fmt.Errorf("GetSFromStrs: empty input")
	}
	bm, err := bigmatrix.NewFromDecimalStringArray(input, 1, 3)
	if err != nil {
		return nil, fmt.Errorf(
			"%s: error from NewFromDecimalStringArray: %q", caller, err.Error(),
		)
	}
	var retVal *bigmatrix.BigMatrix
	retVal, err = getS(bm, caller)
	if err != nil {
		return nil, err
	}
	return retVal, nil
}
