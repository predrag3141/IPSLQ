package knownanswertest

// Copyright (c) 2025 Colin McRae

import (
	"encoding/json"
	"fmt"
	"testing"

	"github.com/predrag3141/IPSLQ/bignumber"
	"github.com/predrag3141/IPSLQ/pslqops"

	"github.com/stretchr/testify/require"
)

func TestPSLQContext(t *testing.T) {
	const (
		xLen                            = 20
		relationElementRange            = 5
		randomRelationProbabilityThresh = 0.001
		maxIterations                   = 20000
		bigNumberPrecision              = 1500
		maxTreeDepth                    = 8
	)

	// Initializations
	var pslqState *pslqops.State
	err := bignumber.Init(bigNumberPrecision)
	pslqContext := NewPSLQContext(xLen, maxTreeDepth, relationElementRange, randomRelationProbabilityThresh)
	require.NotNil(t, pslqContext.InputAsDecimalString)
	require.NotNil(t, pslqContext.InputAsBigInt)
	pslqState, err = pslqops.NewState(pslqContext.InputAsDecimalString, maxTreeDepth)
	require.NoError(t, err)

	// Run PSLQ
	for numIterations := 0; numIterations < maxIterations; numIterations++ {
		// Run one iteration of PSLQ
		var terminated bool
		terminated, err = pslqState.OneIteration(pslqops.NextIntOp)
		require.NoError(t, err)
		err = pslqContext.Update(pslqState, terminated || (numIterations == maxIterations-1))
		require.NoError(t, err)

		// Report directly to the terminal. Reporting to files is reserved for the KATLog test
		if numIterations%1000 == 0 {
			fmt.Printf("\n")
		}
		if numIterations%100 == 0 {
			if pslqState.UsingInverse() {
				fmt.Printf(" M")
			} else {
				fmt.Printf(" H")
			}
			if pslqContext.FoundRelation {
				fmt.Printf("! ")
			} else {
				fmt.Printf("? ")
			}
		} else if numIterations%10 == 0 {
			fmt.Printf(".")
		}

		// Handle the termination flag
		if terminated {
			break
		}
	}
	fmt.Printf("\n")

	// Test JSON
	var pslqContextAsJSON []byte
	var copyOfPSLQContext PSLQContext
	pslqContextAsJSON, err = json.Marshal(pslqContext)
	require.NoError(t, err)
	err = json.Unmarshal(pslqContextAsJSON, &copyOfPSLQContext)
	require.NoError(t, err)
	t.Logf("%s\n", string(pslqContextAsJSON))
}
