package knownanswertest

// Copyright (c) 2025 Colin McRae

import (
	"os"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/predrag3141/IPSLQ/bignumber"
	"github.com/predrag3141/IPSLQ/pslqops"
)

func TestNewKATLog(t *testing.T) {
	const (
		xLen                            = 55
		relationElementRange            = 5
		randomRelationProbabilityThresh = 0.001
		maxIterations                   = 20000
		bigNumberPrecision              = 1500
	)

	// Initialize PSLQ context
	err := bignumber.Init(bigNumberPrecision)
	pc := NewPSLQContext(xLen, relationElementRange, randomRelationProbabilityThresh)
	require.NotNil(t, pc.InputAsDecimalString)
	require.NotNil(t, pc.InputAsBigInt)

	// Initialize the PSLQ state from the PSLQ context
	var state *pslqops.State
	state, err = pslqops.NewState(pc.InputAsDecimalString)
	require.NoError(t, err)

	// Initialize the KAT logger from the PSLQ context
	var kl *KATLog
	kl, err = NewKATLog(os.TempDir(), xLen, 100, 25)
	require.NoError(t, err)

	// Run PSLQ
	for numIterations := 0; numIterations < maxIterations; numIterations++ {
		var terminated bool
		terminated, err = state.OneIteration(pslqops.NextIntOp)
		require.NoError(t, err)
		err = pc.Update(state, terminated || (numIterations == maxIterations-1))
		require.NoError(t, err)
		err = kl.ReportProgress(pc)
		if terminated {
			break
		}
	}
	err = kl.ReportResults(pc)
}
