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
		maxIterations                   = 100000
		bigNumberPrecision              = 1500
		log2EColumnsTested              = 8
		reportingPeriodBeforeInverting  = 100
		reportingPeriodAfterInverting   = 1
	)

	// Initialize PSLQ context
	err := bignumber.Init(bigNumberPrecision)
	pc := NewPSLQContext(xLen, relationElementRange, randomRelationProbabilityThresh)
	require.NotNil(t, pc.InputAsDecimalString)
	require.NotNil(t, pc.InputAsBigInt)

	// Initialize the PSLQ state from the PSLQ context
	var state *pslqops.State
	state, err = pslqops.NewState(pc.InputAsDecimalString, log2EColumnsTested)
	require.NoError(t, err)

	// Initialize the KAT logger from the PSLQ context
	var kl *KATLog
	kl, err = NewKATLog(os.TempDir(), xLen, reportingPeriodBeforeInverting, reportingPeriodAfterInverting)
	require.NoError(t, err)

	// Run PSLQ
	for numIterations := 0; numIterations < maxIterations; numIterations++ {
		// Update every iteration; report progress conditionally, depending
		// on the iteration number, kl.reportingPeriodBeforeInverting and
		// kl.reportingPeriodAfterInverting
		var terminated bool
		terminated, err = state.OneIteration(pslqops.NextIntOp)
		require.NoError(t, err)
		err = pc.Update(state, terminated || (numIterations == maxIterations-1))
		require.NoError(t, err)
		err = kl.ReportProgress(pc, numIterations == 0)
		if terminated {
			break
		}
	}
	err = kl.ReportResults(pc)
}
