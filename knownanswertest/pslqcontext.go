package knownanswertest

// Copyright (c) 2025 Colin McRae

import (
	cr "crypto/rand"
	"fmt"
	"github.com/predrag3141/IPSLQ/pslqops"
	"math"
	"math/big"
	"math/rand"
	"time"
)

// The input to PSLQ will be a xLen-long vector with
// - Entries from the uniform distribution on [-maxX/2,maxX/2].
// - A known, "causal" relation (variable name "relation") with entries from
//   the uniform distribution on [-maxRelationElement/2,maxRelationElement/2]
//
// The question is what maxX needs to be to pose a reasonable challenge to PSLQ.
// A reasonable challenge is for randomRelationProbabilityThresh to be the chance
// that a random relation exists with Euclidean norm less than that of the causal
// relation, or with each entry in [-maxRelationElement/2,maxRelationElement/2].
//
// Figuring out what maxX needs to be requires some hand-waving. The idea is to
// break the computation of <r,x> for a putative r into two parts: the first
// xLen-1 coordinates and the last coordinate. Let
//
// - x1 be the first xLen-1 elements of the PSLQ input, x
// - x2 be the last element of x
// - r1 be the first xLen-1 elements of a potential relation, r, of x
// - r2 be the last element of r
//
// Suppose a random r1 is chosen, and r2 is chosen to make <r,x> as close
// as possible to 0 but with the stipulation that |r2| < maxRelationElement/2.
// The two criteria for such a choice of r2 to yield a relation of x are:
// - <r1,x1> = 0 mod x2
// - |<r1,x1> / x2| < maxRelationElement/2
//
// Assuming these criteria have independent probabilities, their combined probability
// can be estimated by assuming (this is where the hand-waving begins) that
// - |x2| = maxX/4 (since 0 <= |x2| < maxX/2), and
// - |<r1,x1>|^2 is uniform on [0,sqrt(xLen)(maxX/2)] (it is actually chi-square)
//
// The probability that <r1,x1> = 0 mod x2 is
//
// 1/|x2| = 1/maxX/4 = 4/maxX
//
// The probability that |<r1,x1> / x2| < maxRelationElement/2 is
//
// (maxRelationElement/2) / (sqrt(xLen)(maxX/2) / (maxX/4))
//   = (maxRelationElement/2) / (2 sqrt(xLen))
//   = maxRelationElement / (4 sqrt(xLen))
//
// So the combined probability of any given choice of r1 succeeding is
//
// [4/maxX] [maxRelationElement / (4 sqrt(xLen))] = maxRelationElement / (sqrt(xLen) maxX)
//
// The expected number, lambda, of successes over all choices of r1 is
// maxRelationElement^(xLen-1) times that. So
//
// lambda = maxRelationElement^xLen / (sqrt(xLen) maxX)
//
// maxX is to be set so the Poisson probability of 0 successes at random is
// randomRelationProbabilityThresh. The Poisson probability is very close to lambda
// when lambda is small, so set lambda = randomRelationProbabilityThresh:
//
// maxRelationElement^xLen / (sqrt(xLen) maxX) = randomRelationProbabilityThresh
//
// Now multiply by maxX and divide by randomRelationProbabilityThresh to isolate maxX:
//
// maxX = maxRelationElement^xLen / (sqrt(xLen) randomRelationProbabilityThresh)
//
// For simplicity, ignore the small factor of 1/sqrt(xLen);
//
// maxX = maxRelationElement^xLen / randomRelationProbabilityThresh
//
// This is the volume of an xLen-dimensional cube, times 1/randomRelationProbabilityThresh.
// A similar calculation based on the volume of the sphere of possible solutions smaller
// than that of the causal relation is also done. The larger of the two maxX values is
// used.

type PSLQContext struct {
	// Computed before running PSLQ
	InputAsBigInt           []big.Int `json:"input_as_big_int"`
	InputAsDecimalString    []string  `json:"input_as_decimal_string"`
	Relation                []int64   `json:"relation"`
	RelationNorm            float64   `json:"relation_norm"`
	MaxXBasedOnCubeVolume   float64   `json:"max_x_based_on_cube_volume"`
	MaxXBasedOnSphereVolume float64   `json:"max_x_based_on_sphere_volume"`

	// Computed by PSLQ or after running PSLQ
	Solutions                       [][]int64 `json:"solutions"`
	SolutionCount                   int       `json:"solution_count"`
	FoundRelation                   bool      `json:"found_relation"`
	IterationsBeforeInverting       int       `json:"iterations_before_inverting"`
	IterationsAfterInverting        int       `json:"iterations_after_inverting"`
	IterationsBeforeFindingRelation int       `json:"iterations_before_finding_relation"`
	TotalIterations                 int       `json:"total_iterations"`
	ReductionsBeforeInverting       int       `json:"reductions_before_inverting"`
	ReductionsAfterInverting        int       `json:"reductions_after_inverting"`
}

// NewPSLQContext returns input to PSLQ of length xLen with a known solution, m, that PSQL is
// challenged to find. m contains entries within a range of relationElementRange possible
// values, centered at 0. The entries in the PSLQ input this function returns are intended
// to have a random solution not equal to m, but with norm less than |m|, with probability
// randomRelationProbabilityThresh. See the file-level comments for details.
func NewPSLQContext(xLen, relationElementRange int, randomRelationProbabilityThresh float64) *PSLQContext {
	rand.Seed(time.Now().Unix() % 1000)
	relation, relationNorm := getCausalRelation(xLen, relationElementRange)
	maxXBasedOnCubeVolume := math.Pow(float64(relationElementRange), float64(xLen)) / randomRelationProbabilityThresh
	maxXBasedOnSphereVolume := sphereVolume(relationNorm, xLen) / randomRelationProbabilityThresh
	var log2MaxX int64
	if maxXBasedOnSphereVolume > maxXBasedOnCubeVolume {
		log2MaxX = int64(1.0 + math.Log2(maxXBasedOnSphereVolume))
	} else {
		log2MaxX = int64(1.0 + math.Log2(maxXBasedOnCubeVolume))
	}
	maxXAsBigInt := big.NewInt(0).Exp(big.NewInt(2), big.NewInt(log2MaxX), nil)
	inputAsBigInt, inputAsDecimalString := getX(relation, maxXAsBigInt)
	return &PSLQContext{
		InputAsBigInt:                   inputAsBigInt,
		InputAsDecimalString:            inputAsDecimalString,
		Relation:                        relation,
		RelationNorm:                    relationNorm,
		MaxXBasedOnCubeVolume:           maxXBasedOnCubeVolume,
		MaxXBasedOnSphereVolume:         maxXBasedOnSphereVolume,
		FoundRelation:                   false,
		IterationsBeforeFindingRelation: 0,
		IterationsAfterInverting:        0,
		IterationsBeforeInverting:       0,
		ReductionsBeforeInverting:       0,
		ReductionsAfterInverting:        0,
	}
}

// Update  updates flags and counts, and populates pc with the solutions from the
// PSLQ state. The solution updates only occur if setSolutions is true, though a check for
// finding pc.Relation is always done.
func (pc *PSLQContext) Update(state *pslqops.State, setSolutions bool) error {
	// Set flags and counts unrelated to solutions
	pc.IterationsBeforeInverting, pc.IterationsAfterInverting = state.IterationCounts()
	pc.ReductionsBeforeInverting, pc.ReductionsAfterInverting = state.ReductionCounts()
	pc.TotalIterations = pc.IterationsBeforeInverting + pc.IterationsAfterInverting

	// Check solutions to see if any matches pc.Relation
	if setSolutions {
		// This call to UpdateSolutions overwrites any previous solutions
		pc.Solutions = [][]int64{}
	}
	for j := 0; j < state.NumRows(); j++ {
		// Set solutions
		putativeSolution, err := state.GetColumnOfB(j)
		if err != nil {
			return fmt.Errorf(
				"UpdateSolutions: could not get column %d of B: %q", j, err.Error(),
			)
		}
		if (len(putativeSolution) == 0) || !pc.isSolution(putativeSolution) {
			// One or more entries exceeds the capacity of int64, or the putative solution is
			// not an actual solution.
			continue
		}

		// The putative solution is an actual solution
		if (!pc.FoundRelation) && pc.solutionMatchesRelation(putativeSolution) {
			pc.FoundRelation = true
			pc.IterationsBeforeFindingRelation = pc.TotalIterations
		}
		if setSolutions {
			pc.Solutions = append(pc.Solutions, putativeSolution)
		}
	}
	pc.SolutionCount = len(pc.Solutions)
	return nil
}

// isSolution returns whether a solution works against pc.InputAsBigInt
func (pc *PSLQContext) isSolution(solution []int64) bool {
	dotProduct := big.NewInt(0)
	xLen := len(pc.InputAsBigInt)
	for i := 0; i < xLen; i++ {
		dotProduct.Add(dotProduct, big.NewInt(0).Mul(big.NewInt(solution[i]), &pc.InputAsBigInt[i]))
	}
	return dotProduct.Cmp(big.NewInt(0)) == 0
}

// solutionMatchesRelation returns whether solution is the same as the relation seeded into the PSLQ input,
// up to algebraic sign.
func (pc *PSLQContext) solutionMatchesRelation(solution []int64) bool {
	solutionMatches := true
	sgn := int64(1)
	for i := 0; i < len(pc.Relation); i++ {
		if (pc.Relation[i] != 0) && (pc.Relation[i]+solution[i] == 0) {
			sgn = -1
			break
		}
	}
	for i := 0; i < len(pc.Relation); i++ {
		if sgn*pc.Relation[i] != solution[i] {
			solutionMatches = false
		}
	}
	return solutionMatches
}

// getCausalRelation returns a relation that is to be orthogonal to the X vector
// later calculated by getX.
func getCausalRelation(xLen, maxRelationElement int) ([]int64, float64) {
	relation := make([]int64, xLen)
	relationNorm := 1.0 // the last element is 1 so start with 1.0
	for i := 0; i < xLen-1; i++ {
		relation[i] = int64(rand.Intn(maxRelationElement) - (maxRelationElement / 2))
		relationNorm += float64(relation[i] * relation[i])
	}
	relationNorm = math.Sqrt(relationNorm)
	relation[xLen-1] = 1
	return relation, relationNorm
}

// getX returns an xLen-long array, xEntries, of int64s; decimalX, their decimal
// representations; with <xEntries, relation> = 0.
func getX(relation []int64, maxX *big.Int) ([]big.Int, []string) {
	xLen := len(relation)
	xEntries := make([]big.Int, xLen)
	decimalX := make([]string, xLen)
	subTotal := big.NewInt(0)
	maxXOver2 := big.NewInt(0).Quo(maxX, big.NewInt(2))
	var xEntryPlusMaxXOver2 *big.Int
	var err error
	for i := 0; i < xLen-1; i++ {
		xEntryPlusMaxXOver2, err = cr.Int(cr.Reader, maxX)
		if err != nil {
			return nil, nil
		}
		xEntries[i] = *(big.NewInt(0).Sub(xEntryPlusMaxXOver2, maxXOver2))
		decimalX[i] = xEntries[i].String()
		subTotal.Add(subTotal, big.NewInt(0).Mul(&xEntries[i], big.NewInt(relation[i])))
	}
	xEntries[xLen-1] = *(big.NewInt(0).Neg(subTotal))
	decimalX[xLen-1] = xEntries[xLen-1].String()
	return xEntries, decimalX
}

// sphereVolume returns the volume of a sphere with the given radius in the given dimension
func sphereVolume(radius float64, dim int) float64 {
	// Example: for dim = 100, the volume of a sphere is
	// (1/(sqrt(100 pi))) (2*pi*e/100)^50 R^100
	//   = 0.0564189584 * 4.20453056E-39 * R^100
	//   = 2.37215235E-40 R^100
	const twoPiE = float64(2.0 * math.Pi * math.E)
	dimAsFloat64 := float64(dim)
	return (1.0 / math.Sqrt(dimAsFloat64*math.Pi)) *
		math.Pow(twoPiE/dimAsFloat64, 0.5*dimAsFloat64) * math.Pow(radius, dimAsFloat64)
}
