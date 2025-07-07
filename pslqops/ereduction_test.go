package pslqops

import (
	"fmt"
	"github.com/predrag3141/IPSLQ/bigmatrix"
	"github.com/predrag3141/IPSLQ/bignumber"
	"github.com/stretchr/testify/require"
	"math"
	"math/rand"
	"sort"
	"testing"
)

func TestEColumn(t *testing.T) {
	const (
		numTests               = 5
		maxDiagonalElementSize = 100
		mNumRows               = 10 // Requires storing 2^10 columns
	)

	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedColumnCount := 0; unreducedColumnCount < mNumRows; unreducedColumnCount++ {
			// Initializations
			var m *bigmatrix.BigMatrix
			var reciprocalDiagonal []*bignumber.BigNumber
			unreducedColumns, err := getRandomUnreducedColumns(mNumRows, unreducedColumnCount, "TestEColumn")
			require.Len(t, unreducedColumns, unreducedColumnCount)
			m, _, err = createRandomM(
				mNumRows, 2, maxDiagonalElementSize, unreducedColumns, "TestEColumn",
			)
			require.NoError(t, err)
			reciprocalDiagonal, err = getReciprocalDiagonal(m, "TestEColumn")

			// Test newEColumn
			//
			// A tree of columns is created and stored in an array with indices that
			// indicate the location of the column in the binary tree.
			// - The root of the binary tree is indexed by 0.
			// - The level one below the root is indexed by 1 and 2
			// - The level two below the root is indexed by {(2)(1)+1, (2)(1)+2,
			//   (2)(2)+1, (2)(2)+2} = {3,4,5,6}
			// - The level three below the root is indexed by {(2)(3)+1, (2)(3)+2,
			//   (2)(4)+1, (2)(4)+2, (2)(5)+1, (2)(5)+2, (2)(6)+1, (2)(6)+2} = {7,...,14}
			// - The level n below the root is indexed by 2i+1 and 2i+2 for each i
			//   indexing level n-1.
			// - Indices at level n below root are {(2^n)-1,...,(2^(n+1)-2}
			// - With no pruning, the levels are {0,...,mNumRows-2} so each of the arrays
			//   of expected and actual columns is 2^((mNumRows-2)+1)-1 = 2^(mNumRows-1)-1
			//   long. Total storage is just short of 2^mNumRows columns, including both
			//   expected and actual columns.
			expected := struct {
				eColumn *expectedColumn
			}{}
			actual := struct {
				eColumns            []*EColumn
				indicesUsed         []bool
				clone               *EColumn // clones, which are appended to eColumns
				previousLevelMin    int      // lowest index populated at the previous level
				previousLevelMax    int      // highest index populated at the previous level
				identityColumnCount int      // number of columns equal to the identity column
			}{}
			maxNumColumns := (1<<mNumRows - 1) - 1

			for j := 0; j < mNumRows-1; j++ {
				// Refresh actual.indicesUsed and actual.RegularColumn
				actual.indicesUsed = make([]bool, maxNumColumns)
				actual.eColumns = make([]*EColumn, maxNumColumns)

				// The variable, treeDepth, is the depth of the tree after a level ofEColumns
				// has been created for rows j,j+1,...,mNumRows-2. This is 1+(mNumRows-2)-j =
				// (mNumRows-1)-j levels. The last row is j+treeDepth-1 = (j+((mNumRows-1)-j)-1 =
				// mNumRows-2.
				treeDepth := (mNumRows - 1) - j

				// Check the regular columns for column j
				actual.previousLevelMin, actual.previousLevelMax = 0, 0
				for n := 0; n < treeDepth; n++ {
					// For level n == 0, create just one expected and one actual column
					if n == 0 {
						expected.eColumn = newExpectedColumn(t, m, nil, j, j)
						actual.eColumns[0], err = newEColumn(
							m, reciprocalDiagonal, j, "TestEColumn",
						)
						require.NoError(t, err)
						actual.indicesUsed[0] = true
						areEqual, reason := expected.eColumn.equals(actual.eColumns[0])
						require.True(t, areEqual, reason)
						continue
					}

					// Both expected and actual columns depend on the actual columns from the
					// previous level. Expected columns are compared to, and should match, both
					// actual columns generated from the same previous actual column.
					for i := actual.previousLevelMin; i <= actual.previousLevelMax; i++ {
						expected.eColumn = newExpectedColumn(
							t, m, actual.eColumns[i], j+n, j,
						)
						actual.eColumns[2*i+1] = actual.eColumns[i].clone()
						actual.eColumns[2*i+2], err = actual.eColumns[2*i+1].update(
							"TestEColumn",
						)
						actual.indicesUsed[2*i+1] = true
						actual.indicesUsed[2*i+2] = true
						areEqual, reason := expected.eColumn.equals(
							actual.eColumns[2*i+1],
						)
						require.True(t, areEqual, reason)
						areEqual, reason = expected.eColumn.equals(
							actual.eColumns[2*i+2],
						)
						require.True(t, areEqual, reason)
					}

					// Since the number of identity columns at each level should be at most
					// 1, the total number of identity columns should be no more than the
					// number of levels created so far, which is n+1.
					actual.identityColumnCount = 0
					for i := 0; actual.indicesUsed[i]; i++ {
						if actual.eColumns[i].equalsIdentityColumn {
							actual.identityColumnCount++
						}
					}
					require.LessOrEqual(t, actual.identityColumnCount, n+1)

					// Min and max values of previous indices are updated according to the
					// following comment above: "Indices at level n below root are
					// {(2^n)-1,...,(2^(n+1)-2}". After this loop iteration, the current
					// value of n refers to the previous iteration, so n is the iteration
					// number to use in these updates.
					actual.previousLevelMin = (1 << n) - 1
					actual.previousLevelMax = (1 << (n + 1)) - 2
					require.Less(t, actual.previousLevelMin, actual.previousLevelMax)
					require.Less(t, actual.previousLevelMax, len(actual.eColumns))
				}

				// Verify that all indices were used to form the tree of EColumns. The number
				// of columns in a tree that, like this one, was never pruned and includes
				// all previous lengths of columnOfE, is always a power of 2 minus 1.
				//
				// Of the
				expectedColumnCount := (1 << treeDepth) - 1
				for i := 0; i < expectedColumnCount; i++ {
					require.True(t, actual.indicesUsed[i])
				}
				for i := expectedColumnCount; i < maxNumColumns; i++ {
					require.False(t, actual.indicesUsed[i])
				}
			}
		}
	}
}

// Type for sorting an array of EColumns
type ByColumnOfE []*EColumn

func (a ByColumnOfE) Len() int      { return len(a) }
func (a ByColumnOfE) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByColumnOfE) Less(i, j int) bool {
	minLength := len(a[i].columnOfE)
	if minLength > len(a[j].columnOfE) {
		minLength = len(a[j].columnOfE)
	}
	for k := 0; k < minLength; k++ {
		if a[i].columnOfE[k].Cmp(a[j].columnOfE[k]) < 0 {
			return true
		}
	}
	return false
}

func TestEColumnArray(t *testing.T) {
	const (
		numTests               = 5
		maxDiagonalElementSize = 100
		mNumRows               = 10
		log2Tolerance          = -20
	)

	toleranceAsBigNumber := bignumber.NewPowerOfTwo(log2Tolerance)
	toleranceAsFloat64 := math.Pow(2.0, log2Tolerance)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedColumnCount := 0; unreducedColumnCount < mNumRows; unreducedColumnCount++ {
			for _, maxTreeDepth := range []int{0, 1, mNumRows / 2, mNumRows, mNumRows + 1} {
				// Initializations
				var m *bigmatrix.BigMatrix
				var reciprocalDiagonal []*bignumber.BigNumber
				unreducedColumns, err := getRandomUnreducedColumns(mNumRows, unreducedColumnCount, "TestEColumnArray")
				require.Len(t, unreducedColumns, unreducedColumnCount)
				m, _, err = createRandomM(
					mNumRows, 2, maxDiagonalElementSize, unreducedColumns, "TestEColumnArray",
				)
				require.NoError(t, err)
				reciprocalDiagonal, err = getReciprocalDiagonal(m, "TestEColumnArray")

				// Test newEColumnArray
				for j := 0; j < mNumRows-1; j++ {
					var eColumns *EColumnArray
					eColumns, err = newEColumnArray(
						m, reciprocalDiagonal, j, maxTreeDepth, "TestEColumnArray",
					)
					if mNumRows <= maxTreeDepth {
						require.Error(t, err)
						continue
					}

					// Set the expected struct.  The tree depth should be the number of
					// sub-diagonal elements in column j, except no more than maxTreeDepth.
					var maxUsableSqNorm float64
					for i := 0; i < mNumRows; i++ {
						var mIJ *bignumber.BigNumber
						mIJ, err = m.Get(i, j)
						require.NoError(t, err)
						mIJAsFloat64, _ := mIJ.AsFloat().Float64()
						maxUsableSqNorm += mIJAsFloat64 * mIJAsFloat64
					}
					expected := struct {
						treeDepth       int
						maxUsableSqNorm float64
					}{
						treeDepth:       (mNumRows - j) - 1,
						maxUsableSqNorm: maxUsableSqNorm,
					}
					if expected.treeDepth > maxTreeDepth {
						expected.treeDepth = maxTreeDepth
					}

					// Require invariants that should hold right after the first halfLength
					// candidate columns have been created.
					require.Greater(t, expected.maxUsableSqNorm+toleranceAsFloat64, eColumns.maxUsableSqNorm)
					require.Less(t, expected.maxUsableSqNorm-toleranceAsFloat64, eColumns.maxUsableSqNorm)
					require.NoError(t, err)
					require.Equal(t, j, eColumns.colNbr)
					require.Len(t, eColumns.candidateCol, eColumns.fullLength)
					require.GreaterOrEqual(t, expected.treeDepth, 0)
					require.Equal(t, 1<<expected.treeDepth, eColumns.fullLength)
					require.Equal(t, eColumns.fullLength/2, eColumns.halfLength)
					require.Len(t, eColumns.candidateCol, eColumns.fullLength)
					require.NotNil(t, eColumns.candidateCol)
					for i := 0; i < eColumns.halfLength; i++ {
						// After filling in the first halfLength candidate columns, the tree is
						// 1 short of its maximum depth, which is j+maxTreeDepth.
						require.NotNil(t, eColumns.candidateCol[i])
						require.Equal(t, j+expected.treeDepth-1, eColumns.candidateCol[i].rowNbr)
						require.Equal(t, j, eColumns.candidateCol[i].colNbr)
						if i > 0 {
							// newEColumnArray sorts candidate columns
							require.LessOrEqual(t, eColumns.candidateCol[i-1].squaredNorm, eColumns.candidateCol[i].squaredNorm)
						}
					}
					for i := eColumns.halfLength; i < eColumns.fullLength; i++ {
						require.Nil(t, eColumns.candidateCol[i])
					}

					for iterationNumber, isUpdated := 0, true; isUpdated; isUpdated, err = eColumns.update("TestEColumnArray") {
						require.NoError(t, err)
						// It is not required that the candidate columns be sorted by squared
						// norm, because pruning (which is what requires sorting) has not taken
						// place yet. Do require that
						// - Each row number be j + maxTreeDepth + current iteration number - 1.
						//   See comment above about row number right after creating the new
						//   column array.
						// - Each column number be j
						// - The second half of eColumns.candidateCol be nil
						for i := 0; i < eColumns.halfLength; i++ {
							require.NotNil(t, eColumns.candidateCol[i])
							require.Equal(
								t, j+expected.treeDepth+iterationNumber-1, eColumns.candidateCol[i].rowNbr,
							)
							require.Equal(t, j, eColumns.candidateCol[i].colNbr)
						}
						for i := eColumns.halfLength; i < eColumns.fullLength; i++ {
							require.Nil(t, eColumns.candidateCol[i])
						}

						// The special case where maxTreeDepth == 0 is
						if maxTreeDepth == 0 {
							require.Equal(t, 0, eColumns.halfLength)
							require.Equal(t, 1, eColumns.fullLength)
							require.Len(t, eColumns.candidateCol, 1)
							require.Nil(t, eColumns.candidateCol[0])
							continue
						}

						// eColumns.candidateCol contains groups of partially matching
						// columns of E, in which all but the last entry of columnOfE is
						// the same. Each such group should consist of 1 or 2 partially
						// matching EColumns. An invariant for groups of size 1, and
						// invariants for groups of size 2, are tested below.
						//
						// To find the groups of partially matching columns of E, sort by
						// the dictionary ordering on columnOfE.
						eLength := len(eColumns.candidateCol[0].columnOfE)
						sortedCandidateCol := make(ByColumnOfE, eColumns.halfLength)
						for k := 0; k < eColumns.halfLength; k++ {
							require.Len(t, eColumns.candidateCol[k].columnOfE, eLength)
							require.Len(t, eColumns.candidateCol[k].columnOfME, eLength)
							sortedCandidateCol[k] = &EColumn{
								columnOfE:  make([]*bignumber.BigNumber, eLength),
								columnOfME: make([]*bignumber.BigNumber, eLength),
							}
							for n := 0; n < eLength; n++ {
								sortedCandidateCol[k].columnOfE[n] = bignumber.NewFromBigNumber(
									eColumns.candidateCol[k].columnOfE[n],
								)
								sortedCandidateCol[k].columnOfME[n] = bignumber.NewFromBigNumber(
									eColumns.candidateCol[k].columnOfME[n],
								)
							}
						}
						sort.Sort(sortedCandidateCol)

						// Detect unique partial columnOfEs
						uniquePartialColumns := make(map[int][]int)
						cursor := 0
						for k := 0; k < len(sortedCandidateCol); k++ {
							// Set differsFromPrev and differsFromNext
							lastIndexMatchingPrev, lastIndexMatchingNext := -1, -1
							differsFromPrev, differsFromNext := true, true
							isFirst := k == 0
							isLast := k == eColumns.halfLength-1
							if !isFirst {
								for n := 0; n < eLength; n++ {
									if 0 == sortedCandidateCol[k].columnOfE[n].Cmp(
										sortedCandidateCol[k-1].columnOfE[n],
									) {
										lastIndexMatchingPrev = n
									} else {
										break
									}
								}
								require.Less(t, lastIndexMatchingPrev, eLength-1)
								differsFromPrev = lastIndexMatchingPrev != (eLength - 2)
							}
							if !isLast {
								for n := 0; n < eLength; n++ {
									if 0 == sortedCandidateCol[k].columnOfE[n].Cmp(
										sortedCandidateCol[k+1].columnOfE[n],
									) {
										lastIndexMatchingNext = n
									} else {
										break
									}
								}
								require.Less(t, lastIndexMatchingNext, eLength-1)
								differsFromNext = lastIndexMatchingNext != (eLength - 2)
							}

							// In the case where column k differs from column k+1, but matches
							// column k-1 -- i.e., differsFromNext && !differsFromPrev --
							// uniquePartialColumns[cursor-1] already contains []int{k-1, k}.
							// So this case does not require any action in this block.
							if differsFromPrev && differsFromNext {
								uniquePartialColumns[cursor] = []int{k}
								cursor++
							} else if differsFromPrev {
								uniquePartialColumns[cursor] = []int{k, k + 1}
								cursor++
							}

							// 3 partial columns in a row should not be the same
							require.True(t, differsFromPrev || differsFromNext)
						} // end of loop through sortedCandidateCol

						// For each unique partial column, check invariants
						var diagonalElement *bignumber.BigNumber
						var indices []int
						half := bignumber.NewPowerOfTwo(-1)
						one := bignumber.NewFromInt64(1)
						require.NoError(t, err)
						lastRowNbr := -1
						for _, indices = range uniquePartialColumns {
							// Ensure that row numbers are always the same
							rowNbr := sortedCandidateCol[indices[0]].rowNbr
							if lastRowNbr != -1 {
								require.Equal(t, lastRowNbr, rowNbr)
							}
							lastRowNbr = rowNbr

							// Compute 0.5|M[rowNbr][rowNbr]| with a small toleranceAsBigNumber added
							// to it in case of round-off error.
							diagonalElement, err = m.Get(rowNbr, rowNbr)
							require.NoError(t, err)
							halfDiagonalElement := bignumber.NewFromInt64(0).Mul(
								half, diagonalElement,
							)
							absDiagonalElement := bignumber.NewFromInt64(0).Abs(
								diagonalElement,
							)
							absHalfDiagonalElement := bignumber.NewFromInt64(0).Abs(
								halfDiagonalElement,
							)
							absHalfDiagonalElement.MulAdd(one, toleranceAsBigNumber)
							switch len(indices) {
							case 1:
								// sortedCandidateCol[indices[0]] has a unique partial columnOfE.
								// The last entry of ME should have absolute value less than
								// 0.5|M[rowNbr][rowNbr]| = absHalfDiagonalElement (which has a
								// small toleranceAsBigNumber added into it).
								absLastEntryOfME := bignumber.NewFromInt64(0).Abs(
									sortedCandidateCol[indices[0]].columnOfME[eLength-1],
								)
								require.Less(
									t, absLastEntryOfME.Cmp(absHalfDiagonalElement), 0,
								)
							case 2:
								// sortedCandidateCol[indices[*]] contains 2 entries. Of these, one
								// final entry of ME should be less than 0.5|M[rowNbr][rowNbr]| =
								// absHalfDiagonalElement (which has a small toleranceAsBigNumber added
								// into it).
								absLastEntry0OfME := bignumber.NewFromInt64(0).Abs(
									sortedCandidateCol[indices[0]].columnOfME[eLength-1],
								)
								absLastEntry1OfME := bignumber.NewFromInt64(0).Abs(
									sortedCandidateCol[indices[1]].columnOfME[eLength-1],
								)
								comparison0 := absLastEntry0OfME.Cmp(absHalfDiagonalElement) < 0
								comparison1 := absLastEntry1OfME.Cmp(absHalfDiagonalElement) < 0
								require.True(t, comparison0 || comparison1)

								// The absolute value of the sum of the two final entries of
								// ME for the partially equal EColumns should add up to
								// |M[rowNbr][rowNbr]| = absDiagonalElement (which has a small
								// toleranceAsBigNumber added into it).
								sumOfEntriesOfME := bignumber.NewFromInt64(0).Add(
									sortedCandidateCol[indices[0]].columnOfME[eLength-1],
									sortedCandidateCol[indices[1]].columnOfME[eLength-1],
								)
								absSumOfEntriesOfME := bignumber.NewFromInt64(0).Abs(
									sumOfEntriesOfME,
								)
								require.Less(t, absSumOfEntriesOfME.Cmp(absDiagonalElement), 0)
							default:
								// All entries of uniquePartialColumns should have length 1 or 2
								require.True(t, false)
							}
						} // end of range uniquePartialColumns
						iterationNumber++
					}
				}
			}
		}
	}
}

func TestBestColumn(t *testing.T) {
	const (
		numTests               = 5
		maxDiagonalElementSize = 100
		mNumRows               = 10
		maxTreeDepth           = 5
		log2Tolerance          = -20
	)

	counts := struct {
		identityColumn int
		nilBestColumn  int
		columnsTested  int
	}{
		identityColumn: 0,
		nilBestColumn:  0,
		columnsTested:  0,
	}

	toleranceAsFloat64 := math.Pow(2.0, log2Tolerance)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedColumnCount := 0; unreducedColumnCount < mNumRows; unreducedColumnCount++ {
			// Initializations
			var m *bigmatrix.BigMatrix
			var reciprocalDiagonal []*bignumber.BigNumber
			unreducedColumns, err := getRandomUnreducedColumns(mNumRows, unreducedColumnCount, "TestBestColumn")
			require.Len(t, unreducedColumns, unreducedColumnCount)
			m, _, err = createRandomM(
				mNumRows, 2, maxDiagonalElementSize, unreducedColumns, "TestBestColumn",
			)
			require.NoError(t, err)
			reciprocalDiagonal, err = getReciprocalDiagonal(m, "TestBestColumn")

			for j := 0; j < mNumRows-1; j++ {
				var eColumns *EColumnArray
				var bestEColumn *EColumn
				expected := struct {
					treeDepth int
				}{
					treeDepth: (mNumRows - j) - 1,
				}
				if expected.treeDepth > maxTreeDepth {
					expected.treeDepth = maxTreeDepth
				}

				eColumns, err = newEColumnArray(
					m, reciprocalDiagonal, j, maxTreeDepth, "TestBestColumn",
				)
				bestEColumn, err = eColumns.bestColumn("TestBestColumn")
				counts.columnsTested++
				require.NoError(t, err)
				require.Equal(t, 1<<expected.treeDepth, eColumns.fullLength)
				if bestEColumn == nil {
					counts.nilBestColumn++
					continue
				}
				require.Less(t, bestEColumn.squaredNorm-toleranceAsFloat64, eColumns.maxUsableSqNorm)
				require.Equal(t, bestEColumn.rowNbr, mNumRows-1)
				require.Equal(t, bestEColumn.colNbr, j)
				if bestEColumn.equalsIdentityColumn {
					counts.identityColumn++
				}
			}
		}
	}
	t.Logf(
		"\nNil bestColumns: %d\nIdentity best columns: %d\nTotal best columns: %d\n",
		counts.nilBestColumn, counts.identityColumn, counts.columnsTested,
	)
}

func TestGetE(t *testing.T) {
	const (
		numTests               = 5
		maxDiagonalElementSize = 100
		mNumRows               = 10
		eNumRows               = 11
		maxTreeDepth           = 5
		log2Tolerance          = -20
	)

	counts := struct {
		columnsBounded   int
		columnsReduced   int
		columnsProcessed int
		identityMatrix   int
		matricesTested   int
	}{
		columnsBounded:   0,
		columnsReduced:   0,
		columnsProcessed: 0,
		identityMatrix:   0,
		matricesTested:   0,
	}

	toleranceAsBigNumber := bignumber.NewPowerOfTwo(log2Tolerance)
	zero := bignumber.NewFromInt64(0)
	one := bignumber.NewFromInt64(1)
	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedColumnCount := 0; unreducedColumnCount < mNumRows; unreducedColumnCount++ {
			// Initializations
			var m *bigmatrix.BigMatrix
			var columnsReduced, columnsBounded int
			unreducedColumns, err := getRandomUnreducedColumns(mNumRows, unreducedColumnCount, "TestGetE")
			require.Len(t, unreducedColumns, unreducedColumnCount)
			m, _, err = createRandomM(
				mNumRows, 2, maxDiagonalElementSize, unreducedColumns, "TestGetE",
			)
			require.NoError(t, err)
			e := bigmatrix.NewEmpty(eNumRows, eNumRows)
			columnsReduced, columnsBounded, err = getE(m, e, maxTreeDepth, "TestGetE")
			require.NoError(t, err)
			if columnsReduced == 0 {
				counts.identityMatrix++
			}
			counts.columnsReduced += columnsReduced
			counts.columnsBounded += columnsBounded
			counts.columnsProcessed += m.NumRows() - 1
			counts.matricesTested++

			// Check that in the upper right are 0s, and on the diagonal are 1s
			for i := 0; i < eNumRows; i++ {
				var eII *bignumber.BigNumber
				eII, err = e.Get(i, i)
				require.NoError(t, err)
				require.Equal(t, 0, one.Cmp(eII))
				for j := i + 1; j < eNumRows; j++ {
					var eIJ *bignumber.BigNumber
					eIJ, err = e.Get(i, j)
					require.NoError(t, err)
					require.Equal(t, 0, zero.Cmp(eIJ))
				}
			}

			// Check that columns of ME are no longer than columns of M
			var me *bigmatrix.BigMatrix
			me, err = bigmatrix.NewEmpty(mNumRows, mNumRows).MulUpperLeft(m, e)
			require.Equal(t, mNumRows, me.NumRows())
			require.Equal(t, mNumRows, me.NumCols())
			for j := 0; j < mNumRows; j++ {
				// Compute column norms for M and ME
				mColumnNorm := bignumber.NewFromInt64(0)
				meColumnNorm := bignumber.NewFromInt64(0)
				for i := 0; i < mNumRows; i++ {
					var mIJ, meIJ *bignumber.BigNumber
					mIJ, err = m.Get(i, j)
					require.NoError(t, err)
					meIJ, err = me.Get(i, j)
					require.NoError(t, err)
					mIJSq := bignumber.NewFromInt64(0).Mul(mIJ, mIJ)
					meIJSq := bignumber.NewFromInt64(0).Mul(meIJ, meIJ)
					mColumnNorm.MulAdd(one, mIJSq)
					meColumnNorm.MulAdd(one, meIJSq)
				}

				// Require that column j norm of ME be no more than column j norm of M
				mColumnNorm.MulAdd(one, toleranceAsBigNumber)
				require.Greater(t, 0, meColumnNorm.Cmp(mColumnNorm))
			}
		}
	}
	t.Logf(
		"\nBounded columns:   %3d Reduced columns:   %3d Ratio: %f\n"+
			"Reduced columns:   %3d Columns processed: %3d Ratio: %f\n"+
			"Identity matrices: %3d Matrices tested:   %3d Ratio: %f\n",
		counts.columnsBounded, counts.columnsReduced,
		float64(counts.columnsBounded)/float64(counts.columnsReduced),
		counts.columnsReduced, counts.columnsProcessed,
		float64(counts.columnsReduced)/float64(counts.columnsProcessed),
		counts.identityMatrix, counts.matricesTested,
		float64(counts.identityMatrix)/float64(counts.matricesTested),
	)
}

func TestIsColumnReduced(t *testing.T) {
	const (
		numTests         = 100
		maxDiagonalEntry = 100
		numRows          = 17
	)

	for testNbr := 0; testNbr < numTests; testNbr++ {
		for unreducedColumnCount := 0; unreducedColumnCount < numRows; unreducedColumnCount++ {
			// Initializations
			var m *bigmatrix.BigMatrix
			var expected *RandomMInfo
			unreducedColumns, err := getRandomUnreducedColumns(
				numRows, unreducedColumnCount, " TestIsColumnReduced",
			)
			require.NoError(t, err)
			require.Len(t, unreducedColumns, unreducedColumnCount)
			m, expected, err = createRandomM(
				numRows, 2, maxDiagonalEntry, unreducedColumns, "TestIsColumnReduced",
			)
			require.NoError(t, err)

			// Get actual values
			var actual struct {
				isReduced       bool
				unreducedRow    int
				unreducedColumn int
			}
			actual.isReduced, actual.unreducedRow, actual.unreducedColumn, err = isColumnReduced(
				m, -bigNumberBitTolerance, "TestIsColumnReduced",
			)
			require.NoError(t, err)

			// Compare expected to actual
			require.Equal(t, expected.unreducedRow, actual.unreducedRow)
			require.Equal(t, expected.unreducedColumn, actual.unreducedColumn)
			if unreducedColumnCount == 0 {
				require.True(t, actual.isReduced)
			} else {
				require.False(t, actual.isReduced)
			}
		}
	}
}

// possible is a structure type that holds possible entries of E and ME, and the squared
// norm of the entries up to the current row number.
type possible struct {
	entryOfE          *bignumber.BigNumber
	entryOfME         *bignumber.BigNumber
	squaredNorm       float64
	boundedByDiagonal bool
}

type ByEntryOfE []*possible

func (a ByEntryOfE) Len() int      { return len(a) }
func (a ByEntryOfE) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByEntryOfE) Less(i, j int) bool {
	return a[i].entryOfE.Cmp(a[j].entryOfE) < 0
}

type ByEntryOfME []*possible

func (a ByEntryOfME) Len() int      { return len(a) }
func (a ByEntryOfME) Swap(i, j int) { a[i], a[j] = a[j], a[i] }
func (a ByEntryOfME) Less(i, j int) bool {
	return a[i].entryOfME.Cmp(a[j].entryOfME) < 0
}

type expectedColumn struct {
	mNumRows         int
	eNumRows         int
	rowNbr           int
	colNbr           int
	possible         []*possible
	lengthOfPossible int
	tolerance        *bignumber.BigNumber
}

func newExpectedColumn(
	t *testing.T, m *bigmatrix.BigMatrix, previousColumn *EColumn, rowNbr, colNbr int,
) *expectedColumn {
	// Set local variables, mNumRows, rowNbr, colNbr and isIdentityColumn
	// Here is where retVal.possible is always initialized
	const (
		log2Tolerance = -20
	)

	// Check validity of rowNbr and colNbr
	require.Less(t, rowNbr, m.NumRows())
	require.Less(t, colNbr, m.NumRows()-1) // since the last column of E is just (0,0,...,0,1)
	if previousColumn == nil {
		// First-level regular column
		require.Equal(t, rowNbr, colNbr)
	} else {
		// Second or higher level regular column
		require.Greater(t, rowNbr, colNbr)
	}

	// Declarations and initializations
	var err error
	retVal := &expectedColumn{
		mNumRows:  m.NumRows(),
		eNumRows:  1 + m.NumRows(),
		rowNbr:    rowNbr,
		colNbr:    colNbr,
		tolerance: bignumber.NewPowerOfTwo(log2Tolerance),
	}

	// If the previous column is nil, the number of possible entries in E is 1, so
	// retVal can be constructed and returned right away.
	if previousColumn == nil {
		var entryOfME *bignumber.BigNumber
		retVal.lengthOfPossible = 1
		retVal.possible = make([]*possible, retVal.lengthOfPossible)
		entryOfME, err = m.Get(rowNbr, colNbr)
		entryOfMEAsFloat64, _ := entryOfME.AsFloat().Float64()
		retVal.possible[0] = &possible{
			entryOfE:          bignumber.NewFromInt64(1),
			entryOfME:         entryOfME,
			squaredNorm:       entryOfMEAsFloat64 * entryOfMEAsFloat64,
			boundedByDiagonal: true,
		}
		return retVal
	}

	// Ascertain whether the entries of ME in the previous column are bounded in absolute value
	// by half the diagonal elements to their right.
	var prevColumnIsBounded bool
	prevColumnIsBounded, err = previousColumn.isBoundedByDiagonal(retVal.tolerance, "newExpectedColumn")

	// Set partialEntryOfME to ME[rowNbr][colNbr] - M[rowNbr][rowNbr]E[rowNbr][colNbr].
	// The entry of ME[rowNbr][rowNbr] will be
	partialEntryOfME := bignumber.NewFromInt64(0)
	for k := colNbr; k < rowNbr; k++ {
		var mIK *bignumber.BigNumber
		mIK, err = m.Get(rowNbr, k)
		require.NoError(t, err)
		partialEntryOfME.MulAdd(mIK, previousColumn.columnOfE[k])
	}

	// Get M[rowNbr][rowNbr] to compute the term M[rowNbr][rowNbr]E[rowNbr][colNbr]
	// which needs to be added to partialEntryOfME to get ME[rowNbr][rowNbr] for each
	// possible E[rowNbr][rowNbr].
	var mII *bignumber.BigNumber
	mII, err = m.Get(rowNbr, rowNbr)
	require.NoError(t, err)
	mIIAsFloat64, _ := mII.AsFloat().Float64()
	mIISqAsFloat64 := mIIAsFloat64 * mIIAsFloat64

	// Use partialEntryOfME as input to the function, getPossibleEntriesOfE, to compute
	// retVal.possible[*].entryOfE.
	possibleEntriesOfE := getPossibleEntriesOfE(t, m, rowNbr, partialEntryOfME, retVal.tolerance)
	retVal.lengthOfPossible = len(possibleEntriesOfE)
	retVal.possible = make([]*possible, retVal.lengthOfPossible)
	for n := 0; n < retVal.lengthOfPossible; n++ {
		// Set the possible entry of ME
		require.NoError(t, err)
		entryOfME := bignumber.NewFromBigNumber(partialEntryOfME).MulAdd(mII, possibleEntriesOfE[n])
		entryOfMEAsFloat64, _ := entryOfME.AsFloat().Float64()
		entryOfMESqAsFloat64 := entryOfMEAsFloat64 * entryOfMEAsFloat64

		// Set currentColumnIsBounded based on whether the previous column is bounded and
		// the entries of ME and M on the current row.
		currentColumnIsBounded := prevColumnIsBounded
		if entryOfMESqAsFloat64 > math.Pow(2, log2Tolerance)+mIISqAsFloat64/4.0 {
			currentColumnIsBounded = false
		}

		// Set retVal.possible[n]
		retVal.possible[n] = &possible{
			entryOfE:          possibleEntriesOfE[n],
			entryOfME:         entryOfME,
			squaredNorm:       previousColumn.squaredNorm + entryOfMESqAsFloat64,
			boundedByDiagonal: currentColumnIsBounded,
		}
	}

	// Before returning retVal, the next few blocks of code check the following invariants
	// based on the requirement that E be an integer matrix and entries of ME be minimized.
	// - The possible entries of E should be adjacent integers in order to center the
	//   corresponding entries of ME around 0.
	// - If there are exactly two possible entries of E -- e1 and e2 -- giving rise to
	//   corresponding non-zero entries me1 and me2 of ME, then me1 and me2 should have
	//   different signs. To see why, consider |me1| to be <= |me2| without loss of generality.
	//   If me1 and me2 were to have the same sign, it would mean that e2 had been set
	//   incorrectly; |me2| could be reduced by choosing e2 to be the other integer adjacent
	//   to e1.
	// - If there are exactly three entries of E -- e1, e2 and e3 -- giving rise to
	//   corresponding entries me1, me2 and me3 of ME, then without loss of generality,
	//   me1 < me2 < me3. Check that, within tolerance, me1 = -me3 and me2 = 0.

	// Check that the possible entries of E are adjacent integers (first bullet in the long
	// comment above).
	sort.Sort(ByEntryOfE(retVal.possible))
	one := bignumber.NewFromInt64(1)
	for _, p := range retVal.possible {
		require.True(t, p.entryOfE.IsInt())
	}
	for n := 1; n < len(retVal.possible); n++ {
		diff := bignumber.NewFromInt64(0).Sub(
			retVal.possible[n].entryOfE, retVal.possible[n-1].entryOfE,
		)
		require.True(t, diff.Equals(one, retVal.tolerance))
	}

	// Check invariants for the entries of ME (last two bullets in the long comment above).
	sort.Sort(ByEntryOfME(retVal.possible))
	zero := bignumber.NewFromInt64(0)
	require.LessOrEqual(t, len(retVal.possible), 3)
	require.Greater(t, len(retVal.possible), 0)
	if len(retVal.possible) == 2 {
		if !retVal.possible[0].entryOfME.Equals(zero, retVal.tolerance) {
			require.True(t, retVal.possible[0].entryOfME.IsNegative())
		}
		require.False(t, retVal.possible[1].entryOfME.IsNegative())
	} else if len(retVal.possible) == 3 {
		require.True(t, retVal.possible[0].entryOfME.IsNegative())
		require.True(t, retVal.possible[1].entryOfME.Equals(zero, retVal.tolerance))
		require.False(t, retVal.possible[2].entryOfME.IsNegative())
	}
	return retVal
}

// equals returns whether the expected and actual columns are equal, and the reason for
// this determination.
func (e *expectedColumn) equals(a *EColumn) (bool, string) {
	// Initializations
	const float64Tolerance = 1.e-10
	reason := printEColumn(a, "In equals:")
	if e.lengthOfPossible > 0 {
		reason += fmt.Sprintf(
			"expected number of possible values of E[%d][%d] is %d > 0",
			e.rowNbr, e.colNbr, e.lengthOfPossible,
		)
	} else {
		return false, reason + fmt.Sprintf(
			"expected number of possible values of E[%d][%d] is %d <= 0",
			e.rowNbr, e.colNbr, e.lengthOfPossible,
		)
	}

	// Check scalars
	var valid bool
	valid, reason = e.checkScalars(a, reason)
	if !valid {
		return false, reason
	}

	// Check E[rowNbr][colNbr]
	matchingIndexIntoPossible := -1
	valid, reason, matchingIndexIntoPossible = e.checkPossibleEntriesOfE(a, reason)
	if !valid {
		return false, reason
	}

	// Check that, given matchingIndexIntoPossible, ME[a.rowNbr][a.colNbr] matches
	valid, reason = e.checkMEEntry(a, reason, matchingIndexIntoPossible)
	if !valid {
		return false, reason
	}

	// Check that, given matchingIndexIntoPossible, squaredNorm matches
	valid, reason = e.checkSquaredNorm(a, reason, matchingIndexIntoPossible, float64Tolerance)
	if !valid {
		return false, reason
	}

	// Check that, given matchingIndexIntoPossible, squaredNorm matches
	valid, reason = e.checkBoundedByDiagonal(a, reason, matchingIndexIntoPossible, float64Tolerance)
	if !valid {
		return false, reason
	}
	return true, reason
}

// checkScalars checks scalars in the expected column against scalars in the actual column
// and returns
//
// - a conclusion about whether scalars in e match those in the actual column
//
// - the reason for the conclusion
func (e *expectedColumn) checkScalars(a *EColumn, reason string) (bool, string) {
	if e.rowNbr == a.rowNbr {
		reason += fmt.Sprintf(
			"\nexpected rowNbr = %d == %d = actual rowNbr", e.rowNbr, a.rowNbr,
		)
	} else {
		return false, fmt.Sprintf(
			"expected rowNbr = %d != %d = actual rowNbr", e.rowNbr, a.rowNbr,
		)
	}
	if e.colNbr == a.colNbr {
		reason += fmt.Sprintf(
			"\nexpected colNbr = %d == %d = actual colNbr", e.colNbr, a.colNbr,
		)
	} else {
		return false, reason + fmt.Sprintf(
			"\nexpected colNbr = %d != %d = actual colNbr", e.colNbr, a.colNbr,
		)
	}
	if e.eNumRows == len(a.columnOfE) {
		reason += fmt.Sprintf(
			"\nexpected eNumRows = %d == %d = actual len(a.columnOfE)",
			e.eNumRows, len(a.columnOfE),
		)
	} else {
		return false, reason + fmt.Sprintf(
			"\nexpected eNumRows = %d != %d = actual len(a.columnOfE)",
			e.eNumRows, len(a.columnOfE),
		)
	}
	if e.eNumRows == len(a.columnOfME) {
		reason += fmt.Sprintf(
			"\nexpected eNumRows = %d == %d = actual len(a.columnOfME)",
			e.eNumRows, len(a.columnOfME),
		)
	} else {
		return false, reason + fmt.Sprintf(
			"\nexpected eNumRows = %d != %d = actual len(a.columnOfME)",
			e.eNumRows, len(a.columnOfME),
		)
	}
	if e.mNumRows == a.m.NumRows() {
		return true, reason + fmt.Sprintf(
			"\nexpected mNumRows = %d == %d = actual m.NumRows()", e.mNumRows, a.m.NumRows(),
		)
	}
	return false, reason + fmt.Sprintf(
		"\nexpected mNumRows = %d != %d = actual m.NumRows()", e.mNumRows, a.m.NumRows(),
	)
}

// checkPossibleEntriesOfE checks a.columnOfE[a.rowNbr] and returns
//
// - a conclusion about whether an some e.possible[*].entryOfE matches a.columnOfE[a.rowNbr]
//
// - the reason for the conclusion
//
// - the index into e.possible that matches the actual EColumn, if any, else -1
func (e *expectedColumn) checkPossibleEntriesOfE(a *EColumn, reason string) (bool, string, int) {
	// Set matchingIndexIntoPossible and expectedAsStr
	matchingIndexIntoPossible := -1
	expectedAsStr := "e.possible[*].entryOfE = {"
	for ei := 0; ei < e.lengthOfPossible; ei++ {
		_, expectedEEntryAsStr := e.possible[ei].entryOfE.String()
		if ei != 0 {
			expectedAsStr += ","
		}
		expectedAsStr += expectedEEntryAsStr
		if e.possible[ei].entryOfE.Equals(a.columnOfE[a.rowNbr], e.tolerance) {
			matchingIndexIntoPossible = ei
			break
		}
	}
	expectedAsStr += "}"

	// Report the conclusion
	_, actualEEntryAsStr := a.columnOfE[a.rowNbr].String()
	actualAsStr := fmt.Sprintf("%s = a.columnOfE[%d]", actualEEntryAsStr, a.rowNbr)
	if 0 <= matchingIndexIntoPossible {
		_, matchingExpectedEEntryAsStr := e.possible[matchingIndexIntoPossible].entryOfE.String()
		reason += fmt.Sprintf(
			"\n%s contains %s == %s\n",
			expectedAsStr, matchingExpectedEEntryAsStr, actualAsStr,
		)
		return true, reason, matchingIndexIntoPossible
	}
	return false, reason + fmt.Sprintf(
		"\n%s does not contain %s", expectedAsStr, actualAsStr,
	), -1
}

// checkMEEntry checks a.columnOfME[a.rowNbr] and returns
//
//   - a conclusion about whether the expected value of ME[a.rowNbr][a.colNbr] matches
//     a.columnOfME[a.rowNbr]
//
// - the reason for the conclusion
func (e *expectedColumn) checkMEEntry(
	a *EColumn, reason string, matchingIndexIntoPossible int,
) (bool, string) {
	_, expectedMEEntryAsStr := e.possible[matchingIndexIntoPossible].entryOfME.String()
	_, actualMEEntryAsStr := a.columnOfME[a.rowNbr].String()
	expectedAsStr := fmt.Sprintf(
		"e.possible[%d].entryOfME = %s", matchingIndexIntoPossible, expectedMEEntryAsStr,
	)
	actualAsStr := fmt.Sprintf(
		"%s = a.columnOfME[%d]", actualMEEntryAsStr, a.rowNbr,
	)
	if e.possible[matchingIndexIntoPossible].entryOfME.Equals(a.columnOfME[a.rowNbr], e.tolerance) {
		reason += fmt.Sprintf("\n%s == %s\n", expectedAsStr, actualAsStr)
		return true, reason
	}
	return false, reason + fmt.Sprintf("\n%s != %s\n", expectedAsStr, actualAsStr)
}

// checkSquaredNorm checks a.squaredNorm and returns
//
//   - a conclusion about whether the expected squared norm for the column matches
//     a.squaredNorm
//
// - the reason for the conclusion
func (e *expectedColumn) checkSquaredNorm(
	a *EColumn, reason string, matchingIndexIntoPossible int, float64Tolerance float64,
) (bool, string) {
	expectedAsStr := fmt.Sprintf(
		"e.possible[%d].squaredNorm = %f",
		matchingIndexIntoPossible, e.possible[matchingIndexIntoPossible].squaredNorm,
	)
	actualAsStr := fmt.Sprintf("%f = a.squaredNorm", a.squaredNorm)
	if math.Abs(e.possible[matchingIndexIntoPossible].squaredNorm-a.squaredNorm) < float64Tolerance {
		reason += fmt.Sprintf("\n%s == %s", expectedAsStr, actualAsStr)
		return true, reason
	}
	return false, reason + fmt.Sprintf("\n%s != %s", expectedAsStr, actualAsStr)
}

// checkBoundedByDiagonal checks a.squaredNorm and returns
//
//   - whether the expected and actual boundedness for the column match. Here boundedness
//     refers to whether every element of column a is bounded in absolute value by half of
//     the diagonal element of M to its right.
//
// - the reason for the conclusion
//
// - any error
func (e *expectedColumn) checkBoundedByDiagonal(
	a *EColumn, reason string, matchingIndexIntoPossible int, float64Tolerance float64,
) (bool, string) {
	expectedAsStr := fmt.Sprintf(
		"e.possible[%d].boundedByDiagonal = %v",
		matchingIndexIntoPossible, e.possible[matchingIndexIntoPossible].boundedByDiagonal,
	)
	actualIsBoundedByDiagonal, err := a.isBoundedByDiagonal(e.tolerance, "checkBoundedByDiagonal")

	if err != nil {
		return true, reason + fmt.Sprintf(
			"checkBoundedByDiagonal: error calling isBoundedByDiagonal: %q", err.Error(),
		)
	}
	actualAsStr := fmt.Sprintf("%v = a.isBoundedByDiagonal()", actualIsBoundedByDiagonal)
	if e.possible[matchingIndexIntoPossible].boundedByDiagonal == actualIsBoundedByDiagonal {
		reason += fmt.Sprintf("\n%s == %s", expectedAsStr, actualAsStr)
		return true, reason
	}
	return false, reason + fmt.Sprintf("\n%s != %s", expectedAsStr, actualAsStr)
}

// getPossibleEntriesOfE returns the entries of E that would be valid in an EColumn. Typically
// this is a 2-long array of BigNumbers, but on occasion round-off error is negligible and any
// of three entries of E are possible.
func getPossibleEntriesOfE(
	t *testing.T, m *bigmatrix.BigMatrix, rowNbr int, partialEntryOfME, tolerance *bignumber.BigNumber,
) []*bignumber.BigNumber {
	// Compute a non-integer value for E[retVal.rowNbr][retVal.colNbr]. After this,
	// all possible ways of rounding to an integer are computed and stored in retVal.
	var minusMII, entryAsFloat *bignumber.BigNumber
	var retVal []*bignumber.BigNumber
	zero := bignumber.NewFromInt64(0)
	one := bignumber.NewFromInt64(1)
	mII, err := m.Get(rowNbr, rowNbr)
	require.NoError(t, err)
	minusMII = bignumber.NewFromInt64(0).Sub(zero, mII)
	require.NoError(t, err)

	//             rowNbr-1
	// E[rowNbr] =   sum   M[rowNbr][k] E[k][colNbr] / -M[rowNbr][rowNbr]
	//               k=0
	entryAsFloat, err = bignumber.NewFromInt64(0).Quo(partialEntryOfME, minusMII)

	// If entryAsFloat is close to an integer, there are three possible entries of E.
	// Otherwise, there are just two: rounding up or down.
	var roundedTowardsZero, roundedAwayFromZero, thirdValue, absThirdValue *bignumber.BigNumber
	absEntryAsFloat := bignumber.NewFromInt64(0).Abs(entryAsFloat)
	absRoundedTowardsZero := bignumber.NewFromBigNumber(absEntryAsFloat).RoundTowardsZero()
	absRoundedAwayFromZero := bignumber.NewFromInt64(0).Add(absRoundedTowardsZero, one)
	diff := bignumber.NewFromInt64(0).Sub(absEntryAsFloat, absRoundedTowardsZero)
	if diff.Cmp(tolerance) < 0 {
		absThirdValue = bignumber.NewFromInt64(0).Sub(absRoundedTowardsZero, one)
	}
	diff = bignumber.NewFromInt64(0).Sub(absRoundedAwayFromZero, absEntryAsFloat)
	if diff.Cmp(tolerance) < 0 {
		absThirdValue = bignumber.NewFromInt64(0).Add(absRoundedAwayFromZero, one)
	}
	if entryAsFloat.IsNegative() {
		roundedTowardsZero = bignumber.NewFromInt64(0).Sub(zero, absRoundedTowardsZero)
		roundedAwayFromZero = bignumber.NewFromInt64(0).Sub(zero, absRoundedAwayFromZero)
		if absThirdValue != nil {
			thirdValue = bignumber.NewFromInt64(0).Sub(zero, absThirdValue)
		}
	} else {
		roundedTowardsZero = bignumber.NewFromBigNumber(absRoundedTowardsZero)
		roundedAwayFromZero = bignumber.NewFromBigNumber(absRoundedAwayFromZero)
		if absThirdValue != nil {
			thirdValue = bignumber.NewFromBigNumber(absThirdValue)
		}
	}
	if thirdValue == nil {
		retVal = make([]*bignumber.BigNumber, 2)
		retVal[0] = roundedTowardsZero
		retVal[1] = roundedAwayFromZero
	} else {
		retVal = make([]*bignumber.BigNumber, 3)
		retVal[0] = roundedTowardsZero
		retVal[1] = roundedAwayFromZero
		retVal[2] = thirdValue
	}
	return retVal
}

// getRandomUnreducedColumns returns a list of column numbers of M to be passed to createRandomM,
// which will generate an instance of matrix M in which the specified columns are unreduced.
func getRandomUnreducedColumns(numRows, numUnreducedColumns int, caller string) ([]int, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getRandomUnreducedColumns", caller)

	// Check input
	if numRows <= numUnreducedColumns {
		return nil, fmt.Errorf(
			"%s: numRows = %d <= %d = numUnreducedRows", caller, numRows, numUnreducedColumns,
		)
	}

	// Get and return unreduced columns array
	var unreducedColumns []int
	switch numUnreducedColumns {
	case 0:
		unreducedColumns = []int{}
	case 1:
		// Avoid choosing column numRows-1, which has no entries below the diagonal
		unreducedColumns = []int{rand.Intn(numRows - 1)}
	default:
		// As in the case of unreducedColumnCount being 2, avoid choosing row 0
		unreducedColumns = getRandomIndices(numUnreducedColumns, numRows-1)
	}
	return unreducedColumns, nil
}

// columnNeedsReduction returns whether column columnNumber of M needs reduction, what row
// exceeds the threshold, and any error. The threshold is increased by 2^log2tolerance
// if log2tolerance is negative, else by zero.
func columnNeedsReduction(
	m *bigmatrix.BigMatrix, rowThresh []*bignumber.BigNumber,
	columnNumber, log2tolerance int, caller string,
) (bool, int, error) {
	caller = fmt.Sprintf("%s-columnNeedsReduction", caller)
	var tolerance *bignumber.BigNumber
	if log2tolerance < 0 {
		tolerance = bignumber.NewPowerOfTwo(int64(log2tolerance))
	}

	// Check whether all elements in the row are bounded by rowThresh.
	for i := columnNumber + 1; i < m.NumRows(); i++ {
		mIJ, err := m.Get(i, columnNumber)
		if err != nil {
			return false, -1, fmt.Errorf("%s: could not get H[%d][%d]: %q", caller, i, columnNumber, err.Error())
		}
		absMIJ := bignumber.NewFromInt64(0).Abs(mIJ)
		if tolerance != nil {
			// Decrease absHIJ before comparing it to the threshold. This avoids false alarms.
			absMIJ = bignumber.NewFromInt64(0).Sub(absMIJ, tolerance)
		}
		if absMIJ.Cmp(rowThresh[i]) > 0 {
			return true, i, nil
		}
	}

	// No elements in row rowNumber need reducing
	return false, -1, nil
}

// getRowThresholds returns the maximum absolute value in each column of M
// after reduction, and any error.
func getRowThresholds(
	m *bigmatrix.BigMatrix, caller string,
) ([]*bignumber.BigNumber, error) {
	// Initializations
	caller = fmt.Sprintf("%s-getRowThresholds", caller)
	numRows := m.NumRows()
	rowThresh := make([]*bignumber.BigNumber, numRows)
	half := bignumber.NewPowerOfTwo(-1)

	// Set thresholds
	for i := 0; i < numRows; i++ {
		mII, err := m.Get(i, i)
		if err != nil {
			return nil, fmt.Errorf("%s: could not get M[%d][%d]: %q", caller, i, i, err.Error())
		}
		absMII := bignumber.NewFromInt64(0).Abs(mII)
		rowThresh[i] = bignumber.NewFromInt64(0).Mul(half, absMII)
	}
	return rowThresh, nil
}

// isColumnReduced is a convenience function that wraps a series of columnsTested of the columns of M
// that use columnNeedsReduction. Returned values:
//
// - whether M is column reduced
//
// - a row and column that exceeds the threshold for M being reduced, or (-1, -1) if M is reduced
//
// - any error encountered
func isColumnReduced(
	m *bigmatrix.BigMatrix, log2tolerance int, caller string,
) (bool, int, int, error) {
	caller = fmt.Sprintf("%s-isColumnReduced", caller)
	numCols := m.NumCols()
	rowThresh, err := getRowThresholds(m, caller)
	if err != nil {
		return true, -1, -1, err
	}
	for j := 0; j < numCols-1; j++ {
		var needsReduction bool
		var row int
		needsReduction, row, err = columnNeedsReduction(
			m, rowThresh, j, log2tolerance, caller,
		)
		if err != nil {
			return true, -1, -1, err
		}
		if needsReduction {
			return false, row, j, nil
		}
	}
	return true, -1, -1, nil
}
