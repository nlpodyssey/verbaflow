// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decoder

import (
	"container/heap"
	"fmt"
	"math"
	"sort"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/verbaflow/sliceutils"
)

// OutputDiversityControlFunc performs the pre-processing steps that are used to narrow down the set of candidate items
// before using greedy decoding or multinomial sampling to generate the final output.
type OutputDiversityControlFunc func(logits mat.Matrix) (mat.Matrix, error)

// OutputDiversityControl returns a function used to select the next token.
func OutputDiversityControl(temp float64, topK int, topP float64) (OutputDiversityControlFunc, error) {
	if temp < 0 || temp > 1 {
		return nil, fmt.Errorf("invalid temperature value: %f. Must be between 0 and 1", temp)
	}
	if topK < 0 {
		return nil, fmt.Errorf("invalid topK value: %d. Must be >= 0", topK)
	}
	if topP < 0 || topP > 1 {
		return nil, fmt.Errorf("invalid topP value: %f. Must be between 0 and 1", topP)
	}

	result := make([]OutputDiversityControlFunc, 0, 3)
	if temp != 1 {
		result = append(result, TemperatureFunc(temp))
	}
	if topK != 0 {
		result = append(result, TopKFunc(topK, math.Inf(-1)))
	}
	if topP != 1 {
		result = append(result, TopPFunc(topP, math.Inf(-1), 1)) // minSize = 2 if beam search is enabled
	}

	return func(logits mat.Matrix) (mat.Matrix, error) {
		var err error
		for _, p := range result {
			logits, err = p(logits)
			if err != nil {
				return nil, err
			}
		}
		return logits, err
	}, nil
}

// TemperatureFunc applies a temperature to a matrix of scores.
func TemperatureFunc(temperature float64) OutputDiversityControlFunc {
	if temperature == 1 {
		return func(scores mat.Matrix) (mat.Matrix, error) {
			return scores, nil
		}
	}
	if temperature == 0 {
		temperature = 0.01 // avoid division by zero
	}
	invTemperature := 1 / temperature
	return func(scores mat.Matrix) (mat.Matrix, error) {
		return scores.ProdScalar(invTemperature), nil
	}
}

// TopKFunc applies a top-k filter to a matrix of scores.
func TopKFunc(topK int, filterValue float64) OutputDiversityControlFunc {
	return func(scores mat.Matrix) (mat.Matrix, error) {
		topK := topK
		if size := scores.Size(); size <= topK {
			topK = size
		}

		inScores := scores.Data().F64()

		rawTopScores := make(sliceutils.OrderedHeap[float64], len(inScores))
		copy(rawTopScores, inScores)

		topScores := sliceutils.ReverseHeap(&rawTopScores)
		heap.Init(topScores)
		for i := 1; i < topK; i++ {
			heap.Pop(topScores)
		}
		minScore := heap.Pop(topScores).(float64)

		return scores.Apply(func(_, _ int, v float64) float64 {
			if v < minScore {
				return filterValue
			}
			return v
		}), nil
	}
}

// TopPFunc applies a top-p filter to a matrix of scores.
// Note that when using beam decoding (with beam > 1) then minSize must be at least 2.
func TopPFunc[T float.DType](topP, filterValue T, minSize int) OutputDiversityControlFunc {
	return func(scores mat.Matrix) (mat.Matrix, error) {
		dataCopy := make([]T, scores.Size())
		copy(dataCopy, mat.Data[T](scores))
		sortedData := sliceutils.NewIndexedSlice[T](dataCopy)
		sort.Stable(sort.Reverse(sortedData))

		cumulativeProbs := mat.NewVecDense(sortedData.Slice).Softmax().CumSum()
		cumProbData := mat.Data[T](cumulativeProbs)

		indicesToRemove := make([]bool, len(cumProbData))
		for i, cp := range cumProbData {
			indicesToRemove[i] = cp > topP
		}

		if minSize > 1 {
			// Keep at least minSize (minSize-1 because we add the first one below)
			for i := minSize - 1; i >= 0; i-- {
				indicesToRemove[i] = false
			}
		}

		// Shift the indices to the right to keep also the first token above the threshold
		copy(indicesToRemove[1:], indicesToRemove[:len(indicesToRemove)-1])
		indicesToRemove[0] = false

		// Scatter sorted tensors to original indexing

		outData := make([]T, scores.Size())
		copy(outData, mat.Data[T](scores))
		for maskIndex, toRemove := range indicesToRemove {
			if !toRemove {
				continue
			}
			index := sortedData.Indices[maskIndex]
			outData[index] = filterValue
		}

		return mat.NewVecDense[T](outData), nil
	}
}
