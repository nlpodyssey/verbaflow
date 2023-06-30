// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decoder

import (
	"fmt"

	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/rs/zerolog/log"
)

type OutputSelectionFunc func(logits mat.Tensor) (int, float64, error)

func OutputSelection(sampling bool) OutputSelectionFunc {
	if sampling {
		log.Trace().Msg("using multinomial sampling")
		return MultinomialSampling()
	}
	log.Trace().Msg("using greedy decoding")
	return GreedyDecoding()
}

func GreedyDecoding() OutputSelectionFunc {
	return func(logits mat.Tensor) (int, float64, error) {
		probs := logits.(mat.Matrix).Softmax()
		argmax := probs.ArgMax()
		return argmax, probs.ScalarAt(argmax).F64(), nil
	}
}

func MultinomialSampling() OutputSelectionFunc {
	return func(logits mat.Tensor) (int, float64, error) {
		probs := logits.(mat.Matrix).Softmax()
		samples, err := multinomial(probs, 1)
		if err != nil {
			return 0, 0, err
		}
		return samples[0], probs.ScalarAt(samples[0]).F64(), nil
	}
}

// multinomial extracts the next indices from a multinomial probability distribution.
func multinomial(input mat.Tensor, numSamples int) ([]int, error) {
	if numSamples > input.Size() {
		return nil, fmt.Errorf("numSamples (%d) must be less than or equal to the size of the input (%d)", numSamples, input.Size())
	}

	samples := make([]int, 0, numSamples)
	samplesMap := make(map[int]struct{}, numSamples)

	data := input.Data().F64()
	for len(samples) < numSamples {
		p := rand.Float[float64]()

		for i, value := range data {
			p -= value
			if p < 0 {
				if _, alreadySampled := samplesMap[i]; !alreadySampled {
					samplesMap[i] = struct{}{}
					samples = append(samples, i)
				}
				break
			}
		}
	}

	return samples, nil
}
