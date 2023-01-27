// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package verbaflow

import (
	"context"
	"fmt"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/rs/zerolog/log"
	"path/filepath"

	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/verbaflow/rwkvlm"
	"github.com/nlpodyssey/verbaflow/tokenizer"
)

// VerbaFlow is the core struct of the library.
type VerbaFlow struct {
	Model          *rwkvlm.Model
	Tokenizer      tokenizer.Tokenizer
	embeddingsRepo *diskstore.Repository
}

// Load loads a VerbaFlow model from the given directory.
func Load(modelDir string) (*VerbaFlow, error) {
	tk, err := tokenizer.Load(modelDir)
	if err != nil {
		return nil, err
	}
	model, err := rwkvlm.Load(modelDir)
	if err != nil {
		return nil, err
	}
	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelDir, "repo"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository: %w", err)
	}
	err = model.ApplyEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to apply embeddings: %w", err)
	}
	return &VerbaFlow{
		Model:          model,
		Tokenizer:      tk,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

// Close closes the model resources.
func (v *VerbaFlow) Close() error {
	return v.embeddingsRepo.Close()
}

// Generate generates a text from the given prompt.
// The "out" channel is used to stream the generated text.
// The generated text will be at most `maxTokens` long (in addition to the prompt).
func (v *VerbaFlow) Generate(ctx context.Context, prompt string, maxTokens int, stopOnNewLine bool, out chan string) error {
	defer close(out)

	log.Trace().Msgf("Tokenizing prompt: %s", prompt)
	tokenized, err := v.Tokenizer.Tokenize(prompt)
	if err != nil {
		return err
	}
	log.Debug().Msgf("Token IDs: %v", tokenized)

	log.Debug().Msg("Encoding prompt...")
	x, s := v.Model.Encode(tokenized, nil, true)
	x.Value() // wait for the computation to complete

	log.Debug().Msg("Generating text")

	var generated []int

loop:
	for i := 0; i < maxTokens; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			if i > 0 {
				x, s = v.Model.Encode(generated[len(generated)-1:], s, false)
			}
			nextTokenID, err := v.predictNext(x, false)
			if err != nil {
				return err
			}
			generated = append(generated, nextTokenID)
			nextToken, err := v.Tokenizer.ReconstructText([]int{nextTokenID})
			if err != nil {
				return fmt.Errorf("failed to reconstruct text: %v", err)
			}
			if nextToken == "<|endoftext|>" {
				break loop // stop generating
			}
			if nextToken == "\n" && len(generated) > 1 && stopOnNewLine {
				break loop // stop generating
			}
			out <- nextToken
		}
	}

	return nil
}

func (v *VerbaFlow) predictNext(x ag.Node, sample bool) (int, error) {
	logits := v.Model.Predict(x)
	prob := logits.Value().Softmax()
	if sample {
		samples, err := sampleProbMultinomial(prob, 1)
		if err != nil {
			return 0, err
		}
		return samples[0], nil
	}
	best := prob.ArgMax()
	return best, nil
}

// sampleProbMultinomial extracts the next indices from a multinomial probability distribution.
func sampleProbMultinomial(probs mat.Matrix, numSamples int) ([]int, error) {
	if numSamples > probs.Size() {
		return nil, fmt.Errorf("cannot sample numSamples > probs.Size() samples")
	}

	probsData := probs.Data().F64()
	samples := make([]int, 0, numSamples)
	samplesMap := make(map[int]struct{}, numSamples)

	for len(samples) < numSamples {
		p := rand.Float[float64]()

		for probIndex, prob := range probsData {
			p -= prob
			if p < 0 {
				if _, alreadySampled := samplesMap[probIndex]; !alreadySampled {
					samplesMap[probIndex] = struct{}{}
					samples = append(samples, probIndex)
				}
				break
			}
		}
	}

	return samples, nil
}
