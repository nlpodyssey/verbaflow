// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package verbaflow

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"unicode"

	"github.com/nlpodyssey/rwkv"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/rand"
	"github.com/nlpodyssey/verbaflow/rwkvlm"
	"github.com/nlpodyssey/verbaflow/tokenizer"
	"github.com/rs/zerolog/log"
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
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("error: unable to find the model file or directory '%s'. Please ensure that the model has been successfully downloaded and converted before trying again", modelDir)
		}
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

	var nodesToRelease []ag.Node
	defer func() {
		ag.ReleaseGraph(nodesToRelease...)
		runtime.GC()
	}()

	tokenized, err := v.tokenizePrompt(prompt)
	if err != nil {
		return err
	}

	x, s, err := v.encodePrompt(tokenized)
	if err != nil {
		return err
	}
	nodesToRelease = append(nodesToRelease, extractNodesToRelease(x, s)...)

	log.Debug().Msg("Generating text")

	lastGeneratedTokenID := -1
	canPrint := false
loop:
	for i := 0; ; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
			lastGeneratedTokenID, err = v.predictNext(v.Model.Predict(x), false)
			if err != nil {
				return err
			}
			nextToken, err := v.Tokenizer.ReconstructText([]int{lastGeneratedTokenID})
			if err != nil {
				return fmt.Errorf("failed to reconstruct text: %v", err)
			}
			if !canPrint && containsPrintableChar(nextToken) {
				canPrint = true
			}
			if isDone(nextToken, stopOnNewLine, canPrint) {
				break loop
			}
			if canPrint {
				out <- nextToken
			}
			if i >= maxTokens {
				break loop // stop generating
			}
			x, s = v.Model.Encode([]int{lastGeneratedTokenID}, s, false)
			nodesToRelease = append(nodesToRelease, extractNodesToRelease(x, s)...)
		}
	}

	return nil
}

func containsPrintableChar(s string) bool {
	for _, r := range s {
		if unicode.IsPrint(r) {
			return true
		}
	}
	return false
}

func isDone(nextToken string, stopOnNewLine bool, canPrint bool) bool {
	return nextToken == "<|endoftext|>" || (nextToken == "\n" && canPrint && stopOnNewLine)
}

func extractNodesToRelease(x ag.Node, s rwkv.State) []ag.Node {
	nodes := []ag.Node{x}
	for _, layer := range s {
		nodes = append(nodes, layer.FfnXX, layer.AttXX, layer.AttAA, layer.AttBB, layer.AttPP)
	}
	return nodes
}

func (v *VerbaFlow) tokenizePrompt(prompt string) ([]int, error) {
	log.Trace().Msgf("Tokenizing prompt: %s", prompt)
	tokenized, err := v.Tokenizer.Tokenize(prompt)
	if err != nil {
		return nil, err
	}
	log.Debug().Msgf("Token IDs: %v", tokenized)
	return tokenized, nil
}

func (v *VerbaFlow) encodePrompt(tokenized []int) (ag.Node, rwkv.State, error) {
	log.Debug().Msg("Encoding prompt...")
	x, s := v.Model.Encode(tokenized, nil, true)
	x.Value() // wait for the computation to complete
	return x, s, nil
}

func (v *VerbaFlow) predictNext(logits ag.Node, sample bool) (int, error) {
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
