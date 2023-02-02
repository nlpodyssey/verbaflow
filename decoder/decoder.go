// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decoder

import (
	"context"
	"fmt"
	"math"
	"reflect"

	"github.com/nlpodyssey/rwkv"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/verbaflow/encoder"
	"github.com/nlpodyssey/verbaflow/rwkvlm"
	"github.com/rs/zerolog/log"
)

var floatNegInf = float.Interface(math.Inf(-1))

type Decoder struct {
	model              *rwkvlm.Model
	applyOutputControl OutputDiversityControlFunc
	applySelection     OutputSelectionFunc
	opts               DecodingOptions
}

// DecodingOptions contains the options for the conditional text generation.
type DecodingOptions struct {
	// MaxLen is the maximum number of tokens to generate.
	MaxLen int
	// MinLen is the minimum number of tokens to generate.
	MinLen int
	// StopSequencesIDs is a list of token ids that if generated, the generation process will stop.
	StopSequencesIDs [][]int
	// EndTokenID is the end-of-sequence token (default: 0).
	EndTokenID int
	// Temperature is the temperature used to control the randomness of the generated text.
	Temp float64
	// TopK is the number of tokens to consider when sampling the next token.
	TopK int
	// TopP is the cumulative probability of the tokens to consider when sampling the next token.
	TopP float64
	// UseSampling uses sampling to generate the next token.
	UseSampling bool
	// BadWordsIds is a list of token ids that are not allowed to be generated.
	BadWordsIDs [][]int
	// EndThreshold is the minimum score that the EOS token must achieve to stop the text generation process, regardless of other higher-scored tokens.
	EndThreshold float64
}

func New(m *rwkvlm.Model, opts DecodingOptions) *Decoder {
	return &Decoder{
		model:              m,
		applyOutputControl: OutputDiversityControl(opts.Temp, opts.TopK, opts.TopP),
		applySelection:     OutputSelection(opts.UseSampling),
		opts:               opts,
	}
}

type TokenSequence = []int

type Result struct {
	// Sequence is a list of generated tokens ids.
	Sequence []int
	// Score is the sum of the negative log probabilities of the generated tokens.
	Score float64
}

func (d *Decoder) Decode(ctx context.Context, input encoder.Result) (*Result, error) {
	if input.HiddenRepresentation == nil || input.State == nil {
		return nil, fmt.Errorf("invalid input: hidden representation and state are required")
	}

	nt := &ag.NodesTracker{}
	defer nt.ReleaseNodes()

	x, s := input.HiddenRepresentation, input.State

	var sequence []int
	var sumNegLogProbs float64

Loop:
	for i := 0; ; i++ {
		select {
		case <-ctx.Done():
			x.Value() // wait for the computation to finish
			break Loop
		default:
			logits, err := d.predict(ctx, nt, x)
			if err != nil {
				return nil, err
			}
			candidates, err := d.applyOutputControl(d.adjustLogits(logits.Value(), len(sequence)))
			if err != nil {
				return nil, err
			}
			selectedOutput, selectedOutputScore, err := d.applySelection(candidates)
			if err != nil {
				return nil, err
			}
			sequence = append(sequence, selectedOutput)
			sumNegLogProbs += -math.Log(selectedOutputScore)

			if stopGeneration := d.checkStopConditions(sequence); stopGeneration {
				break Loop
			}
			x, err = d.encode(ctx, nt, selectedOutput, s)
			if err != nil {
				return nil, err
			}
		}
	}

	sequence = d.removeEndTokenID(sequence)

	return &Result{
		Sequence: sequence,
		Score:    sumNegLogProbs,
	}, nil
}

// adjustLogits checks if the sequence is too short and if so, set the logits of the end token to a very low value.
func (d *Decoder) adjustLogits(logits mat.Matrix, sequenceLength int) mat.Matrix {
	if sequenceLength >= d.opts.MinLen {
		return logits
	}
	logits.SetVecScalar(d.opts.EndTokenID, floatNegInf)
	return logits
}

// removeEndTokenID removes the end token ID from the sequence if present.
func (d *Decoder) removeEndTokenID(sequence []int) []int {
	if len(sequence) == 0 {
		return sequence
	}
	if sequence[len(sequence)-1] == d.opts.EndTokenID {
		return sequence[:len(sequence)-1]
	}
	return sequence
}

func (d *Decoder) checkStopConditions(sequence []int) bool {
	if len(sequence) >= d.opts.MaxLen {
		log.Trace().Msgf("Reached max length (%d)", d.opts.MaxLen)
		return true
	}
	last := sequence[len(sequence)-1]
	if last == d.opts.EndTokenID {
		log.Trace().Msgf("Reached end token (%d)", d.opts.EndTokenID)
		return true
	}
	if len(sequence) >= d.opts.MinLen && hasStopSequence(sequence, d.opts.StopSequencesIDs) {
		log.Trace().Msgf("Reached stop sequence (%v)", d.opts.StopSequencesIDs)
		return true
	}
	return false
}

func hasStopSequence(sequence []int, stopSequences [][]int) bool {
	for _, stopSeq := range stopSequences {
		if len(sequence) < len(stopSeq) {
			continue
		}
		if reflect.DeepEqual(stopSeq, sequence[len(sequence)-len(stopSeq):]) {
			log.Trace().Msgf("Stopping generation because the stop sequence '%q' was found", stopSeq)
			return true
		}
	}
	return false
}

func (d *Decoder) predict(_ context.Context, nt *ag.NodesTracker, x ag.Node) (ag.Node, error) {
	return nt.TrackNode(d.model.Predict(x)), nil
}

func (d *Decoder) encode(ctx context.Context, nt *ag.NodesTracker, tokenID int, state rwkv.State) (ag.Node, error) {
	x, s := d.model.Encode(ctx, []int{tokenID}, state, false)
	nt.TrackNodes(extractNodesToRelease(x, s)...)
	return x, nil
}

// extractNodesToRelease extracts the nodes to release from the states.
// It also considers the explicit x node.
func extractNodesToRelease(x ag.Node, s rwkv.State) []ag.Node {
	nodes := []ag.Node{x}
	for _, layer := range s {
		nodes = append(nodes, layer.FfnXX, layer.AttXX, layer.AttAA, layer.AttBB, layer.AttPP)
	}
	return nodes
}
