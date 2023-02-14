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
	MaxLen int `json:"max_len" yaml:"max_len"`
	// MinLen is the minimum number of tokens to generate.
	MinLen int `json:"min_len" yaml:"min_len"`
	// StopSequencesIDs is a list of token ids that if generated, the generation process will stop.
	StopSequencesIDs [][]int `json:"stop_sequences_ids" yaml:"stop_sequences_ids"`
	// EndTokenID is the end-of-sequence token (default: 0).
	EndTokenID int `json:"end_token_id" yaml:"end_token_id"`
	// SkipEndTokenID when true, the end token is not added to the generated sequence.
	SkipEndTokenID bool `json:"skip_end_token_id" yaml:"skip_end_token_id"`
	// Temperature is the temperature used to control the randomness of the generated text.
	Temp float64 `json:"temp" yaml:"temp"`
	// TopK is the number of tokens to consider when sampling the next token.
	TopK int `json:"top_k" yaml:"top_k"`
	// TopP is the cumulative probability of the tokens to consider when sampling the next token.
	TopP float64 `json:"top_p" yaml:"top_p"`
	// UseSampling uses sampling to generate the next token.
	UseSampling bool `json:"use_sampling" yaml:"use_sampling"`
}

// GeneratedToken is the result of a single step of the decoder.
type GeneratedToken struct {
	// TokenID is the ID of the token predicted by the decoder at the current step.
	TokenID int
	// SumNegLogProbs is the sum of the negative log probabilities up to the current step.
	SumNegLogProbs float64
}

func New(m *rwkvlm.Model, opts DecodingOptions) (*Decoder, error) {
	dc, err := OutputDiversityControl(opts.Temp, opts.TopK, opts.TopP)
	if err != nil {
		return nil, err
	}
	return &Decoder{
		model:              m,
		opts:               opts,
		applyOutputControl: dc,
		applySelection:     OutputSelection(opts.UseSampling),
	}, nil
}

func (d *Decoder) Decode(ctx context.Context, input encoder.Result, chGen chan GeneratedToken) error {
	defer close(chGen)

	if input.Encoding == nil || input.State == nil {
		return fmt.Errorf("invalid input: hidden representation and state are required")
	}

	// free the computational graph after the generation is finished
	nt := &ag.NodesTracker{}
	defer nt.ReleaseNodes()

	x, s := input.Encoding, input.State

	var sequence []int
	var sumNegLogProbs float64

Loop:
	for i := 0; ; i++ {
		select {
		case <-ctx.Done():
			log.Trace().Msgf("Generation cancelled after %d steps due to context cancellation", i)
			break Loop
		default:
			tokenID, tokenScore, err := d.generateToken(ctx, x, i, nt)
			if err != nil {
				return err
			}
			sequence = append(sequence, tokenID)
			sumNegLogProbs -= math.Log(tokenScore)

			chGen <- GeneratedToken{
				TokenID:        tokenID,
				SumNegLogProbs: sumNegLogProbs,
			}

			if d.checkStopConditions(sequence) {
				break Loop
			}

			// update the hidden representation `x` with the result of encoding the last generated token,
			// which is used as input for the next iteration of the loop.
			x, err = d.encode(ctx, nt, tokenID, s)
			if err != nil {
				return err
			}
		}
	}

	log.Trace().Msgf("[%.2f] Generated token IDs: %v", sumNegLogProbs, sequence)

	return nil
}

// generateToken performs a single step of the decoding process.
// It returns the selected output token ID and its score.
func (d *Decoder) generateToken(_ context.Context, x ag.Node, seqLen int, nt *ag.NodesTracker) (int, float64, error) {
	logits := nt.TrackNode(d.model.Predict(x))
	candidates, err := d.applyOutputControl(d.adjustLogits(logits.Value(), seqLen))
	if err != nil {
		return 0, 0, err
	}
	return d.applySelection(candidates)
}

// adjustLogits checks if the sequence is too short and if so, set the logits of the end token to a very low value.
func (d *Decoder) adjustLogits(logits mat.Matrix, sequenceLength int) mat.Matrix {
	if sequenceLength >= d.opts.MinLen {
		return logits
	}
	log.Trace().Msgf("Sequence too short (%d), setting end token (%d) logits to -inf", sequenceLength, d.opts.EndTokenID)
	logits.SetVecScalar(d.opts.EndTokenID, floatNegInf)
	return logits
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
			log.Trace().Msgf("Reached stop sequence %v", stopSeq)
			return true
		}
	}
	return false
}

func (d *Decoder) encode(ctx context.Context, nt *ag.NodesTracker, tokenID int, state rwkv.State) (ag.Node, error) {
	x, s := d.model.Encode(ctx, []int{tokenID}, state)
	nt.TrackNodes(waitForNodes(extractNodesToRelease(x, s))...)
	return x, nil
}

// waitForNodes waits for the nodes to be computed.
// It is used to ensure that the nodes are computed before releasing them.
func waitForNodes(nodes []ag.Node) []ag.Node {
	for _, n := range nodes {
		n.Value()
	}
	return nodes
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
