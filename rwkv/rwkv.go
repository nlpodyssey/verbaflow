// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkv

import (
	"encoding/gob"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Model implements the RWKV neural network.
type Model struct {
	nn.Module
	Layers []*Layer
	Config Config
}

// Config is the configuration of the RWKV model.
type Config struct {
	DModel       int
	NumLayers    int
	RescaleLayer int
}

func init() {
	gob.Register(&Model{})
}

// New returns a new RWKV model.
func New[T float.DType](c Config) *Model {
	m := &Model{Config: c}
	for i := 0; i < c.NumLayers; i++ {
		m.Layers = append(m.Layers, NewLayer[T](c))
	}
	return m
}

// ForwardSingle performs the forward step for a single element of the sequence.
func (m *Model) ForwardSingle(x mat.Tensor, state State) (mat.Tensor, State) {
	if len(state) == 0 {
		state = NewState(m.Config)
	}
	for i, layer := range m.Layers {
		x = layer.ForwardSingle(x, state[i])

		if (i+1)%m.Config.RescaleLayer == 0 {
			x = ag.ProdScalar(x, mat.Scalar(0.5))
		}
	}
	return x, state
}

// ForwardSequence performs the forward step for the entire sequence, just a bit more optimized.
// It is equivalent to calling ForwardSingle for each element of the sequence, for example:
//
//	var x mat.Tensor
//	for _, e := range encoded {
//		x, s = m.ForwardSingle(e, s)
//	}
//	return x, s
//
// It returns the last computed state.
func (m *Model) ForwardSequence(x []mat.Tensor, state State) ([]mat.Tensor, State) {
	if len(state) == 0 {
		state = NewState(m.Config)
	}
	for i, layer := range m.Layers {
		x = layer.ForwardSequence(x, state[i])

		if (i+1)%m.Config.RescaleLayer == 0 {
			for j := range x {
				x[j] = ag.ProdScalar(x[j], mat.Scalar(0.5))
			}
		}
	}
	return x, state
}
