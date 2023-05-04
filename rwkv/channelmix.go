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

var _ nn.Model = &TimeMix{}

// ChannelMix implements the channel mix module.
type ChannelMix struct {
	nn.Module
	Key        *nn.Param
	Value      *nn.Param
	Receptance *nn.Param
	TimeMixK   *nn.Param
	TimeMixR   *nn.Param
}

func init() {
	gob.Register(&ChannelMix{})
}

func NewChannelMix[T float.DType](c Config) *ChannelMix {
	hidden := 4 * c.DModel
	return &ChannelMix{
		Key:        nn.NewParam(mat.NewEmptyDense[T](hidden, c.DModel)),
		Value:      nn.NewParam(mat.NewEmptyDense[T](c.DModel, hidden)),
		Receptance: nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		TimeMixK:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixR:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
	}
}

// ForwardSingle performs the forward step for a single node.
func (m *ChannelMix) ForwardSingle(x ag.Node, state *LayerState) (rkv ag.Node) {
	xx := state.FfnXX

	xk := ag.Add(ag.Prod(x, m.TimeMixK), ag.Prod(ag.ReverseSubOne(m.TimeMixK), xx))
	xr := ag.Add(ag.Prod(x, m.TimeMixR), ag.Prod(ag.ReverseSubOne(m.TimeMixR), xx))

	k := ag.Mul(m.Key, xk)
	k = ag.Square(ag.ReLU(k))
	kv := ag.Mul(m.Value, k)
	r := ag.Sigmoid(ag.Mul(m.Receptance, xr))
	rkv = ag.Prod(r, kv)

	state.FfnXX = x
	return
}

// ForwardSequence performs the forward step for a sequence of nodes.
// The state is updated with the last node of the sequence.
func (m *ChannelMix) ForwardSequence(x []ag.Node, state *LayerState) (rkv []ag.Node) {
	xx := append([]ag.Node{state.FfnXX}, x[:len(x)-1]...) // token shift

	xk := add(prod(m.TimeMixK, x), prod(ag.ReverseSubOne(m.TimeMixK), xx))
	xr := add(prod(m.TimeMixR, x), prod(ag.ReverseSubOne(m.TimeMixR), xx))

	k := mul(m.Key, xk)
	k = square(relu(k))
	kv := mul(m.Value, k)
	r := sigmoid(mul(m.Receptance, xr))
	rkv = prod2(r, kv)

	state.FfnXX = x[len(x)-1]
	return
}
