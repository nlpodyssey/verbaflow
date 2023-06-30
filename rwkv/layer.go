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
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

type Layer struct {
	nn.Module

	LN0 *layernorm.Model
	LN1 *layernorm.Model
	LN2 *layernorm.Model

	ChanMix *ChannelMix
	TimeMix *TimeMix
}

func init() {
	gob.Register(&Layer{})
}

func NewLayer[T float.DType](c Config) *Layer {
	return &Layer{
		LN0:     layernorm.New[T](c.DModel, 1e-6),
		LN1:     layernorm.New[T](c.DModel, 1e-6),
		LN2:     layernorm.New[T](c.DModel, 1e-6),
		ChanMix: NewChannelMix[T](c),
		TimeMix: NewTimeMix[T](c),
	}
}

func (m *Layer) ForwardSingle(x mat.Tensor, state *LayerState) mat.Tensor {
	if m.LN0 != nil {
		x = m.LN0.Forward(x)[0]
	}
	x = ag.Add(x, m.TimeMix.ForwardSingle(m.LN1.Forward(x)[0], state))
	x = ag.Add(x, m.ChanMix.ForwardSingle(m.LN2.Forward(x)[0], state))
	return x
}

func (m *Layer) ForwardSequence(x []mat.Tensor, state *LayerState) []mat.Tensor {
	if m.LN0 != nil {
		x = m.LN0.Forward(x...)
	}
	x = add(x, m.TimeMix.ForwardSequence(m.LN1.Forward(x...), state))
	x = add(x, m.ChanMix.ForwardSequence(m.LN2.Forward(x...), state))
	return x
}
