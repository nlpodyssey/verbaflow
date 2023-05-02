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

// TimeMix is a model that implements the TimeMix component.
type TimeMix struct {
	nn.Module
	Key        *nn.Param
	Value      *nn.Param
	Receptance *nn.Param
	Output     *nn.Param
	TimeDecay  *nn.Param
	TimeFirst  *nn.Param
	TimeMixK   *nn.Param
	TimeMixV   *nn.Param
	TimeMixR   *nn.Param
	Config     Config
}

func init() {
	gob.Register(&TimeMix{})
}

func NewTimeMix[T float.DType](c Config) *TimeMix {
	return &TimeMix{
		Config:     c,
		Key:        nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		Value:      nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		Receptance: nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		Output:     nn.NewParam(mat.NewEmptyDense[T](c.DModel, c.DModel)),
		TimeDecay:  nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeFirst:  nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixK:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixV:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
		TimeMixR:   nn.NewParam(mat.NewEmptyVecDense[T](c.DModel)),
	}
}

// ForwardSingle performs the forward step for a single input.
func (m *TimeMix) ForwardSingle(x ag.Node, state *LayerState) ag.Node {
	xx, aa, bb, pp := state.AttXX, state.AttAA, state.AttBB, state.AttPP

	// Step 1: mix with previous time step.
	xk := ag.Add(ag.Prod(m.TimeMixK, x), ag.Prod(ag.ReverseSubOne(m.TimeMixK), xx))
	xv := ag.Add(ag.Prod(m.TimeMixV, x), ag.Prod(ag.ReverseSubOne(m.TimeMixV), xx))
	xr := ag.Add(ag.Prod(m.TimeMixR, x), ag.Prod(ag.ReverseSubOne(m.TimeMixR), xx))

	k := ag.Mul(m.Key, xk)
	v := ag.Mul(m.Value, xv)
	r := ag.Sigmoid(ag.Mul(m.Receptance, xr))

	// Step 2: calculate the output.
	ww := ag.Add(k, m.TimeFirst)
	p := ag.Max(pp, ww)
	e1 := ag.Exp(ag.Sub(pp, p))
	e2 := ag.Exp(ag.Sub(ww, p))
	a := ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v))
	b := ag.Add(ag.Prod(e1, bb), e2)
	rwkv := ag.Prod(r, ag.Div(a, b))
	out := ag.Mul(m.Output, rwkv)

	// Step 3: update the state.
	ww = ag.Add(pp, m.TimeDecay)
	p = ag.Max(ww, k)
	e1 = ag.Exp(ag.Sub(ww, p))
	e2 = ag.Exp(ag.Sub(k, p))

	state.AttXX = x
	state.AttAA = ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v))
	state.AttBB = ag.Add(ag.Prod(e1, bb), e2)
	state.AttPP = p

	return out
}

// ForwardSequence performs the forward step for a sequence of inputs.
// The state is updated at the end of the sequence.
func (m *TimeMix) ForwardSequence(x []ag.Node, state *LayerState) []ag.Node {
	aa, bb, pp := state.AttAA, state.AttBB, state.AttPP
	xx := append([]ag.Node{state.AttXX}, x[:len(x)-1]...)

	xk := add(prod(m.TimeMixK, x), prod(ag.ReverseSubOne(m.TimeMixK), xx))
	xv := add(prod(m.TimeMixV, x), prod(ag.ReverseSubOne(m.TimeMixV), xx))
	xr := add(prod(m.TimeMixR, x), prod(ag.ReverseSubOne(m.TimeMixR), xx))

	k := mul(m.Key, xk)
	v := mul(m.Value, xv)
	r := sigmoid(mul(m.Receptance, xr))

	wkv := make([]ag.Node, len(r))
	for i := 0; i < len(r); i++ {
		ww := ag.Add(k[i], m.TimeFirst)
		p := ag.Max(pp, ww)
		e1 := ag.Exp(ag.Sub(pp, p))
		e2 := ag.Exp(ag.Sub(ww, p))
		a := ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v[i]))
		b := ag.Add(ag.Prod(e1, bb), e2)
		wkv[i] = ag.Div(a, b)

		// update intermediate values
		ww = ag.Add(pp, m.TimeDecay)
		p = ag.Max(ww, k[i])
		e1 = ag.Exp(ag.Sub(ww, p))
		e2 = ag.Exp(ag.Sub(k[i], p))
		aa = ag.Add(ag.Prod(e1, aa), ag.Prod(e2, v[i]))
		bb = ag.Add(ag.Prod(e1, bb), e2)
		pp = p
	}

	rwkv := prod2(r, wkv)
	out := mul(m.Output, rwkv)

	state.AttXX = x[len(x)-1]
	state.AttAA = aa
	state.AttBB = bb
	state.AttPP = pp

	return out
}
