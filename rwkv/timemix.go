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
		Key:        nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel, c.DModel))),
		Value:      nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel, c.DModel))),
		Receptance: nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel, c.DModel))),
		Output:     nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel, c.DModel))),
		TimeDecay:  nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel))),
		TimeFirst:  nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel))),
		TimeMixK:   nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel))),
		TimeMixV:   nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel))),
		TimeMixR:   nn.NewParam(mat.NewDense[T](mat.WithShape(c.DModel))),
	}
}

// ForwardSingle performs the forward step for a single input.
func (m *TimeMix) ForwardSingle(x mat.Tensor, state *LayerState) mat.Tensor {
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

func (m *TimeMix) ForwardSequence(x []mat.Tensor, state *LayerState) []mat.Tensor {
	xx := append([]mat.Tensor{state.AttXX}, x[:len(x)-1]...)

	xk, xv, xr := m.computeIntermediateValues(x, xx)
	k, v, r := m.computeKeyValuesReceptance(xk, xv, xr)

	wkv, newAA, newBB, newPP := m.updateAttentionScores(state.AttAA, state.AttBB, state.AttPP, k, v)

	out := mul(m.Output, prod2(r, wkv))

	updateState(state, x, newAA, newBB, newPP)

	return out
}

func (m *TimeMix) computeIntermediateValues(x, xx []mat.Tensor) (xk, xv, xr []mat.Tensor) {
	xk = add(prod(m.TimeMixK, x), prod(ag.ReverseSubOne(m.TimeMixK), xx))
	xv = add(prod(m.TimeMixV, x), prod(ag.ReverseSubOne(m.TimeMixV), xx))
	xr = add(prod(m.TimeMixR, x), prod(ag.ReverseSubOne(m.TimeMixR), xx))
	return
}

func (m *TimeMix) computeKeyValuesReceptance(xk, xv, xr []mat.Tensor) (k, v, r []mat.Tensor) {
	k = mul(m.Key, xk)
	v = mul(m.Value, xv)
	r = sigmoid(mul(m.Receptance, xr))
	return
}

func (m *TimeMix) updateAttentionScores(aa, bb, pp mat.Tensor, k, v []mat.Tensor) (wkv []mat.Tensor, newAA, newBB, newPP mat.Tensor) {
	wkv = make([]mat.Tensor, len(k))
	newAA, newBB, newPP = aa, bb, pp
	for i := 0; i < len(k); i++ {
		wkv[i], newAA, newBB, newPP = m.singleStepAttention(newAA, newBB, newPP, k[i], v[i])
	}
	return
}

func (m *TimeMix) singleStepAttention(aa, bb, pp, ki, vi mat.Tensor) (wkv, newAA, newBB, newPP mat.Tensor) {
	calcExp := func(x, y mat.Tensor) (mat.Tensor, mat.Tensor) {
		p := ag.Max(x, y)
		return ag.Exp(ag.Sub(x, p)), ag.Exp(ag.Sub(y, p))
	}

	ww := ag.Add(ki, m.TimeFirst)
	e1, e2 := calcExp(pp, ww)
	a := ag.Add(ag.Prod(e1, aa), ag.Prod(e2, vi))
	b := ag.Add(ag.Prod(e1, bb), e2)
	wkv = ag.Div(a, b)

	ww = ag.Add(pp, m.TimeDecay)
	e1, e2 = calcExp(ww, ki)
	newAA = ag.Add(ag.Prod(e1, aa), ag.Prod(e2, vi))
	newBB = ag.Add(ag.Prod(e1, bb), e2)
	newPP = ag.Max(ww, ki)

	return
}

func updateState(state *LayerState, x []mat.Tensor, aa, bb, pp mat.Tensor) {
	state.AttXX = x[len(x)-1]
	state.AttAA = aa
	state.AttBB = bb
	state.AttPP = pp
}
