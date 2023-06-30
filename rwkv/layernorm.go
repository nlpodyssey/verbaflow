// Copyright 2019 spaGO Authors. All rights reserved.
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

var _ nn.Model = &Model{}

type LayerNormModel struct {
	nn.Module
	W   *nn.Param
	B   *nn.Param
	Eps *nn.Buffer
}

func init() {
	gob.Register(&Model{})
}

func NewLayerNorm[T float.DType](size int, eps float64) *LayerNormModel {
	return &LayerNormModel{
		W:   nn.NewParam(mat.NewDense[T](mat.WithShape(size))),
		B:   nn.NewParam(mat.NewDense[T](mat.WithShape(size))),
		Eps: nn.Buf(mat.Scalar[T](T(eps))),
	}
}

// Forward performs the forward step for each input node and returns the result.
// y = (x - E\[x\]) / sqrt(VAR\[x\] + [EPS]) * g + b
func (m *LayerNormModel) Forward(xs ...mat.Tensor) []mat.Tensor {
	if len(xs) == 0 {
		return nil
	}
	out := make([]mat.Tensor, len(xs))
	for i, x := range xs {
		mean := ag.ReduceMean(x)
		dev := ag.SubScalar(x, mean)
		stdDev := ag.Sqrt(ag.Add(ag.ReduceMean(ag.Square(dev)), m.Eps))
		out[i] = ag.Add(ag.Prod(ag.DivScalar(dev, stdDev), m.W), m.B)
	}
	return out
}
