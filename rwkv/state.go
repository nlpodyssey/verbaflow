// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkv

import (
	"github.com/nlpodyssey/spago/mat"
)

type State []*LayerState

type LayerState struct {
	FfnXX mat.Tensor
	AttXX mat.Tensor
	AttAA mat.Tensor
	AttBB mat.Tensor
	AttPP mat.Tensor
}

func NewState(c Config) State {
	state := make(State, c.NumLayers)
	for i := 0; i < c.NumLayers; i++ {
		state[i] = &LayerState{
			FfnXX: mat.NewDense[float32](mat.WithShape(c.DModel)),
			AttXX: mat.NewDense[float32](mat.WithShape(c.DModel)),
			AttAA: mat.NewDense[float32](mat.WithShape(c.DModel)),
			AttBB: mat.NewDense[float32](mat.WithShape(c.DModel)),
			AttPP: mat.NewDense[float32](mat.WithShape(c.DModel), mat.WithBacking(mat.CreateInitializedSlice[float32](c.DModel, -1e30))),
		}
	}
	return state
}
