// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkv

import (
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
)

type State []*LayerState

type LayerState struct {
	FfnXX ag.Node
	AttXX ag.Node
	AttAA ag.Node
	AttBB ag.Node
	AttPP ag.Node
}

func NewState(c Config) State {
	state := make(State, c.NumLayers)
	for i := 0; i < c.NumLayers; i++ {
		state[i] = &LayerState{
			FfnXX: mat.NewEmptyVecDense[float32](c.DModel),
			AttXX: mat.NewEmptyVecDense[float32](c.DModel),
			AttAA: mat.NewEmptyVecDense[float32](c.DModel),
			AttBB: mat.NewEmptyVecDense[float32](c.DModel),
			AttPP: mat.NewInitVecDense[float32](c.DModel, -1e30),
		}
	}
	return state
}
