// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package encoder

import (
	"context"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/verbaflow/rwkv"
	"github.com/nlpodyssey/verbaflow/rwkvlm"
)

type Encoder struct {
	model *rwkvlm.Model
}

type Result struct {
	Encoding ag.Node
	State    rwkv.State
}

func New(model *rwkvlm.Model) *Encoder {
	return &Encoder{model: model}
}

func (e *Encoder) Encode(ctx context.Context, tokens []int) (Result, error) {
	x, s := e.model.Encode(ctx, nil, tokens...)
	x.Value() // wait for the value to be computed
	return Result{
		Encoding: x,
		State:    s,
	}, nil
}
