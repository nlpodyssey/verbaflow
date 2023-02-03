// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkvlm

import (
	"encoding/gob"

	"github.com/nlpodyssey/spago/ag"
	emb "github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
)

// Embeddings embeds the token embeddings.
type Embeddings struct {
	nn.Module
	Tokens *emb.Model[int]
	Config Config
}

func init() {
	gob.Register(&Embeddings{})
}

// NewEmbeddings returns a new embedding module.
func NewEmbeddings[T float.DType](c emb.Config, repo store.Repository) *Embeddings {
	return &Embeddings{
		Tokens: emb.New[T, int](c, repo),
	}
}

// Encode performs the input encoding.
func (m *Embeddings) Encode(tokens []int) []ag.Node {
	return m.Tokens.Encode(tokens)
}
