// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package tokenizer

import "github.com/nlpodyssey/verbaflow/tokenizer/internal/bpetokenizer"

// Tokenizer is the interface that wraps the basic tokenizers methods.
type Tokenizer interface {
	// Tokenize returns the sequence of token IDs for the given text.
	Tokenize(text string) ([]int, error)
	// ReconstructText returns the text corresponding to the given sequence of token IDs.
	ReconstructText(ids []int) (string, error)
}

// Load loads a tokenizer from the given path.
func Load(path string) (Tokenizer, error) {
	tk, err := bpetokenizer.Load(path, bpetokenizer.ControlTokensIDs{})
	if err != nil {
		return nil, err
	}
	return tk, nil
}
