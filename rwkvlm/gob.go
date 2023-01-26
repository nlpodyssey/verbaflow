// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkvlm

import (
	"bufio"
	"encoding/gob"
	"io"
	"os"

	"github.com/nlpodyssey/rwkv"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

func gobEncode(obj *Model, w io.Writer) error {
	bw := bufio.NewWriter(w)
	encoder := gob.NewEncoder(bw)

	for _, chunk := range getChunksForGobEncoding(obj) {
		if err := encoder.Encode(chunk); err != nil {
			return err
		}
		if err := bw.Flush(); err != nil {
			return err
		}
	}
	return nil
}

func getChunksForGobEncoding(obj *Model) []interface{} {
	chunks := []interface{}{
		obj.Config,
		obj.Embeddings,
		obj.LN,
		obj.Linear.(*nn.BaseParam),
		obj.Encoder.Config,
	}
	for _, layer := range obj.Encoder.Layers {
		chunks = append(chunks, layer)
	}
	return chunks
}

// loadFromFile uses Gob to deserialize objects files to memory.
// See gobDecoding for further details.
func loadFromFile(filename string) (*Model, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	return gobDecoding(f)
}

func gobDecoding(r io.Reader) (*Model, error) {
	obj := &Model{
		LN:      &layernorm.Model{},
		Linear:  &nn.BaseParam{},
		Encoder: &rwkv.Model{},
	}

	br := bufio.NewReader(r)
	decoder := gob.NewDecoder(br)

	w := nn.BaseParam{}

	if err := decoder.Decode(&obj.Config); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&obj.Embeddings); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&obj.LN); err != nil {
		return nil, err
	}
	if err := decoder.Decode(&w); err != nil {
		return nil, err
	}
	obj.Linear = &w
	if err := decoder.Decode(&obj.Encoder.Config); err != nil {
		return nil, err
	}

	obj.Encoder.Layers = make([]*rwkv.Layer, obj.Config.NumHiddenLayers)
	for i := range obj.Encoder.Layers {
		if err := decoder.Decode(&obj.Encoder.Layers[i]); err != nil {
			return nil, err
		}
	}

	return obj, nil
}
