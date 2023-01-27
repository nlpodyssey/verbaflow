// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkvlm

import (
	"encoding/gob"
	"encoding/json"
	"os"
	"path/filepath"

	"github.com/nlpodyssey/rwkv"
	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/embeddings"
	"github.com/nlpodyssey/spago/embeddings/store"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
)

type Model struct {
	nn.Module
	Embeddings *Embeddings
	Encoder    *rwkv.Model
	LN         *layernorm.Model
	Linear     nn.Param `spago:"type:weights"`
	Config     Config
}

type Config struct {
	DModel              int    `json:"d_model"`
	NumHiddenLayers     int    `json:"num_hidden_layers"`
	RescaleLayer        int    `json:"rescale_layer"`
	VocabSize           int    `json:"vocab_size"`
	EmbeddingsStoreName string `json:"embeddings_store_name"`
}

func LoadConfig(filePath string) (Config, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return Config{}, err
	}
	defer file.Close()

	var config Config
	jsonDecoder := json.NewDecoder(file)
	if err := jsonDecoder.Decode(&config); err != nil {
		return Config{}, err
	}
	return config, nil
}

func init() {
	gob.Register(&Model{})
}

func New[T float.DType](c Config, repo store.Repository) *Model {
	return &Model{
		Config: c,
		Encoder: rwkv.New[T](rwkv.Config{
			DModel:       c.DModel,
			NumLayers:    c.NumHiddenLayers,
			RescaleLayer: c.RescaleLayer,
		}),
		LN:         layernorm.New[T](c.DModel, 1e-6),
		Linear:     nn.NewParam(mat.NewEmptyDense[T](c.VocabSize, c.DModel)),
		Embeddings: NewEmbeddings[T](c, repo),
	}
}

// Load loads a pre-trained model from the given path.
func Load(dir string) (*Model, error) {
	m, err := loadFromFile(filepath.Join(dir, DefaultOutputFilename))
	if err != nil {
		return nil, err
	}
	return m, nil
}

// Dump saves the Model to a file.
// See gobEncode for further details.
func Dump(obj *Model, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = e
		}
	}()
	if err = gobEncode(obj, f); err != nil {
		return err
	}
	return nil
}

// ApplyEmbeddings sets the embeddings of the model.
func (m *Model) ApplyEmbeddings(repo *diskstore.Repository) (err error) {
	nn.Apply(m, func(model nn.Model, name string) {
		switch em := model.(type) {
		case *embeddings.Model[[]byte], *embeddings.Model[int], *embeddings.Model[string]:
			if e := em.(interface {
				UseRepository(repo store.Repository) error
			}).UseRepository(repo); e != nil && err == nil {
				err = e
			}
		}
	})
	return err
}

// Encode returns the encoding of the given input considering the last state.
func (m *Model) Encode(context []int, s rwkv.State, encodeFullSequence bool) (ag.Node, rwkv.State) {
	if encodeFullSequence {
		// transform the context into a sequence of embeddings
		encoded := m.Embeddings.Encode(context)
		var x ag.Node
		for _, e := range encoded {
			x, s = m.Encoder.Forward(e, s)
		}
		return x, s
	}

	// encode only the last token
	x := m.Embeddings.Encode(context[len(context)-1:])[0]
	return m.Encoder.Forward(x, s)
}

// Predict returns the prediction logits of the next token.
func (m *Model) Predict(x ag.Node) ag.Node {
	return ag.Mul(m.Linear, m.LN.Forward(x)[0])
}
