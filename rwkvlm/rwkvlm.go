// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkvlm

import (
	"context"
	"encoding/gob"
	"encoding/json"
	"fmt"
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
	"github.com/rs/zerolog/log"
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
	// DModel primarily corresponds to the embedding size.
	//
	// When converting a torch model, it can be left zero, letting the
	// process deduce the value automatically.
	DModel int `json:"d_model"`
	// NumHiddenLayers is the number of hidden layers.
	//
	// When converting a torch model, it can be left zero, letting the
	// process deduce the value automatically.
	NumHiddenLayers int `json:"num_hidden_layers"`
	// VocabSize is the vocabulary size.
	//
	// When converting a torch model, it can be left zero, letting the
	// process deduce the value automatically.
	VocabSize           int    `json:"vocab_size"`
	RescaleLayer        int    `json:"rescale_layer"`
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
		LN:     layernorm.New[T](c.DModel, 1e-6),
		Linear: nn.NewParam(mat.NewEmptyDense[T](c.VocabSize, c.DModel)),
		Embeddings: NewEmbeddings[T](embeddings.Config{
			Size:      c.DModel,
			StoreName: c.EmbeddingsStoreName,
			Trainable: false,
		}, repo),
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
		return fmt.Errorf("failed to open model dump file %q for writing: %w", filename, err)
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = fmt.Errorf("failed to close model dump file %q: %w", filename, e)
		}
	}()
	if err = gobEncode(obj, f); err != nil {
		return fmt.Errorf("failed to encode model dump: %w", err)
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
// At least one token is required, otherwise can panic.
// If the input is a sequence, the last state is returned.
func (m *Model) Encode(_ context.Context, tokens []int, s rwkv.State) (ag.Node, rwkv.State) {
	x := m.Embeddings.Encode(tokens)
	if len(x) == 1 {
		return m.Encoder.ForwardSingle(x[0], s)
	}

	log.Trace().Msgf("Encoding sequence of %d tokens...", len(x))
	var h []ag.Node
	h, s = m.Encoder.ForwardSequence(x, s)
	return h[len(h)-1], s
}

// Predict returns the prediction logits of the next token.
func (m *Model) Predict(x ag.Node) ag.Node {
	return ag.Mul(m.Linear, m.LN.Forward(x)[0])
}
