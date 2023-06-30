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

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
	"github.com/nlpodyssey/verbaflow/rwkv"
	"github.com/rs/zerolog/log"
)

type Model struct {
	nn.Module
	Embeddings *embedding.Model
	Encoder    *rwkv.Model
	LN         *layernorm.Model
	Linear     *nn.Param
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

func New[T float.DType](c Config) *Model {
	return &Model{
		Config: c,
		Encoder: rwkv.New[T](rwkv.Config{
			DModel:       c.DModel,
			NumLayers:    c.NumHiddenLayers,
			RescaleLayer: c.RescaleLayer,
		}),
		LN:         layernorm.New[T](c.DModel, 1e-6),
		Linear:     nn.NewParam(mat.NewDense[T](mat.WithShape(c.VocabSize, c.DModel))),
		Embeddings: embedding.New[T](c.VocabSize, c.DModel),
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

// Encode performs EncodeTokens and EncodeEmbeddings.
func (m *Model) Encode(ctx context.Context, s rwkv.State, tokens ...int) (mat.Tensor, rwkv.State) {
	encoded, err := m.Embeddings.Encode(tokens)
	if err != nil {
		log.Fatal().Msgf("failed to encode tokens: %w", err)
	}
	return m.EncodeEmbeddings(ctx, s, encoded)
}

// EncodeTokens returns the embeddings of the given tokens.
func (m *Model) EncodeTokens(_ context.Context, tokens ...int) []mat.Tensor {
	encoded, err := m.Embeddings.Encode(tokens)
	if err != nil {
		log.Fatal().Msgf("failed to encode tokens: %w", err)
	}
	return encoded
}

// EncodeEmbeddings returns the encoding of the given input considering the last state.
// At least one token is required, otherwise can panic.
// If the input is a sequence, the last state is returned.
func (m *Model) EncodeEmbeddings(_ context.Context, s rwkv.State, xs []mat.Tensor) (mat.Tensor, rwkv.State) {
	if len(xs) == 1 {
		return m.Encoder.ForwardSingle(xs[0], s)
	}

	log.Trace().Msgf("Encoding sequence of %d tokens...", len(xs))
	var h []mat.Tensor
	h, s = m.Encoder.ForwardSequence(xs, s)
	return h[len(h)-1], s
}

// Predict returns the prediction logits of the next token.
func (m *Model) Predict(x mat.Tensor) mat.Tensor {
	return ag.Mul(m.Linear, m.LN.Forward(x)[0])
}
