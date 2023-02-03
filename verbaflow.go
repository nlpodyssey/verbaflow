// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package verbaflow

import (
	"context"
	"fmt"
	"os"
	"path/filepath"

	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/verbaflow/decoder"
	"github.com/nlpodyssey/verbaflow/encoder"
	"github.com/nlpodyssey/verbaflow/rwkvlm"
	"github.com/nlpodyssey/verbaflow/tokenizer"
	"github.com/rs/zerolog/log"
)

// VerbaFlow is the core struct of the library.
type VerbaFlow struct {
	Model          *rwkvlm.Model
	Tokenizer      tokenizer.Tokenizer
	embeddingsRepo *diskstore.Repository
}

// Load loads a VerbaFlow model from the given directory.
func Load(modelDir string) (*VerbaFlow, error) {
	tk, err := tokenizer.Load(modelDir)
	if err != nil {
		return nil, err
	}
	model, err := rwkvlm.Load(modelDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("error: unable to find the model file or directory '%s'. Please ensure that the model has been successfully downloaded and converted before trying again", modelDir)
		}
		return nil, err
	}
	embeddingsRepo, err := diskstore.NewRepository(filepath.Join(modelDir, "spago_embedding"), diskstore.ReadOnlyMode)
	if err != nil {
		return nil, fmt.Errorf("failed to load embeddings repository: %w", err)
	}
	err = model.ApplyEmbeddings(embeddingsRepo)
	if err != nil {
		return nil, fmt.Errorf("failed to apply embeddings: %w", err)
	}
	return &VerbaFlow{
		Model:          model,
		Tokenizer:      tk,
		embeddingsRepo: embeddingsRepo,
	}, nil
}

// Close closes the model resources.
func (vf *VerbaFlow) Close() error {
	return vf.embeddingsRepo.Close()
}

// Generate generates a text from the given prompt.
// The "out" channel is used to stream the generated text.
// The generated text will be at most `maxTokens` long (in addition to the prompt).
func (vf *VerbaFlow) Generate(ctx context.Context, prompt string, buffer decoder.ChannelBuffer, opts decoder.DecodingOptions) error {
	log.Trace().Msgf("Tokenizing prompt: %q", prompt)
	tokenized, err := vf.Tokenizer.Tokenize(prompt)
	if err != nil {
		return err
	}

	log.Trace().Msgf("Preprocessing token IDs: %v", tokenized)
	encoderOutput, err := encoder.New(vf.Model).Encode(ctx, tokenized)
	if err != nil {
		return err
	}

	log.Trace().Msg("Generating...")
	err = decoder.New(vf.Model, opts).Decode(ctx, encoderOutput, buffer)
	if err != nil {
		return err
	}

	return nil
}

func (vf *VerbaFlow) TokenByID(id int) (string, error) {
	return vf.Tokenizer.ReconstructText([]int{id})
}
