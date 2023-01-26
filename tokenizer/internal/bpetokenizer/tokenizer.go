// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpetokenizer

import (
	"fmt"
	"path/filepath"
	"strings"

	"github.com/nlpodyssey/gotokenizers/encodings"
	"github.com/nlpodyssey/gotokenizers/models"
	"github.com/nlpodyssey/gotokenizers/models/bpemodel"
	"github.com/nlpodyssey/gotokenizers/normalizedstring"
	"github.com/nlpodyssey/gotokenizers/pretokenizedstring"
	"github.com/nlpodyssey/gotokenizers/pretokenizers/bytelevelpretokenizer"
	"github.com/nlpodyssey/gotokenizers/vocabulary"
)

const (
	defaultCacheCapacity           = 0
	defaultDropout                 = 0.0
	defaultUnknownToken            = ""
	defaultContinuingSubwordPrefix = ""
	defaultEndOfWordSuffix         = ""
	defaultPrefixSpaceEnabled      = false
	defaultOffsetsTrimmingEnabled  = true
	defaultUnknownFusionEnabled    = false
)

// BPETokenizer is a higher-level tokenizer, which includes byte-level pre-tokenization.
type BPETokenizer struct {
	preTokenizer         *bytelevelpretokenizer.ByteLevelPreTokenizer
	model                *bpemodel.BPEModel
	vocab                *vocabulary.Vocabulary
	extraSpecialTokenIDs map[int]string
	ControlTokenIDs      ControlTokensIDs

	StripPaddingTokensDuringTextReconstruction bool
}

type ControlTokensIDs struct {
	EosTokenID           int
	BosTokenID           int
	PadTokenID           int
	DecoderStartTokenID  int
	ExtraSpecialTokenIDs map[int]string
}

// Load returns a BPETokenizer from a file.
func Load(path string, controlTokensIDs ControlTokensIDs) (*BPETokenizer, error) {
	vocabularyFilename := filepath.Join(path, "vocab.json")
	vocab, err := vocabulary.FromJSONFile(vocabularyFilename)
	if err != nil {
		return nil, fmt.Errorf("loading vocabulary from file %s: %w", vocabularyFilename, err)
	}

	mergesFilename := filepath.Join(path, "merges.txt")
	merges, err := bpemodel.MergeMapFromFile(
		mergesFilename,
		vocab,
		len(defaultContinuingSubwordPrefix),
	)
	if err != nil {
		return nil, fmt.Errorf("loading merges from file %s: %w", mergesFilename, err)
	}

	preTokenizer := bytelevelpretokenizer.New(
		bytelevelpretokenizer.DefaultSplittingRegexp,
		defaultPrefixSpaceEnabled,
		defaultOffsetsTrimmingEnabled,
	)

	model := bpemodel.New(
		vocab,
		merges,
		defaultCacheCapacity,
		defaultDropout,
		defaultUnknownToken,
		defaultContinuingSubwordPrefix,
		defaultEndOfWordSuffix,
		defaultUnknownFusionEnabled,
	)

	t := &BPETokenizer{
		preTokenizer:    preTokenizer,
		model:           model,
		vocab:           vocab,
		ControlTokenIDs: controlTokensIDs,
		StripPaddingTokensDuringTextReconstruction: false,
	}
	if controlTokensIDs.ExtraSpecialTokenIDs != nil {
		t.SetExtraSpecialTokens(controlTokensIDs.ExtraSpecialTokenIDs)
	}

	return t, nil
}

func (t *BPETokenizer) SetExtraSpecialTokens(extra map[int]string) {
	t.extraSpecialTokenIDs = extra
}

// Encode converts a text into an encoded tokens representation useful for Transformer architectures.
// It tokenizes using byte-level pre-tokenization and BPE tokenization.
func (t *BPETokenizer) Encode(text string) (*encodings.Encoding, error) {
	pts := pretokenizedstring.FromString(text)

	err := t.preTokenizer.PreTokenize(pts)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer PreTokenize for %s: %w", text, err)
	}

	err = pts.Tokenize(
		func(ns *normalizedstring.NormalizedString) ([]models.Token, error) {
			return t.model.Tokenize(ns.Get())
		},
	)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer Tokenize for %s: %w", text, err)
	}

	encoding, err := pts.IntoEncoding(0, 0)
	if err != nil {
		return nil, fmt.Errorf("BPETokenizer Encoding for %s: %w", text, err)
	}
	return encoding, nil
}

// Tokenize returns the token IDs of the input text applying the EOS pad token.
func (t *BPETokenizer) Tokenize(text string) ([]int, error) {
	encoded, err := t.Encode(text)
	if err != nil {
		return nil, err
	}
	return encoded.IDs, nil
}

// ReconstructText returns the text of the input token IDs removing the padding token.
func (t *BPETokenizer) ReconstructText(tokenIds []int) (string, error) {
	if !t.StripPaddingTokensDuringTextReconstruction {
		return t.internalDetokenize(tokenIds), nil
	}

	stripPaddingTokensFn := func(tokenIds []int) []int {
		result := make([]int, 0, len(tokenIds))
		for _, id := range tokenIds {
			if id == t.ControlTokenIDs.EosTokenID || id == t.ControlTokenIDs.PadTokenID || id == t.ControlTokenIDs.BosTokenID || id == t.ControlTokenIDs.DecoderStartTokenID {
				continue
			}
			result = append(result, id)
		}
		return result
	}

	return t.internalDetokenize(stripPaddingTokensFn(tokenIds)), nil
}

// Detokenize flatten and merges a list of ids into a single string.
// TODO: handle proper detokenization
func (t *BPETokenizer) internalDetokenize(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if s, ok := t.extraSpecialTokenIDs[id]; ok {
			sb.WriteString(s)
			continue
		}

		if s, ok := t.vocab.GetString(id); ok {
			sb.WriteString(s)
		}
	}
	out := sb.String()
	out = strings.Replace(out, "Ġ", " ", -1)
	out = strings.Replace(out, "Ċ", "\n", -1)
	return out
}
