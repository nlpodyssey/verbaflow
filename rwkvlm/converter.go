// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkvlm

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/embeddings/store/diskstore"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const (
	DefaultPyModelFilename = "pytorch_model.pt"
	DefaultOutputFilename  = "spago_model.bin"
)

type ConverterConfig struct {
	// The path to the directory where the models will be read from and written to.
	ModelDir string
	// The path to the input model file (default "pytorch_model.pt")
	PyModelFilename string
	// The path to the output model file (default "spago_model.pt")
	GoModelFilename string
	// If true, overwrite the model file if it already exists (default "false")
	OverwriteIfExist bool
}

type mappingParam struct {
	value   mat.Matrix
	matched bool
}

// ConvertPickledModelToRWKVLM converts a PyTorch model to a RWKVLM model.
// It expects a configuration file "config.json" in the same directory as the model file containing the model configuration.
func ConvertPickledModelToRWKVLM[T float.DType](config *ConverterConfig) error {
	if config.PyModelFilename == "" {
		config.PyModelFilename = DefaultPyModelFilename
	}
	if config.GoModelFilename == "" {
		config.GoModelFilename = DefaultOutputFilename
	}

	outputFilename := filepath.Join(config.ModelDir, config.GoModelFilename)

	if info, err := os.Stat(outputFilename); !config.OverwriteIfExist && err == nil && !info.IsDir() {
		log.Debug().Str("model", outputFilename).Msg("Model file already exists, skipping conversion")
		return nil
	}

	modelConfig, err := LoadConfig(filepath.Join(config.ModelDir, "config.json"))
	if err != nil {
		return err
	}

	pyParams, err := extractPyTorchModelParamsFromPickleFile[T](filepath.Join(config.ModelDir, config.PyModelFilename))
	if err != nil {
		return err
	}

	repo, err := diskstore.NewRepository(filepath.Join(config.ModelDir, "repo"), diskstore.ReadWriteMode)
	if err != nil {
		panic(err)
	}
	defer func() {
		err = repo.Close()
		if err != nil {
			panic(err)
		}
	}()
	if err := repo.DropAll(); err != nil {
		panic(err)
	}

	model := New[T](modelConfig, repo)
	params := mapPyTorchParamsToRWKVLM(model)

	mapping := make(map[string]*mappingParam)
	for k, v := range params {
		mapping[k] = &mappingParam{value: v, matched: false}
	}

	{
		// Embeddings
		source := pyParams["emb.weight"]
		size := model.Config.DModel
		for i := 0; i < modelConfig.VocabSize; i++ {
			item, _ := model.Embeddings.Tokens.Embedding(i)
			item.ReplaceValue(mat.NewVecDense[T](source[i*size : (i+1)*size]))
		}
		runtime.GC()
	}

	// All the other parameters
	for name, value := range pyParams {
		param, ok := mapping[name]
		if !ok {
			continue
		}
		if param.value.Size() != len(value) {
			return fmt.Errorf("Error setting %s: dim mismatch", name)
		}
		mat.SetData[T](param.value, value)
		param.matched = true
	}

	if zerolog.GlobalLevel() <= zerolog.DebugLevel {
		for key, value := range mapping {
			if !value.matched {
				log.Debug().Str("parameter", key).Msg("Parameter not initialized")
			}
		}
		for name := range pyParams {
			if name == "emb.weight" {
				continue // skip embeddings
			}
			if _, ok := mapping[name]; !ok {
				log.Debug().Str("parameter", name).Msg("Parameter not mapped")
			}
		}
	}

	fmt.Printf("Serializing model to \"%s\"... ", outputFilename)
	runtime.GC()
	err = Dump(model, outputFilename)
	if err != nil {
		return err
	}
	runtime.GC()
	fmt.Println("Done.")

	return nil
}

func mapPyTorchParamsToRWKVLM(m *Model) map[string]mat.Matrix {
	params := make(map[string]mat.Matrix)

	// RWKV parameters
	for i := 0; i < m.Config.NumHiddenLayers; i++ {
		layer := m.Encoder.Layers[i]
		prefix := fmt.Sprintf("blocks.%d", i)

		if i == 0 {
			params[fmt.Sprintf("%s.ln0.weight", prefix)] = layer.LN0.W.Value()
			params[fmt.Sprintf("%s.ln0.bias", prefix)] = layer.LN0.B.Value()
		}

		params[fmt.Sprintf("%s.ln1.weight", prefix)] = layer.LN1.W.Value()
		params[fmt.Sprintf("%s.ln1.bias", prefix)] = layer.LN1.B.Value()
		params[fmt.Sprintf("%s.ln2.weight", prefix)] = layer.LN2.W.Value()
		params[fmt.Sprintf("%s.ln2.bias", prefix)] = layer.LN2.B.Value()

		params[fmt.Sprintf("%s.att.time_decay", prefix)] = layer.TimeMix.TimeDecay.Value()
		params[fmt.Sprintf("%s.att.time_first", prefix)] = layer.TimeMix.TimeFirst.Value()
		params[fmt.Sprintf("%s.att.time_mix_k", prefix)] = layer.TimeMix.TimeMixK.Value()
		params[fmt.Sprintf("%s.att.time_mix_v", prefix)] = layer.TimeMix.TimeMixV.Value()
		params[fmt.Sprintf("%s.att.time_mix_r", prefix)] = layer.TimeMix.TimeMixR.Value()
		params[fmt.Sprintf("%s.att.key.weight", prefix)] = layer.TimeMix.Key.Value()
		params[fmt.Sprintf("%s.att.value.weight", prefix)] = layer.TimeMix.Value.Value()
		params[fmt.Sprintf("%s.att.receptance.weight", prefix)] = layer.TimeMix.Receptance.Value()
		params[fmt.Sprintf("%s.att.output.weight", prefix)] = layer.TimeMix.Output.Value()

		params[fmt.Sprintf("%s.ffn.time_mix_k", prefix)] = layer.ChanMix.TimeMixK.Value()
		params[fmt.Sprintf("%s.ffn.time_mix_r", prefix)] = layer.ChanMix.TimeMixR.Value()
		params[fmt.Sprintf("%s.ffn.key.weight", prefix)] = layer.ChanMix.Key.Value()
		params[fmt.Sprintf("%s.ffn.receptance.weight", prefix)] = layer.ChanMix.Receptance.Value()
		params[fmt.Sprintf("%s.ffn.value.weight", prefix)] = layer.ChanMix.Value.Value()
	}

	// Decoder parameters
	params["ln_out.weight"] = m.LN.W.Value()
	params["ln_out.bias"] = m.LN.B.Value()
	params["head.weight"] = m.Linear.Value()

	return params
}

// extractPyTorchModelParamsFromPickleFile returns the model parameters.
func extractPyTorchModelParamsFromPickleFile[T float.DType](filename string) (map[string][]T, error) {
	result, err := pytorch.Load(filename)
	if err != nil {
		return nil, err
	}
	params := make(map[string][]T)
	fn := func(name string, tensor *pytorch.Tensor) {
		if _, ok := tensor.Source.(*pytorch.FloatStorage); ok {
			params[name] = extractTensorValuesAsFloat32Slice[T](tensor)
		}
	}
	switch r := result.(type) {
	case *types.OrderedDict:
		yieldOrderedDict(r, fn)
	case *types.Dict:
		yieldDict(r, fn)
	}
	return params, err
}

func yieldOrderedDict(dict *types.OrderedDict, fn func(name string, tensor *pytorch.Tensor)) {
	for key, entry := range dict.Map {
		fn(key.(string), entry.Value.(*pytorch.Tensor))
	}
}

func yieldDict(dict *types.Dict, fn func(name string, tensor *pytorch.Tensor)) {
	for _, entry := range *dict {
		fn(entry.Key.(string), entry.Value.(*pytorch.Tensor))
	}
}

// extractTensorValuesAsFloat32Slice returns the underlying values of a PyTorch tensor as a T slice.
// It returns the extractTensorValuesAsFloat32Slice using the row-major representation, possibly converting column-major order to row-major order.
func extractTensorValuesAsFloat32Slice[T float.DType](t *pytorch.Tensor) []T {
	if len(t.Size) == 0 || len(t.Size) > 2 {
		panic("gopickleutils: number of sizes not supported")
	}
	size := t.Size[0]
	if len(t.Size) > 1 {
		size *= t.Size[1]
	}
	orig := t.Source.(*pytorch.FloatStorage).Data[t.StorageOffset : t.StorageOffset+size]
	data := make([]T, len(orig))

	if len(t.Size) == 1 || t.Size[1] == 1 || t.Size[0] == 1 || t.Stride[1] == 1 {
		for i, val := range orig {
			data[i] = T(val)
		}
		return data
	}

	s0, s1 := t.Size[1], t.Size[0]
	for i := 0; i < s0; i++ {
		for j := 0; j < s1; j++ {
			data[i+j*s0] = T(orig[j+i*s1])
		}
	}
	return data
}
