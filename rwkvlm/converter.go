// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkvlm

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/nlpodyssey/gopickle/pytorch"
	"github.com/nlpodyssey/gopickle/types"
	"github.com/nlpodyssey/spago/mat"
	"github.com/nlpodyssey/spago/mat/float"
	"github.com/nlpodyssey/spago/nn"
	"github.com/nlpodyssey/spago/nn/embedding"
	"github.com/nlpodyssey/spago/nn/normalization/layernorm"
	"github.com/nlpodyssey/verbaflow/rwkv"
	"github.com/rs/zerolog/log"
)

const (
	DefaultPyModelFilename = "pytorch_model.pt"
	DefaultOutputFilename  = "spago_model.bin"

	DefaultLayerNormEps = 1e-5
)

type ConverterConfig struct {
	// The path to the directory where the models will be read from and written to.
	ModelDir string
	// The path to the input model file (default "pytorch_model.pt")
	PyModelFilename string
	// The path to the output model file (default "spago_model.bin")
	GoModelFilename string
	// If true, overwrite the model file if it already exists (default "false")
	OverwriteIfExist bool
}

// ConvertPickledModelToRWKVLM converts a PyTorch model to a RWKVLM model.
// It expects a configuration file "config.json" in the same directory as the model file containing the model configuration.
func ConvertPickledModelToRWKVLM[T float.DType](config ConverterConfig) error {
	if config.PyModelFilename == "" {
		config.PyModelFilename = DefaultPyModelFilename
	}
	if config.GoModelFilename == "" {
		config.GoModelFilename = DefaultOutputFilename
	}

	outputFilename := filepath.Join(config.ModelDir, config.GoModelFilename)

	if !config.OverwriteIfExist && fileExists(outputFilename) {
		log.Debug().Str("model", outputFilename).Msg("Model file already exists, skipping conversion")
		return nil
	}

	configFilename := filepath.Join(config.ModelDir, "config.json")
	modelConfig, err := LoadConfig(configFilename)
	if err != nil {
		return fmt.Errorf("failed to load config file %q: %w", configFilename, err)
	}

	inFilename := filepath.Join(config.ModelDir, config.PyModelFilename)
	conv := newConverter[T](modelConfig, inFilename, outputFilename)
	err = conv.run()
	if err != nil {
		return fmt.Errorf("model conversion failed: %w", err)
	}
	return nil
}

func fileExists(name string) bool {
	info, err := os.Stat(name)
	return err == nil && !info.IsDir()
}

type converter[T float.DType] struct {
	model       *Model
	inFilename  string
	outFilename string
	embRepoPath string
	params      paramsMap
}

func newConverter[T float.DType](conf Config, inFilename, outFilename string) *converter[T] {
	return &converter[T]{
		model:       &Model{Config: conf},
		inFilename:  inFilename,
		outFilename: outFilename,
	}
}

func (c *converter[T]) run() error {
	funcs := []func() error{
		c.loadTorchModelParams,
		c.convEmbeddings,
		c.convLinear,
		c.convRootLayerNorm,
		c.convBlocks,
		c.dumpModel,
	}
	for _, fn := range funcs {
		if err := fn(); err != nil {
			return err
		}
	}
	return nil
}

func (c *converter[T]) dumpModel() (err error) {
	return Dump(c.model, c.outFilename)
}

func (c *converter[T]) convRootLayerNorm() (err error) {
	c.model.LN, err = c.convLayerNorm("ln_out", c.params)
	if err != nil {
		err = fmt.Errorf("failed to convert layer-norm: %w", err)
	}
	return
}

func (c *converter[T]) convEmbeddings() error {
	embWeight, err := c.params.fetch("emb.weight")
	if err != nil {
		return err
	}

	vecs, err := c.tensorToVectors(embWeight)
	if err != nil {
		return fmt.Errorf("failed to convert embeddings: %w", err)
	}

	if vs := c.model.Config.VocabSize; vs == 0 {
		c.model.Config.VocabSize = len(vecs)
	} else if len(vecs) != vs {
		return fmt.Errorf("expected embedding vectors to match vocabulary size %d, actual %d", vs, len(vecs))
	}

	if dm := c.model.Config.DModel; dm == 0 {
		c.model.Config.DModel = vecs[0].Size()
	} else if dm != vecs[0].Size() {
		return fmt.Errorf("expected embedding vectors to match configured size %d, actual %d", dm, vecs[0].Size())
	}

	embs := embedding.New[T](c.model.Config.VocabSize, c.model.Config.DModel)
	for i, vec := range vecs {
		embs.Weights[i].ReplaceValue(vec)
	}
	c.model.Embeddings = embs

	return nil
}

func (c *converter[T]) convLinear() error {
	headWeight, err := c.params.fetch("head.weight")
	if err != nil {
		return err
	}

	m, err := c.tensorToMatrix(headWeight)
	if err != nil {
		return fmt.Errorf("failed to convert head-weight/linear: %w", err)
	}

	if vs := c.model.Config.VocabSize; m.Rows() != vs {
		return fmt.Errorf("expected head-weight/linear rows to match vocabulary size %d, actual %d", vs, m.Rows())
	}
	if dm := c.model.Config.DModel; m.Columns() != dm {
		return fmt.Errorf("expected head-weight/linear columns to match DModel %d, actual %d", dm, m.Columns())
	}

	c.model.Linear = nn.NewParam(m)
	return nil
}

func (c *converter[T]) convBlocks() error {
	allBlocksParams := c.params.fetchPrefixed("blocks.")
	numBlocks, err := countBlocks(allBlocksParams)
	if err != nil {
		return err
	}
	if numBlocks == 0 {
		return fmt.Errorf("no blocks/layers found in parameters")
	}
	if hl := c.model.Config.NumHiddenLayers; hl == 0 {
		c.model.Config.NumHiddenLayers = numBlocks
	} else if hl != numBlocks {
		return fmt.Errorf("expected %d blocks/layers, actual %d", hl, numBlocks)
	}

	conf := rwkv.Config{
		DModel:       c.model.Config.DModel,
		NumLayers:    c.model.Config.NumHiddenLayers,
		RescaleLayer: c.model.Config.RescaleLayer,
	}

	layers := make([]*rwkv.Layer, numBlocks)
	for i := range layers {
		blockParams := allBlocksParams.fetchPrefixed(fmt.Sprintf("%d.", i))
		layers[i], err = c.convBlock(i, conf, blockParams)
		if err != nil {
			return fmt.Errorf("failed to convert block/layer %d: %w", i, err)
		}
	}

	c.model.Encoder = &rwkv.Model{
		Config: conf,
		Layers: layers,
	}
	return nil
}

func (c *converter[T]) convBlock(id int, conf rwkv.Config, params paramsMap) (_ *rwkv.Layer, err error) {
	layer := &rwkv.Layer{}

	layer.ChanMix, err = c.convChanMix(id, params.fetchPrefixed("ffn."))
	if err != nil {
		return nil, fmt.Errorf("failed to convert ffn/channel-mix: %w", err)
	}

	layer.TimeMix, err = c.convTimeMix(id, conf, params.fetchPrefixed("att."))
	if err != nil {
		return nil, fmt.Errorf("failed to convert att/time-mix: %w", err)
	}

	if id == 0 {
		layer.LN0, err = c.convLayerNorm("ln0", params)
		if err != nil {
			return nil, fmt.Errorf("failed to convert layer-norm 0: %w", err)
		}
	}

	layer.LN1, err = c.convLayerNorm("ln1", params)
	if err != nil {
		return nil, fmt.Errorf("failed to convert layer-norm 1: %w", err)
	}

	layer.LN2, err = c.convLayerNorm("ln2", params)
	if err != nil {
		return nil, fmt.Errorf("failed to convert layer-norm 2: %w", err)
	}

	return layer, nil
}

func (c *converter[T]) convChanMix(id int, params paramsMap) (*rwkv.ChannelMix, error) {
	dm := c.model.Config.DModel
	outScale := math.Pow(2, float64(id/c.model.Config.RescaleLayer))

	key, err := c.fetchParamToMatrix(params, "key.weight", [2]int{dm * 4, dm})
	if err != nil {
		return nil, fmt.Errorf("failed to convert key weight: %w", err)
	}

	receptance, err := c.fetchParamToMatrix(params, "receptance.weight", [2]int{dm, dm})
	if err != nil {
		return nil, fmt.Errorf("failed to convert receptance weight: %w", err)
	}

	value, err := c.fetchParamToMatrix(params, "value.weight", [2]int{dm, dm * 4})
	if err != nil {
		return nil, fmt.Errorf("failed to convert value weight: %w", err)
	}
	if outScale != 1 {
		value.ProdScalarInPlace(1 / outScale)
	}

	tmk, err := c.fetchParamToSqueezedVector(params, "time_mix_k", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert time-mix-k: %w", err)
	}

	tmr, err := c.fetchParamToSqueezedVector(params, "time_mix_r", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert time-mix-r: %w", err)
	}

	return &rwkv.ChannelMix{
		Key:        nn.NewParam(key),
		Value:      nn.NewParam(value),
		Receptance: nn.NewParam(receptance),
		TimeMixK:   nn.NewParam(tmk),
		TimeMixR:   nn.NewParam(tmr),
	}, nil
}

func (c *converter[T]) convTimeMix(id int, conf rwkv.Config, params paramsMap) (*rwkv.TimeMix, error) {
	dm := c.model.Config.DModel
	outScale := math.Pow(2, float64(id/c.model.Config.RescaleLayer))

	key, err := c.fetchParamToMatrix(params, "key.weight", [2]int{dm, dm})
	if err != nil {
		return nil, fmt.Errorf("failed to convert key weight: %w", err)
	}

	receptance, err := c.fetchParamToMatrix(params, "receptance.weight", [2]int{dm, dm})
	if err != nil {
		return nil, fmt.Errorf("failed to convert receptance weight: %w", err)
	}

	output, err := c.fetchParamToMatrix(params, "output.weight", [2]int{dm, dm})
	if err != nil {
		return nil, fmt.Errorf("failed to convert output weight: %w", err)
	}
	if outScale != 1 {
		output.ProdScalarInPlace(1 / outScale)
	}

	value, err := c.fetchParamToMatrix(params, "value.weight", [2]int{dm, dm})
	if err != nil {
		return nil, fmt.Errorf("failed to convert value weight: %w", err)
	}

	tDecay, err := c.fetchParamToSqueezedVector(params, "time_decay", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert time-decay: %w", err)
	}
	tDecay = tDecay.Exp().ProdScalarInPlace(-1)

	tFirst, err := c.fetchParamToSqueezedVector(params, "time_first", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert time-first: %w", err)
	}

	tmk, err := c.fetchParamToSqueezedVector(params, "time_mix_k", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert time-mix-k: %w", err)
	}

	tmr, err := c.fetchParamToSqueezedVector(params, "time_mix_r", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert time-mix-r: %w", err)
	}

	tmv, err := c.fetchParamToSqueezedVector(params, "time_mix_v", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert time-mix-v: %w", err)
	}

	return &rwkv.TimeMix{
		Config:     conf,
		Key:        nn.NewParam(key),
		Value:      nn.NewParam(value),
		Receptance: nn.NewParam(receptance),
		Output:     nn.NewParam(output),
		TimeDecay:  nn.NewParam(tDecay),
		TimeFirst:  nn.NewParam(tFirst),
		TimeMixK:   nn.NewParam(tmk),
		TimeMixV:   nn.NewParam(tmv),
		TimeMixR:   nn.NewParam(tmr),
	}, nil
}

func (c *converter[T]) convLayerNorm(name string, params paramsMap) (*layernorm.Model, error) {
	dm := c.model.Config.DModel

	w, err := c.fetchParamToVector(params, name+".weight", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert layer-norm weight: %w", err)
	}

	b, err := c.fetchParamToVector(params, name+".bias", dm)
	if err != nil {
		return nil, fmt.Errorf("failed to convert layer-norm bias: %w", err)
	}

	return &layernorm.Model{
		W:   nn.NewParam(w),
		B:   nn.NewParam(b),
		Eps: nn.Const[T](DefaultLayerNormEps),
	}, nil
}

func (c *converter[T]) loadTorchModelParams() error {
	torchModel, err := pytorch.Load(c.inFilename)
	if err != nil {
		return fmt.Errorf("failed to load torch model %q: %w", c.inFilename, err)
	}
	c.params, err = makeParamsMap(torchModel)
	if err != nil {
		return fmt.Errorf("failed to read model params: %w", err)
	}
	return nil
}

func (c *converter[T]) tensorToVectors(t *pytorch.Tensor) ([]mat.Matrix, error) {
	if len(t.Size) != 2 {
		return nil, fmt.Errorf("expected 2 dimensions, actual %d", len(t.Size))
	}

	data, err := c.tensorData(t)
	if err != nil {
		return nil, err
	}

	rows := t.Size[0]
	cols := t.Size[1]

	vecs := make([]mat.Matrix, rows)
	for i := range vecs {
		d := data[i*cols : (i*cols)+cols]
		vecs[i] = mat.NewVecDense[T](c.castMatrixData(d))
	}

	return vecs, nil
}

func (c *converter[T]) tensorToMatrix(t *pytorch.Tensor) (mat.Matrix, error) {
	if len(t.Size) != 2 {
		return nil, fmt.Errorf("expected 2 dimensions, actual %d", len(t.Size))
	}

	data, err := c.tensorData(t)
	if err != nil {
		return nil, err
	}

	return mat.NewDense[T](t.Size[0], t.Size[1], c.castMatrixData(data)), nil
}

func (c *converter[T]) tensorToVector(t *pytorch.Tensor) (mat.Matrix, error) {
	if len(t.Size) != 1 {
		return nil, fmt.Errorf("expected 1 dimension, actual %d", len(t.Size))
	}

	data, err := c.tensorData(t)
	if err != nil {
		return nil, err
	}

	return mat.NewVecDense[T](c.castMatrixData(data)), nil
}

func (c *converter[T]) tensorToSqueezedVector(t *pytorch.Tensor) (mat.Matrix, error) {
	data, err := c.tensorData(t)
	if err != nil {
		return nil, err
	}
	return mat.NewVecDense[T](c.castMatrixData(data)), nil
}

func (c *converter[T]) castMatrixData(d []float32) []T {
	return float.SliceValueOf[T](float.SliceInterface(d))
}

func (c *converter[T]) tensorData(t *pytorch.Tensor) ([]float32, error) {
	st, ok := t.Source.(*pytorch.BFloat16Storage)
	if !ok {
		return nil, fmt.Errorf("only BFloat16Storage is supported, actual %T", t.Source)
	}
	size := tensorDataSize(t)
	return st.Data[t.StorageOffset : t.StorageOffset+size], nil
}

func (c *converter[T]) fetchParamToVector(params paramsMap, name string, expectedSize int) (mat.Matrix, error) {
	t, err := params.fetch(name)
	if err != nil {
		return nil, err
	}
	v, err := c.tensorToVector(t)
	if err != nil {
		return nil, err
	}
	if v.Size() != expectedSize {
		return nil, fmt.Errorf("expected vector size %d, actual %d", expectedSize, v.Size())
	}
	return v, nil
}

func (c *converter[T]) fetchParamToSqueezedVector(params paramsMap, name string, expectedSize int) (mat.Matrix, error) {
	t, err := params.fetch(name)
	if err != nil {
		return nil, err
	}
	v, err := c.tensorToSqueezedVector(t)
	if err != nil {
		return nil, err
	}
	if v.Size() != expectedSize {
		return nil, fmt.Errorf("expected squeezed vector size %d, actual %d", expectedSize, v.Size())
	}
	return v, nil
}

func (c *converter[T]) fetchParamToMatrix(params paramsMap, name string, expectedSize [2]int) (mat.Matrix, error) {
	t, err := params.fetch(name)
	if err != nil {
		return nil, err
	}
	m, err := c.tensorToMatrix(t)
	if err != nil {
		return nil, err
	}
	if m.Rows() != expectedSize[0] || m.Columns() != expectedSize[1] {
		return nil, fmt.Errorf("expected matrix size %dx%d, actual %dx%d",
			expectedSize[0], expectedSize[1], m.Rows(), m.Columns())
	}
	return m, nil
}

func countBlocks(params paramsMap) (int, error) {
	max := 0
	for k := range params {
		before, _, ok := strings.Cut(k, ".")
		if !ok {
			return 0, fmt.Errorf("block/layer parameter names expected to start with number, actual name %q", k)
		}
		num, err := strconv.Atoi(before)
		if err != nil {
			return 0, fmt.Errorf("block/layer parameter names expected to start with number, actual name %q: %w", k, err)
		}
		if num > max {
			max = num
		}
	}
	return max + 1, nil
}

func tensorDataSize(t *pytorch.Tensor) int {
	size := t.Size[0]
	for _, v := range t.Size[1:] {
		size *= v
	}
	return size
}

func cast[T any](v any) (t T, _ error) {
	t, ok := v.(T)
	if !ok {
		return t, fmt.Errorf("type assertion failed: expected %T, actual %T", t, v)
	}
	return
}

type paramsMap map[string]*pytorch.Tensor

func makeParamsMap(torchModel any) (paramsMap, error) {
	od, err := cast[*types.OrderedDict](torchModel)
	if err != nil {
		return nil, err
	}

	params := make(paramsMap, od.Len())

	for k, item := range od.Map {
		name, err := cast[string](k)
		if err != nil {
			return nil, fmt.Errorf("wrong param name type: %w", err)
		}
		tensor, err := cast[*pytorch.Tensor](item.Value)
		if err != nil {
			return nil, fmt.Errorf("wrong value type for param %q: %w", name, err)
		}
		params[name] = tensor
	}

	return params, nil
}

// fetchParam gets a value from params by its name, removing the entry
// from the map.
func (p paramsMap) fetch(name string) (*pytorch.Tensor, error) {
	t, ok := p[name]
	if !ok {
		return nil, fmt.Errorf("parameter %q not found", name)
	}
	delete(p, name)
	return t, nil
}

func (p paramsMap) fetchPrefixed(prefix string) paramsMap {
	out := make(paramsMap, len(p))
	for k, v := range p {
		if after, ok := strings.CutPrefix(k, prefix); ok {
			out[after] = v
			delete(p, k)
		}
	}
	return out
}
