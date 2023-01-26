// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"os"

	"github.com/nlpodyssey/verbaflow/rwkvlm"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Println("Usage: go run cmd/main.go [convert model_dir] | [inference model_dir]")
		return
	}

	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr})

	switch args[0] {
	case "convert":
		if len(args) < 2 {
			fmt.Println("Error: missing model dir argument")
			return
		}
		modelDir := args[1]
		log.Info().Msgf("Converting model in dir: %s", modelDir)
		if err := convert(modelDir); err != nil {
			log.Fatal().Err(err).Send()
		}
		log.Info().Msg("Done.")
	case "inference":
		if len(args) < 2 {
			fmt.Println("Error: missing model dir argument")
			return
		}
		modelDir := args[1]
		log.Info().Msgf("Performing inference on model in dir: %s", modelDir)
		// TODO: implement inference
	default:
		fmt.Println("Error: invalid command")
	}
}

func convert(modelDir string) error {
	return rwkvlm.ConvertPickledModelToRWKVLM[float32](&rwkvlm.ConverterConfig{
		ModelDir:         modelDir,
		OverwriteIfExist: true,
	})
}
