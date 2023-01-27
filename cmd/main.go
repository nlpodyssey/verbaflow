// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/verbaflow"
	"github.com/nlpodyssey/verbaflow/rwkvlm"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
)

const (
	maxTokens     = 100
	stopOnNewLine = true
)

func main() {
	args := os.Args[1:]
	if len(args) == 0 {
		fmt.Println("Usage: go run cmd/main.go [convert model_dir] | [inference model_dir]")
		return
	}

	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr}).Level(zerolog.DebugLevel)

	switch args[0] {
	case "convert":
		if len(args) < 2 {
			fmt.Println("Error: missing model dir argument")
			return
		}
		modelDir := args[1]
		log.Debug().Msgf("Converting model in dir: %s", modelDir)
		if err := convert(modelDir); err != nil {
			log.Fatal().Err(err).Send()
		}
		log.Debug().Msg("Done.")
	case "inference":
		if len(args) < 2 {
			fmt.Println("Error: missing model dir argument")
			return
		}
		modelDir := args[1]
		log.Debug().Msgf("Performing inference on model in dir: %s", modelDir)
		if err := inference(modelDir); err != nil {
			log.Fatal().Err(err).Send()
		}
	default:
		fmt.Println("Error: invalid command")
	}
}

func convert(modelDir string) error {
	return rwkvlm.ConvertPickledModelToRWKVLM[float32](&rwkvlm.ConverterConfig{
		ModelDir:         modelDir,
		OverwriteIfExist: false,
	})
}

func inference(modelDir string) error {
	log.Debug().Msgf("Loading model...")
	vf, err := verbaflow.Load(modelDir)
	if err != nil {
		return err
	}
	defer vf.Close()

	log.Debug().Msgf("Ready.")

	fn := func(text string) error {
		genCh := make(chan string) // the channel is used to receive the text as the generation progresses

		go func() {
			err2 := vf.Generate(context.Background(), text, maxTokens, stopOnNewLine, genCh)
			if err2 != nil {
				log.Fatal().Err(err2).Send()
			}
		}()

		// print the stream of generated tokens
		for token := range genCh {
			fmt.Print(token)
		}
		fmt.Println()
		return nil
	}

	err = forEachInput(os.Stdin, fn)
	if err != nil {
		log.Fatal().Err(err)
	}

	return nil
}

// forEachInput calls the given callback function for each line of input.
func forEachInput(r io.Reader, callback func(text string) error) (err error) {
	scanner := bufio.NewScanner(r)
	for {
		fmt.Print("> ")
		scanner.Scan()
		text := scanner.Text()
		switch text {
		case "":
			continue
		case "(quit)":
			break
		}
		text = strings.Replace(text, `\n`, "\n", -1)
		if err = callback(text); err != nil {
			break
		}
	}
	return err
}

func init() {
	ag.SetDebugMode(false)
}
