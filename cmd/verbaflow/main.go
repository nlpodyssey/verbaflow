// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"strings"

	"github.com/nlpodyssey/spago/ag"
	"github.com/nlpodyssey/verbaflow"
	"github.com/nlpodyssey/verbaflow/downloader"
	"github.com/nlpodyssey/verbaflow/rwkvlm"
	"github.com/nlpodyssey/verbaflow/service"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/urfave/cli/v2"
)

func main() {
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr}).Level(zerolog.InfoLevel)

	app := &cli.App{
		Name:  "verbaflow",
		Usage: "Perform various operations with a language model",
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:  "log-level",
				Usage: "set log level (trace, debug, info, warn, error, fatal, panic)",
				Action: func(c *cli.Context, s string) error {
					return setDebugLevel(s)
				},
				Value:   "info",
				EnvVars: []string{"VERBAFLOW_LOGLEVEL"},
			},
			&cli.StringFlag{
				Name:     "model-dir",
				Usage:    "directory of the model to operate on",
				Required: true,
			},
		},
		Commands: []*cli.Command{
			{
				Name:  "download",
				Usage: "Download model to directory",
				Action: func(c *cli.Context) error {
					if err := download(c.String("model-dir")); err != nil {
						log.Err(err).Send()
					}
					return nil
				},
			},
			{
				Name:  "convert",
				Usage: "Convert model in directory",
				Action: func(c *cli.Context) error {
					if err := convert(c.String("model-dir")); err != nil {
						log.Fatal().Err(err).Send()
					}
					return nil
				},
			},
			{
				Name:  "inference",
				Usage: "Serve a gRPC inference endpoint",
				Action: func(c *cli.Context) error {
					modelDir := c.String("model-dir")
					address := c.String("address")

					ctx, stop := signal.NotifyContext(c.Context, os.Interrupt, os.Kill)
					defer stop()

					if err := inference(ctx, modelDir, address); err != nil {
						fmt.Print(err)
						log.Err(err).Send()
					}
					return nil
				},
				Flags: []cli.Flag{
					&cli.StringFlag{
						Name:     "address",
						Usage:    "The address to listen on for gRPC connections",
						Value:    ":50051",
						Required: false,
					},
				},
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal().Err(err).Send()
	}
}

func setDebugLevel(debugLevel string) error {
	level, err := zerolog.ParseLevel(debugLevel)
	if err != nil {
		return err
	}
	log.Logger = log.Level(level)
	return nil
}

func download(modelDir string) error {
	log.Debug().Msgf("Downloading model in dir: %s", modelDir)
	dir, name, err := splitPathAndModelName(modelDir)
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	err = downloader.Download(dir, name, false, "")
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	log.Debug().Msg("Done.")
	return nil
}

func convert(modelDir string) error {
	log.Debug().Msgf("Converting model in dir: %s", modelDir)
	err := rwkvlm.ConvertPickledModelToRWKVLM[float32](rwkvlm.ConverterConfig{
		ModelDir:         modelDir,
		OverwriteIfExist: false,
	})
	if err != nil {
		log.Fatal().Err(err).Send()
	}
	log.Debug().Msg("Done.")
	return nil
}

func inference(ctx context.Context, modelDir string, address string) error {
	log.Debug().Msgf("Starting inference server for model in dir: %s", modelDir)
	log.Debug().Msgf("Loading model...")
	vf, err := verbaflow.Load(modelDir)
	if err != nil {
		return err
	}

	log.Debug().Msgf("Server listening on %s", address)
	server := service.NewServer(vf)
	return server.Start(ctx, address)
}

// splitPathAndModelName separate the models directory from the model name, which format is "organization/model"
func splitPathAndModelName(path string) (string, string, error) {
	dirs := strings.Split(strings.TrimSuffix(path, "/"), "/")
	if len(dirs) < 3 {
		return "", "", fmt.Errorf("path must have at least three levels of directories")
	}
	lastDir := dirs[len(dirs)-1]
	secondLastDir := dirs[len(dirs)-2]

	pathExceptLastTwo := strings.Join(dirs[:len(dirs)-2], "/")
	return pathExceptLastTwo, filepath.Join(secondLastDir, lastDir), nil
}

func init() {
	ag.SetForceSyncExecution(false)
}
