// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/signal"
	"strings"
	"text/template"

	"github.com/nlpodyssey/verbaflow"
	"github.com/nlpodyssey/verbaflow/api"
	"github.com/nlpodyssey/verbaflow/decoder"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"github.com/urfave/cli/v2"
	"google.golang.org/grpc"
	"gopkg.in/yaml.v3"
)

type pTemplate struct {
	pt   *template.Template
	data string // the raw data of the template
}

var defaultPromptTemplate = pTemplate{
	data: "{{.Text}}",
	pt:   template.Must(template.New("").Parse("{{.Text}}")),
}

func main() {
	log.Logger = log.Output(zerolog.ConsoleWriter{Out: os.Stderr}).Level(zerolog.TraceLevel)

	app := &cli.App{
		Name:  "PromptTester",
		Usage: "Test how the language model responds to different prompts",
		Action: func(c *cli.Context) error {
			opts, err := decodingOptionsFromFile(c.String("dconfig"))
			if err != nil {
				return fmt.Errorf("error reading decoding options: %w", err)
			}
			log.Info().Msgf("Decoding options:\n %+v\n", opts)
			promptt, err := promptTemplateFromFile(c.String("promptt"))
			if err != nil {
				return fmt.Errorf("error reading prompt template: %w", err)
			}
			if err := inference(opts, promptt, c.String("endpoint")); err != nil {
				log.Err(err).Send()
			}
			return nil
		},
		Flags: []cli.Flag{
			&cli.StringFlag{
				Name:  "log-level",
				Usage: "set log level (trace, debug, info, warn, error, fatal, panic)",
				Action: func(c *cli.Context, s string) error {
					return setDebugLevel(s)
				},
				Value: "trace",
			},
			&cli.StringFlag{
				Name:     "dconfig",
				Usage:    "the path to the YAML configuration file for the decoding options",
				Required: true,
			},
			&cli.StringFlag{
				Name:     "endpoint",
				Usage:    "The address of the gRPC server",
				Value:    ":50051",
				Required: false,
			},
			&cli.StringFlag{
				Name:     "promptt",
				Usage:    `the path to the prompt template file. If not specified, the default template \n\n{{.Text}} will be used`,
				Required: false,
			},
		},
	}

	if err := app.Run(os.Args); err != nil {
		log.Fatal().Err(err).Send()
	}
}

func inference(opts decoder.DecodingOptions, promptt pTemplate, endpoint string) error {

	text, err := inputTextFromStdin()
	if err != nil {
		return err
	}

	conn, err := grpc.Dial(endpoint, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		log.Fatal().Msgf("Failed to connect: %v", err)
	}
	defer conn.Close()

	client := api.NewLanguageModelClient(conn)

	log.Trace().Msgf("Building prompt from template: %q", promptt.data)
	input, err := buildInputPrompt(text, promptt.data)
	if err != nil {
		return err
	}
	log.Trace().Msgf("Input fields: %+v", input)
	prompt, err := verbaflow.BuildPromptFromTemplate(input, promptt.pt)
	if err != nil {
		return err
	}
	log.Trace().Msgf("Final prompt: %q", prompt)

	req := &api.TokenGenerationRequest{
		Prompt:             prompt,
		DecodingParameters: decodingOptionsToGRPC(opts),
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, os.Kill)
	defer stop()

	stream, err := client.GenerateTokens(ctx, req)
	if err != nil {
		return fmt.Errorf("failed to call GenerateTokens: %v", err)
	}

	for {
		res, err := stream.Recv()

		if err != nil {
			if err == io.EOF {
				break
			} else if strings.Contains(err.Error(), "rpc error: code = Canceled desc = context canceled") {
				log.Debug().Msg("Context canceled.")
				break
			} else {
				return fmt.Errorf("failed to receive a message: %v", err)
			}
		}

		fmt.Printf(res.Token)
	}
	log.Debug().Msg("Done.")
	return nil
}

func inputTextFromStdin() (string, error) {
	info, err := os.Stdin.Stat()
	if err != nil {
		return "", fmt.Errorf("error getting standard input info: %w", err)
	}
	if info.Size() == 0 {
		return "", fmt.Errorf("no input provided")
	}
	input, err := io.ReadAll(os.Stdin)
	if err != nil {
		return "", fmt.Errorf("error reading from standard input: %w", err)
	}

	log.Trace().Msgf("Input: %q", string(input))

	return string(input), nil
}

func buildInputPrompt(text, data string) (verbaflow.InputPrompt, error) {
	if strings.Contains(data, "{{.Question}}") { // extractive question answering
		log.Trace().Msgf("Splitting text into passage and question parts by \\n\\n")
		spl := strings.Split(text, "\n\n")
		if len(spl) != 2 {
			return verbaflow.InputPrompt{}, fmt.Errorf("required passage and question separated by \\n\\n")
		}
		return verbaflow.InputPrompt{
			Text:     spl[0],
			Question: spl[1],
		}, nil
	}

	return verbaflow.InputPrompt{Text: text}, nil
}

func setDebugLevel(debugLevel string) error {
	level, err := zerolog.ParseLevel(debugLevel)
	if err != nil {
		return err
	}
	log.Logger = log.Level(level)
	return nil
}

func decodingOptionsFromFile(filepath string) (decoder.DecodingOptions, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return decoder.DecodingOptions{}, fmt.Errorf("error reading configuration file: %w", err)
	}
	var opts decoder.DecodingOptions
	if err := yaml.Unmarshal(data, &opts); err != nil {
		return decoder.DecodingOptions{}, fmt.Errorf("error unmarshaling configuration file: %w", err)
	}
	return opts, nil
}

func promptTemplateFromFile(filepath string) (pTemplate, error) {
	if filepath == "" {
		return defaultPromptTemplate, nil
	}
	t, err := template.ParseFiles(filepath)
	if err != nil {
		return pTemplate{}, fmt.Errorf("error parsing template file: %w", err)
	}
	b, err := os.ReadFile(filepath) // just pass the file name
	if err != nil {
		return pTemplate{}, fmt.Errorf("error reading template file: %w", err)
	}
	return pTemplate{
		pt:   t,
		data: string(b),
	}, nil
}

func decodingOptionsToGRPC(opts decoder.DecodingOptions) *api.DecodingParameters {
	return &api.DecodingParameters{
		MaxLen:         int32(opts.MaxLen),
		MinLen:         int32(opts.MinLen),
		Temperature:    float32(opts.Temp),
		TopK:           int32(opts.TopK),
		TopP:           float32(opts.TopP),
		UseSampling:    opts.UseSampling,
		EndTokenId:     int32(opts.EndTokenID),
		SkipEndTokenId: opts.SkipEndTokenID,
	}
}
