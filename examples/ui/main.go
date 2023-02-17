// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"flag"
	"fmt"
	"net"
	"net/http"
	"os"
	"os/signal"
	"time"

	"github.com/nlpodyssey/verbaflow/api"
	"github.com/nlpodyssey/verbaflow/decoder"
	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"gopkg.in/yaml.v3"
)

func main() {
	if err := run(); err != nil {
		log.Fatal().Err(err).Send()
	}
}

func run() error {
	configFilename := flag.String("config", "config.yaml", "configuration YAML file")
	listenAddress := flag.String("listen", ":8088", "listening address")
	vfAddress := flag.String("vfaddress", ":50051", "VerbaFlow gRPC server address")
	jsonLog := flag.Bool("json-log", false, "listening address")
	logLevel := flag.String("log-level", "info", "log level")
	flag.Parse()

	// -----

	if !*jsonLog {
		log.Logger = log.Logger.Output(zerolog.ConsoleWriter{Out: os.Stderr})
	}
	if err := setDebugLevel(*logLevel); err != nil {
		return err
	}

	// -----

	decodingOpts, err := decodingOptionsFromFile(*configFilename)
	if err != nil {
		return err
	}

	vfGrpc, err := grpc.Dial(*vfAddress, grpc.WithInsecure(), grpc.WithBlock())
	if err != nil {
		return fmt.Errorf("failed to dial %q: %w", *vfAddress, err)
	}
	defer func() {
		if err = vfGrpc.Close(); err != nil {
			log.Warn().Err(err).Msgf("failed to close gRPC connection to %q", vfAddress)
		}
	}()
	lmClient := api.NewLanguageModelClient(vfGrpc)

	// -----

	listener, err := net.Listen("tcp", *listenAddress)
	if err != nil {
		return fmt.Errorf("failed to listen on %v: %w", listener.Addr(), err)
	}
	log.Info().Msgf("listening on %v", listener.Addr())

	s := &http.Server{
		Handler:      NewUIServer(lmClient, decodingOpts),
		ReadTimeout:  10 * time.Second,
		WriteTimeout: 10 * time.Second,
	}

	serveErrChan := make(chan error, 1)
	go func() {
		serveErrChan <- s.Serve(listener)
	}()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, os.Interrupt, os.Kill)

	select {
	case err = <-serveErrChan:
		log.Err(err).Msg("failed to serve")
	case sig := <-sigChan:
		log.Warn().Msgf("received signal %v", sig)
	}

	signal.Stop(sigChan)
	defer func() {
		close(sigChan)
		close(serveErrChan)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err = s.Shutdown(ctx); err != nil {
		return fmt.Errorf("failed to shutdown server: %w", err)
	}
	return nil
}

func decodingOptionsFromFile(filepath string) (decoder.DecodingOptions, error) {
	data, err := os.ReadFile(filepath)
	if err != nil {
		return decoder.DecodingOptions{}, fmt.Errorf("error reading configuration file: %w", err)
	}
	var opts decoder.DecodingOptions
	if err = yaml.Unmarshal(data, &opts); err != nil {
		return decoder.DecodingOptions{}, fmt.Errorf("error unmarshaling configuration file: %w", err)
	}
	return opts, nil
}

func setDebugLevel(v string) error {
	level, err := zerolog.ParseLevel(v)
	if err != nil {
		return fmt.Errorf("invalid log level: %q", v)
	}
	log.Logger = log.Logger.Level(level)
	return nil
}
