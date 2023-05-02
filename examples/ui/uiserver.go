// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/nlpodyssey/verbaflow/api"
	"github.com/nlpodyssey/verbaflow/decoder"
	"github.com/rs/zerolog/log"
	"nhooyr.io/websocket"
	"nhooyr.io/websocket/wsjson"
)

type UIServer struct {
	lmClient  api.LanguageModelClient
	decParams *api.DecodingParameters
}

func NewUIServer(lmClient api.LanguageModelClient, decOpts decoder.DecodingOptions) *UIServer {
	return &UIServer{
		lmClient:  lmClient,
		decParams: decodingOptionsToGRPC(decOpts),
	}
}

func (s *UIServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	c, err := websocket.Accept(w, r, &websocket.AcceptOptions{
		InsecureSkipVerify: true,
	})
	if err != nil {
		http.Error(w, fmt.Sprintf("websocket.Accept error: %v", err), http.StatusInternalServerError)
		log.Err(err).Msg("websocket.Accept error")
		return
	}

	defer func() {
		_ = c.Close(websocket.StatusInternalError, "")
	}()

	ctx, cancelMainCtx := context.WithCancel(r.Context())
	defer cancelMainCtx()

	readChan := make(chan ClientMessage, 1)
	defer close(readChan)

	go func() {
		for {
			var msg ClientMessage
			if err := wsjson.Read(ctx, c, &msg); err != nil {
				log.Warn().Err(err).Msg("failed to read JSON message")
				cancelMainCtx()
				return
			}
			readChan <- msg
		}
	}()

	for {
		cliMsg := <-readChan
		if cliMsg.Type != "prompt" {
			log.Warn().Msgf("unexpected message type: %+v", cliMsg)
			continue
		}

		func() {
			genCtx, cancel := context.WithCancelCause(ctx)
			defer cancel(nil)

			go func() {
				for {
					select {
					case msg := <-readChan:
						if msg.Type == "stop-token-stream" {
							cancel(errStopTokenStream)
							return
						}
						log.Warn().Msgf("ignoring message type during tokens streaming: %+v", msg)
					case <-genCtx.Done():
						return
					}
				}
			}()

			log.Trace().Msgf("prompt: %s", cliMsg.Value)

			tokenStream, err := s.lmClient.GenerateTokens(genCtx, &api.TokenGenerationRequest{
				Prompt:             strings.ReplaceAll(cliMsg.Value, `\n`, "\n"),
				DecodingParameters: s.decParams,
			})
			if err != nil {
				err = fmt.Errorf("failed to generate tokens: %w", err)
				log.Warn().Err(err).Send()
				_ = wsjson.Write(ctx, c, ServerMessage{Type: "error", Value: err.Error()})
				return
			}

			for {
				token, err := tokenStream.Recv()
				if err != nil {
					if err == io.EOF || context.Cause(genCtx) == errStopTokenStream {
						if err = wsjson.Write(ctx, c, ServerMessage{Type: "end-of-tokens"}); err != nil {
							log.Warn().Err(err).Msg("failed to write message")
							cancelMainCtx()
						}
						return
					}
					err = fmt.Errorf("failed to receive token stream: %w", err)
					log.Warn().Err(err).Send()
					_ = wsjson.Write(ctx, c, ServerMessage{Type: "error", Value: err.Error()})
					cancelMainCtx()
					return
				}

				if err = wsjson.Write(ctx, c, ServerMessage{Type: "token", Value: token.Token}); err != nil {
					log.Warn().Err(err).Msg("failed to write message")
					cancelMainCtx()
					return
				}
			}
		}()
	}
}

type ClientMessage struct {
	Type  string `json:"type"`
	Value string `json:"value,omitempty"`
}

type ServerMessage struct {
	Type  string `json:"type"`
	Value string `json:"value,omitempty"`
}

var errStopTokenStream = errors.New("stop token stream")

//func (s *UIServer) serveWSIteration(c *websocket.Conn, ctx context.Context) error {
//
//}

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
