// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package service

import (
	"context"
	"fmt"
	"net"
	"time"

	"github.com/nlpodyssey/verbaflow"
	"github.com/nlpodyssey/verbaflow/api"
	"github.com/nlpodyssey/verbaflow/decoder"
	"github.com/rs/zerolog/log"
	"google.golang.org/grpc"
	"google.golang.org/grpc/health"
	"google.golang.org/grpc/health/grpc_health_v1"
)

type Server struct {
	api.UnimplementedLanguageModelServer
	vf         *verbaflow.VerbaFlow
	health     *health.Server
	grpcServer *grpc.Server
}

func NewServer(vf *verbaflow.VerbaFlow) *Server {
	return &Server{
		vf:         vf,
		health:     health.NewServer(),
		grpcServer: grpc.NewServer(),
	}
}

func (s *Server) Start(ctx context.Context, address string) error {
	lis, err := net.Listen("tcp", address)
	if err != nil {
		return fmt.Errorf("failed to listen: %w", err)
	}

	grpc_health_v1.RegisterHealthServer(s.grpcServer, s.health)
	api.RegisterLanguageModelServer(s.grpcServer, s)

	s.health.SetServingStatus(api.LanguageModel_ServiceDesc.ServiceName, grpc_health_v1.HealthCheckResponse_SERVING)

	go s.shutDownServerWhenContextIsDone(ctx)
	return s.grpcServer.Serve(lis)
}

// shutDownServerWhenContextIsDone shuts down the server when the context is done.
func (s *Server) shutDownServerWhenContextIsDone(ctx context.Context) {
	<-ctx.Done()
	log.Info().Msg("context done, shutting down server")
	s.health.Shutdown()
	s.grpcServer.GracefulStop()
	log.Info().Msg("server shut down successfully")
}

// GenerateTokens implements the GenerateTokens method of the LanguageModel service.
func (s *Server) GenerateTokens(req *api.TokenGenerationRequest, stream api.LanguageModel_GenerateTokensServer) error {
	ctx := stream.Context()
	log.Debug().Msgf("Received request from", ctx.Value("client"))

	opts := grpcToDecodingOptions(req.GetDecodingParameters())

	// chGen is a channel that will receive the generated tokens
	chGen := make(chan decoder.GeneratedToken, opts.MaxLen)
	errCh := make(chan error)
	go func() {
		log.Trace().Msgf("Decoding...")
		start := time.Now()
		errCh <- s.vf.Generate(ctx, req.GetPrompt(), chGen, opts)
		log.Trace().Msgf("Inference time: %.2f seconds", time.Since(start).Seconds())
	}()

	checkWriteConditions := func(tokenID int) bool {
		return !(tokenID == opts.EndTokenID && opts.SkipEndTokenID)
	}

	for gen := range chGen {
		if !checkWriteConditions(gen.TokenID) {
			continue
		}
		token, err := s.vf.TokenByID(gen.TokenID)
		if err != nil {
			return fmt.Errorf("failed to reconstruct text for token ID %d", gen.TokenID)
		}
		if err = stream.Send(&api.GeneratedToken{
			Token: token,
			Score: float32(gen.SumNegLogProbs),
		}); err != nil {
			return err
		}
	}

	err := <-errCh
	if err != nil {
		return err
	}

	log.Debug().Msg("Done.")
	return nil
}

func grpcToDecodingOptions(dp *api.DecodingParameters) decoder.DecodingOptions {
	return decoder.DecodingOptions{
		MaxLen:           int(dp.MaxLen),
		MinLen:           int(dp.MinLen),
		StopSequencesIDs: nil,
		EndTokenID:       int(dp.EndTokenId),
		SkipEndTokenID:   dp.SkipEndTokenId,
		Temp:             float64(dp.Temperature),
		TopK:             int(dp.TopK),
		TopP:             float64(dp.TopP),
		UseSampling:      dp.UseSampling,
	}
}
