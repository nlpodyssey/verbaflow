// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package decoder

// StepResult is the result of a single step of the decoder.
type StepResult struct {
	// TokenID is the ID of the token predicted by the decoder at the current step.
	TokenID int
	// SumNegLogProbs is the sum of the negative log probabilities up to the current step.
	SumNegLogProbs float64
}

// Buffer is the interface that wraps the basic buffer methods.
type Buffer interface {
	// Write writes the given step result to the buffer.
	Write(stepResult StepResult) error
	// Close closes the buffer.
	Close()
}

// ChannelBuffer is a buffer that writes the results to a channel.
type ChannelBuffer chan StepResult

// Write writes the given step result to the buffer.
func (cb ChannelBuffer) Write(stepResult StepResult) error {
	cb <- stepResult
	return nil
}

// Close closes the buffer.
func (cb ChannelBuffer) Close() {
	close(cb)
}
