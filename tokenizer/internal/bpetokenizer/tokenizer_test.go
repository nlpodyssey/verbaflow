// Copyright 2020 spaGO Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpetokenizer

import (
	"testing"
)

func TestNew(t *testing.T) {
	tokenizer, err := Load("testdata/dummy-roberta-model", ControlTokensIDs{})
	if err != nil {
		t.Fatal(err)
	}
	if tokenizer == nil {
		t.Fatal("expected *BPETokenizer, actual nil")
	}
}
