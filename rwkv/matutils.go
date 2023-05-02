// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rwkv

import "github.com/nlpodyssey/spago/ag"

// This file contains operations on matrices, which have certain limitations in terms of
// expressiveness and efficiency. To address these limitations, it is highly recommended
// to transition to using tensors, which offer more powerful and flexible operations,
// including broadcasting. Broadcasting allows us to perform element-wise operations
// between tensors of different shapes and sizes, making it easier to write and optimize
// complex mathematical computations.

func sigmoid(x []ag.Node) []ag.Node {
	y := make([]ag.Node, len(x))
	for i := range x {
		y[i] = ag.Sigmoid(x[i])
	}
	return y
}

func relu(x []ag.Node) []ag.Node {
	y := make([]ag.Node, len(x))
	for i := range x {
		y[i] = ag.ReLU(x[i])
	}
	return y
}

func square(x []ag.Node) []ag.Node {
	y := make([]ag.Node, len(x))
	for i := range x {
		y[i] = ag.Square(x[i])
	}
	return y
}

func mul(a ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(b))
	for i := range b {
		c[i] = ag.Mul(a, b[i])
	}
	return c
}

func add(a, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(a))
	for i := range a {
		c[i] = ag.Add(a[i], b[i])
	}
	return c
}

func prod(a ag.Node, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(b))
	for i := range b {
		c[i] = ag.Prod(a, b[i])
	}
	return c
}

func prod2(a, b []ag.Node) []ag.Node {
	c := make([]ag.Node, len(b))
	for i := range b {
		c[i] = ag.Prod(a[i], b[i])
	}
	return c
}
