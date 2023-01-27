# VerbaFlow

Welcome to VerbaFlow, a neural architecture written in Go designed specifically for language modeling tasks. 
Built on the robust RWKV RNN, this model is optimized for efficient performance on standard CPUs, enabling smooth running of relatively large language models even on consumer hardware.

With the ability to utilize pretrained models on the [Pile](https://arxiv.org/abs/2101.00027) dataset, VerbaFlow performs comparably to GPT-like Transformer models in predicting the next token, as well as in other tasks such as sentiment analysis, question answering, and general conversation. 

This package is a Go port of the original Python [RWKV-LN](https://github.com/BlinkDL/RWKV-LM) by PENG Bo. 

# Installation

Requirements:

* [Go 1.19](https://golang.org/dl/)

Clone this repo or get the library:

```console
go get -u github.com/nlpodyssey/verbaflow
```

# Usage

The following commands can be used to build and use VerbaFlow:

```console
go build -o verbaflow cmd/main.go
```

This command builds the go program and creates an executable named `verbaflow`.

```console
./verbaflow download models/nlpodyssey/RWKV-4-Pile-3B-Instruct
```

This command downloads the model specified (in this case, "nlpodyssey/RWKV-4-Pile-3B-Instruct" under the "models" directory)

```console
./verbaflow convert models/nlpodyssey/RWKV-4-Pile-3B-Instruct
```

This command converts the downloaded model to the format used by the program.

```console
./verbaflow inference models/nlpodyssey/RWKV-4-Pile-3B-Instruct
```

This command runs the inference on the specified model.

Please make sure to have the necessary dependencies installed before running the above commands.

> The library is optimized to run in x86-64 CPUs. If you want to run it on a different architecture, you can use the `GOARCH=amd64` environment variable.

## Dependencies

A list of the main dependencies follows:

- [Spago](http://github.com/nlpodyssey/spago) - Machine Learning framework
- [RWKV](http://github.com/nlpodyssey/rwkv) - RWKV RNN implementation
- [GoTokenizers](http://github.com/nlpodyssey/gotokenizers) - Tokenizers library
- [GoPickle](http://github.com/nlpodyssey/gopickle) - Pickle library for Go

# Roadmap

- [x] Download pretrained models from the Hugging Face models hub
- [ ] Effective "prompts" catalog
- [ ] Better sampling
- [ ] Beam search
- [ ] Tokenizer
- [ ] Unit tests
- [ ] Code refactoring
- [ ] Documentation
- [ ] gRPC/HTTP API

# Credits

Thanks [PENG Bo](https://github.com/BlinkDL) for creating the RWKV RNN and all related resources, including pre-trained models!

# Trivia about the project's name

"VerbaFlow" combines "verba", which is the Latin word for *words*, and "flow", which alludes to the characteristics of recurrent neural networks by evoking the idea of a fluent and continuous flow of words, which is made possible by the network's ability to maintain an internal state and "remember" previous words and context when generating new words.