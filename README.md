# VerbaFlow

Welcome to VerbaFlow, a neural architecture written in Go designed specifically for language modeling tasks. 
Built on the robust RWKV RNN, this model is optimized for efficient performance on standard CPUs, enabling smooth running of relatively large language models even on consumer hardware.

With the ability to utilize pretrained models on the [Pile](https://arxiv.org/abs/2101.00027) dataset, VerbaFlow performs comparably to GPT-like Transformer models in predicting the next token, as well as in other tasks such as sentiment analysis, question answering, and general conversation. 

This package is a Go port of the original Python [RWKV-LN](https://github.com/BlinkDL/RWKV-LM) by PENG Bo ([BlinkDL](https://github.com/BlinkDL) on GitHub). 

# Installation

Requirements:

* [Go 1.19](https://golang.org/dl/)

Clone this repo or get the library:

```console
go get -u github.com/nlpodyssey/verbaflow
```

# Usage

To utilize VerbaFlow to its full potential, we recommend using the pre-trained model `RWKV-4-Pile-3B-Instruct`, available on the [Hugging Face Hub](https://huggingface.co/nlpodyssey/RWKV-4-Pile-3B-Instruct).
This model has been fine-tuned using the [Pile](https://huggingface.co/datasets/the_pile) dataset and has been specially designed to understand and execute human instructions, as fine-tuned on the [xP3](https://huggingface.co/datasets/bigscience/xP3all) dataset. 
The original `RWKV-4-Pile-3B-Instruct-test1-20230124` model, from which this model is derived, was trained by PENG Bo and can be accessed [here](https://huggingface.co/BlinkDL/rwkv-4-pile-3b).

> The library is optimized to run in x86-64 CPUs. If you want to run it on a different architecture, you can use the `GOARCH=amd64` environment variable.

The following commands can be used to build and use VerbaFlow:

```console
go build ./cmd/verbaflow
```

This command builds the go program and creates an executable named `verbaflow`.

```console
./verbaflow -model-dir models/nlpodyssey/RWKV-4-Pile-3B-Instruct download
```

This command downloads the model specified (in this case, "nlpodyssey/RWKV-4-Pile-3B-Instruct" under the "models" directory)

```console
./verbaflow -model-dir models/nlpodyssey/RWKV-4-Pile-3B-Instruct convert
```

This command converts the downloaded model to the format used by the program.

```console
./verbaflow -log-level trace -model-dir models/nlpodyssey/RWKV-4-Pile-3B-Instruct inference --address :50051
```

This command runs the gRPC inference endpoint on the specified model.

Please make sure to have the necessary dependencies installed before running the above commands.

## Examples

One of the most interesting features of the LLM is the ability to react based on the prompt.

Run the `verbaflow` gRPC endpoint with the command in inference, then run the `client` example entering the following prompts:

### Example 1

Prompt:

```console
\nQ: Briefly: The Universe is expanding, its constituent galaxies flying apart like pieces of cosmic shrapnel in the aftermath of the Big Bang. Which section of a newspaper would this article likely appear in?\n\nA:
```

Expected output:

```console
Science and Technology
```

### Example 2

Prompt:

```console
\nQ:Translate the following text from French to English Je suis le père le plus heureux du monde\n\nA:
```

Expected output:

```console
I am the happiest father in the world.
```

## Dependencies

A list of the main dependencies follows:

- [Spago](http://github.com/nlpodyssey/spago) - Machine Learning framework
- [RWKV](http://github.com/nlpodyssey/rwkv) - RWKV RNN implementation
- [GoTokenizers](http://github.com/nlpodyssey/gotokenizers) - Tokenizers library
- [GoPickle](http://github.com/nlpodyssey/gopickle) - Pickle library for Go

# Roadmap

- [x] Download pretrained models from the Hugging Face models hub
- [ ] Effective "prompts" catalog
- [x] Better sampling
- [ ] Beam search
- [ ] Better Tokenizer
- [ ] Unit tests
- [ ] Code refactoring
- [ ] Documentation
- [x] gRPC ~~/HTTP~~ API

# Credits

Thanks [PENG Bo](https://github.com/BlinkDL) for creating the RWKV RNN and all related resources, including pre-trained models!

# Trivia about the project's name

"VerbaFlow" combines "verba", which is the Latin word for *words*, and "flow", which alludes to the characteristics of recurrent neural networks by evoking the idea of a fluent and continuous flow of words, which is made possible by the network's ability to maintain an internal state and "remember" previous words and context when generating new words.