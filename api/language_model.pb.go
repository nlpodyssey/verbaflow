// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        v3.21.5
// source: language_model.proto

package api

import (
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// TokenGenerationRequest contains the prompt and decoding parameters for generating tokens
type TokenGenerationRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Prompt is the input string to use as a starting point for token generation
	Prompt string `protobuf:"bytes,1,opt,name=prompt,proto3" json:"prompt,omitempty"`
	// DecodingParameters are the parameters to use for token generation
	DecodingParameters *DecodingParameters `protobuf:"bytes,2,opt,name=decoding_parameters,json=decodingParameters,proto3" json:"decoding_parameters,omitempty"`
}

func (x *TokenGenerationRequest) Reset() {
	*x = TokenGenerationRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_language_model_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *TokenGenerationRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*TokenGenerationRequest) ProtoMessage() {}

func (x *TokenGenerationRequest) ProtoReflect() protoreflect.Message {
	mi := &file_language_model_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use TokenGenerationRequest.ProtoReflect.Descriptor instead.
func (*TokenGenerationRequest) Descriptor() ([]byte, []int) {
	return file_language_model_proto_rawDescGZIP(), []int{0}
}

func (x *TokenGenerationRequest) GetPrompt() string {
	if x != nil {
		return x.Prompt
	}
	return ""
}

func (x *TokenGenerationRequest) GetDecodingParameters() *DecodingParameters {
	if x != nil {
		return x.DecodingParameters
	}
	return nil
}

// DecodingParameters contains the parameters to use for token generation
type DecodingParameters struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// MaxLen is the maximum number of tokens to generate.
	MaxLen int32 `protobuf:"varint,1,opt,name=max_len,json=maxLen,proto3" json:"max_len,omitempty"`
	// MinLen is the minimum number of tokens to generate.
	MinLen int32 `protobuf:"varint,2,opt,name=min_len,json=minLen,proto3" json:"min_len,omitempty"`
	// Temperature controls the randomness of the generated tokens. A higher temperature will result in more diverse generated tokens.
	Temperature float32 `protobuf:"fixed32,3,opt,name=temperature,proto3" json:"temperature,omitempty"`
	// TopK is the maximum number of tokens to consider when sampling the next token.
	TopK int32 `protobuf:"varint,4,opt,name=top_k,json=topK,proto3" json:"top_k,omitempty"`
	// TopP is the cumulative probability of the tokens to consider when sampling the next token.
	TopP float32 `protobuf:"fixed32,5,opt,name=top_p,json=topP,proto3" json:"top_p,omitempty"`
	// UseSampling uses sampling to generate the next token.
	UseSampling bool `protobuf:"varint,6,opt,name=use_sampling,json=useSampling,proto3" json:"use_sampling,omitempty"`
	// EndTokenID is the end-of-sequence token (default: 0).
	EndTokenId int32 `protobuf:"varint,7,opt,name=end_token_id,json=endTokenId,proto3" json:"end_token_id,omitempty"`
	// SkipEndTokenID when true, the end token is not added to the generated sequence.
	SkipEndTokenId bool `protobuf:"varint,8,opt,name=skip_end_token_id,json=skipEndTokenId,proto3" json:"skip_end_token_id,omitempty"`
	// StopSequences are the sequences of token ids that will cause the generation to stop.
	StopSequences []*Sequence `protobuf:"bytes,9,rep,name=stop_sequences,json=stopSequences,proto3" json:"stop_sequences,omitempty"`
}

func (x *DecodingParameters) Reset() {
	*x = DecodingParameters{}
	if protoimpl.UnsafeEnabled {
		mi := &file_language_model_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *DecodingParameters) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*DecodingParameters) ProtoMessage() {}

func (x *DecodingParameters) ProtoReflect() protoreflect.Message {
	mi := &file_language_model_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use DecodingParameters.ProtoReflect.Descriptor instead.
func (*DecodingParameters) Descriptor() ([]byte, []int) {
	return file_language_model_proto_rawDescGZIP(), []int{1}
}

func (x *DecodingParameters) GetMaxLen() int32 {
	if x != nil {
		return x.MaxLen
	}
	return 0
}

func (x *DecodingParameters) GetMinLen() int32 {
	if x != nil {
		return x.MinLen
	}
	return 0
}

func (x *DecodingParameters) GetTemperature() float32 {
	if x != nil {
		return x.Temperature
	}
	return 0
}

func (x *DecodingParameters) GetTopK() int32 {
	if x != nil {
		return x.TopK
	}
	return 0
}

func (x *DecodingParameters) GetTopP() float32 {
	if x != nil {
		return x.TopP
	}
	return 0
}

func (x *DecodingParameters) GetUseSampling() bool {
	if x != nil {
		return x.UseSampling
	}
	return false
}

func (x *DecodingParameters) GetEndTokenId() int32 {
	if x != nil {
		return x.EndTokenId
	}
	return 0
}

func (x *DecodingParameters) GetSkipEndTokenId() bool {
	if x != nil {
		return x.SkipEndTokenId
	}
	return false
}

func (x *DecodingParameters) GetStopSequences() []*Sequence {
	if x != nil {
		return x.StopSequences
	}
	return nil
}

// Sequence is a sequence of token ids
type Sequence struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Sequence is the sequence of token ids
	Sequence []int32 `protobuf:"varint,1,rep,packed,name=sequence,proto3" json:"sequence,omitempty"`
}

func (x *Sequence) Reset() {
	*x = Sequence{}
	if protoimpl.UnsafeEnabled {
		mi := &file_language_model_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Sequence) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Sequence) ProtoMessage() {}

func (x *Sequence) ProtoReflect() protoreflect.Message {
	mi := &file_language_model_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Sequence.ProtoReflect.Descriptor instead.
func (*Sequence) Descriptor() ([]byte, []int) {
	return file_language_model_proto_rawDescGZIP(), []int{2}
}

func (x *Sequence) GetSequence() []int32 {
	if x != nil {
		return x.Sequence
	}
	return nil
}

// GeneratedToken contains a generated token, its score, and its encoded representation
type GeneratedToken struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	// Token is the generated token
	Token string `protobuf:"bytes,1,opt,name=token,proto3" json:"token,omitempty"`
	// Score is the sum of the negative log probabilities up to the current step.
	Score float32 `protobuf:"fixed32,2,opt,name=score,proto3" json:"score,omitempty"`
}

func (x *GeneratedToken) Reset() {
	*x = GeneratedToken{}
	if protoimpl.UnsafeEnabled {
		mi := &file_language_model_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *GeneratedToken) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*GeneratedToken) ProtoMessage() {}

func (x *GeneratedToken) ProtoReflect() protoreflect.Message {
	mi := &file_language_model_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use GeneratedToken.ProtoReflect.Descriptor instead.
func (*GeneratedToken) Descriptor() ([]byte, []int) {
	return file_language_model_proto_rawDescGZIP(), []int{3}
}

func (x *GeneratedToken) GetToken() string {
	if x != nil {
		return x.Token
	}
	return ""
}

func (x *GeneratedToken) GetScore() float32 {
	if x != nil {
		return x.Score
	}
	return 0
}

var File_language_model_proto protoreflect.FileDescriptor

var file_language_model_proto_rawDesc = []byte{
	0x0a, 0x14, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x5f, 0x6d, 0x6f, 0x64, 0x65, 0x6c,
	0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x03, 0x61, 0x70, 0x69, 0x22, 0x7a, 0x0a, 0x16, 0x54,
	0x6f, 0x6b, 0x65, 0x6e, 0x47, 0x65, 0x6e, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x16, 0x0a, 0x06, 0x70, 0x72, 0x6f, 0x6d, 0x70, 0x74, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x06, 0x70, 0x72, 0x6f, 0x6d, 0x70, 0x74, 0x12, 0x48, 0x0a,
	0x13, 0x64, 0x65, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67, 0x5f, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65,
	0x74, 0x65, 0x72, 0x73, 0x18, 0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x17, 0x2e, 0x61, 0x70, 0x69,
	0x2e, 0x44, 0x65, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74,
	0x65, 0x72, 0x73, 0x52, 0x12, 0x64, 0x65, 0x63, 0x6f, 0x64, 0x69, 0x6e, 0x67, 0x50, 0x61, 0x72,
	0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73, 0x22, 0xb8, 0x02, 0x0a, 0x12, 0x44, 0x65, 0x63, 0x6f,
	0x64, 0x69, 0x6e, 0x67, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73, 0x12, 0x17,
	0x0a, 0x07, 0x6d, 0x61, 0x78, 0x5f, 0x6c, 0x65, 0x6e, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52,
	0x06, 0x6d, 0x61, 0x78, 0x4c, 0x65, 0x6e, 0x12, 0x17, 0x0a, 0x07, 0x6d, 0x69, 0x6e, 0x5f, 0x6c,
	0x65, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x06, 0x6d, 0x69, 0x6e, 0x4c, 0x65, 0x6e,
	0x12, 0x20, 0x0a, 0x0b, 0x74, 0x65, 0x6d, 0x70, 0x65, 0x72, 0x61, 0x74, 0x75, 0x72, 0x65, 0x18,
	0x03, 0x20, 0x01, 0x28, 0x02, 0x52, 0x0b, 0x74, 0x65, 0x6d, 0x70, 0x65, 0x72, 0x61, 0x74, 0x75,
	0x72, 0x65, 0x12, 0x13, 0x0a, 0x05, 0x74, 0x6f, 0x70, 0x5f, 0x6b, 0x18, 0x04, 0x20, 0x01, 0x28,
	0x05, 0x52, 0x04, 0x74, 0x6f, 0x70, 0x4b, 0x12, 0x13, 0x0a, 0x05, 0x74, 0x6f, 0x70, 0x5f, 0x70,
	0x18, 0x05, 0x20, 0x01, 0x28, 0x02, 0x52, 0x04, 0x74, 0x6f, 0x70, 0x50, 0x12, 0x21, 0x0a, 0x0c,
	0x75, 0x73, 0x65, 0x5f, 0x73, 0x61, 0x6d, 0x70, 0x6c, 0x69, 0x6e, 0x67, 0x18, 0x06, 0x20, 0x01,
	0x28, 0x08, 0x52, 0x0b, 0x75, 0x73, 0x65, 0x53, 0x61, 0x6d, 0x70, 0x6c, 0x69, 0x6e, 0x67, 0x12,
	0x20, 0x0a, 0x0c, 0x65, 0x6e, 0x64, 0x5f, 0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x5f, 0x69, 0x64, 0x18,
	0x07, 0x20, 0x01, 0x28, 0x05, 0x52, 0x0a, 0x65, 0x6e, 0x64, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x49,
	0x64, 0x12, 0x29, 0x0a, 0x11, 0x73, 0x6b, 0x69, 0x70, 0x5f, 0x65, 0x6e, 0x64, 0x5f, 0x74, 0x6f,
	0x6b, 0x65, 0x6e, 0x5f, 0x69, 0x64, 0x18, 0x08, 0x20, 0x01, 0x28, 0x08, 0x52, 0x0e, 0x73, 0x6b,
	0x69, 0x70, 0x45, 0x6e, 0x64, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x49, 0x64, 0x12, 0x34, 0x0a, 0x0e,
	0x73, 0x74, 0x6f, 0x70, 0x5f, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x73, 0x18, 0x09,
	0x20, 0x03, 0x28, 0x0b, 0x32, 0x0d, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x53, 0x65, 0x71, 0x75, 0x65,
	0x6e, 0x63, 0x65, 0x52, 0x0d, 0x73, 0x74, 0x6f, 0x70, 0x53, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63,
	0x65, 0x73, 0x22, 0x26, 0x0a, 0x08, 0x53, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x12, 0x1a,
	0x0a, 0x08, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x18, 0x01, 0x20, 0x03, 0x28, 0x05,
	0x52, 0x08, 0x73, 0x65, 0x71, 0x75, 0x65, 0x6e, 0x63, 0x65, 0x22, 0x3c, 0x0a, 0x0e, 0x47, 0x65,
	0x6e, 0x65, 0x72, 0x61, 0x74, 0x65, 0x64, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x12, 0x14, 0x0a, 0x05,
	0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x74, 0x6f, 0x6b,
	0x65, 0x6e, 0x12, 0x14, 0x0a, 0x05, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28,
	0x02, 0x52, 0x05, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x32, 0x55, 0x0a, 0x0d, 0x4c, 0x61, 0x6e, 0x67,
	0x75, 0x61, 0x67, 0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x12, 0x44, 0x0a, 0x0e, 0x47, 0x65, 0x6e,
	0x65, 0x72, 0x61, 0x74, 0x65, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x73, 0x12, 0x1b, 0x2e, 0x61, 0x70,
	0x69, 0x2e, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x47, 0x65, 0x6e, 0x65, 0x72, 0x61, 0x74, 0x69, 0x6f,
	0x6e, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x13, 0x2e, 0x61, 0x70, 0x69, 0x2e, 0x47,
	0x65, 0x6e, 0x65, 0x72, 0x61, 0x74, 0x65, 0x64, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x30, 0x01, 0x42,
	0x25, 0x5a, 0x23, 0x67, 0x69, 0x74, 0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x6e, 0x6c,
	0x70, 0x6f, 0x64, 0x79, 0x73, 0x73, 0x65, 0x79, 0x2f, 0x76, 0x65, 0x72, 0x62, 0x61, 0x66, 0x6c,
	0x6f, 0x77, 0x2f, 0x61, 0x70, 0x69, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_language_model_proto_rawDescOnce sync.Once
	file_language_model_proto_rawDescData = file_language_model_proto_rawDesc
)

func file_language_model_proto_rawDescGZIP() []byte {
	file_language_model_proto_rawDescOnce.Do(func() {
		file_language_model_proto_rawDescData = protoimpl.X.CompressGZIP(file_language_model_proto_rawDescData)
	})
	return file_language_model_proto_rawDescData
}

var file_language_model_proto_msgTypes = make([]protoimpl.MessageInfo, 4)
var file_language_model_proto_goTypes = []interface{}{
	(*TokenGenerationRequest)(nil), // 0: api.TokenGenerationRequest
	(*DecodingParameters)(nil),     // 1: api.DecodingParameters
	(*Sequence)(nil),               // 2: api.Sequence
	(*GeneratedToken)(nil),         // 3: api.GeneratedToken
}
var file_language_model_proto_depIdxs = []int32{
	1, // 0: api.TokenGenerationRequest.decoding_parameters:type_name -> api.DecodingParameters
	2, // 1: api.DecodingParameters.stop_sequences:type_name -> api.Sequence
	0, // 2: api.LanguageModel.GenerateTokens:input_type -> api.TokenGenerationRequest
	3, // 3: api.LanguageModel.GenerateTokens:output_type -> api.GeneratedToken
	3, // [3:4] is the sub-list for method output_type
	2, // [2:3] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_language_model_proto_init() }
func file_language_model_proto_init() {
	if File_language_model_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_language_model_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*TokenGenerationRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_language_model_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*DecodingParameters); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_language_model_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Sequence); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_language_model_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*GeneratedToken); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_language_model_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   4,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_language_model_proto_goTypes,
		DependencyIndexes: file_language_model_proto_depIdxs,
		MessageInfos:      file_language_model_proto_msgTypes,
	}.Build()
	File_language_model_proto = out.File
	file_language_model_proto_rawDesc = nil
	file_language_model_proto_goTypes = nil
	file_language_model_proto_depIdxs = nil
}
