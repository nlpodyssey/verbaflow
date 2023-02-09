// Copyright 2023 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package verbaflow

import (
	"bytes"
	"fmt"
	"text/template"
)

// InputPrompt is the input for the prompt generation.
type InputPrompt struct {
	Text           string `json:"text"`
	Question       string `json:"question,omitempty"`
	TargetLanguage string `json:"target_language,omitempty"`
}

// BuildPromptFromTemplateFile builds a prompt applying the given input to the template file.
func BuildPromptFromTemplateFile(input InputPrompt, filename string) (string, error) {
	pt, err := template.ParseFiles(filename)
	if err != nil {
		return "", fmt.Errorf("unable to read the template file: %w", err)
	}
	return BuildPromptFromTemplate(input, pt)
}

// BuildPromptFromTemplate builds a prompt applying the given input to the template.
func BuildPromptFromTemplate(input InputPrompt, pt *template.Template) (string, error) {
	result := new(bytes.Buffer)
	err := pt.Execute(result, input)
	if err != nil {
		return "", fmt.Errorf("unable to execute the template: %w", err)
	}
	return result.String(), nil
}
