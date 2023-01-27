// Copyright 2022 NLP Odyssey Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package downloader

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"

	"github.com/rs/zerolog/log"
)

const (
	// Hugging Face repository URL, in the format:
	// "https://huggingface.co/{model_id}/resolve/{revision}/{filename}"
	huggingFaceCoPrefix = "https://huggingface.co/%s/resolve/%s/%s"
	// Default revision name for fetching model from Hugging Face repository
	defaultRevision = "main"
)

// modelsFiles contains the set of files to download.
var modelsFiles = []string{
	"config.json", "pytorch_model.pt", "vocab.json", "merges.txt",
}

// Download downloads a supported pre-trained model from huggingface.co
// repositories.
//
// If one or more directory levels don't yet exist, they are created
// setting the permissions bits to 0755 (rwxr-xr-x).
//
// By setting the flag overwriteIfExist to false, any file that already
// exists is kept and considered as already successfully downloaded. If
// the flag is otherwise set to true, existing files will be forcefully
// downloaded and overwritten.
func Download(modelsDir, modelName string, overwriteIfExists bool, accessToken string) error {
	return downloader{
		modelPath:        filepath.Join(modelsDir, modelName),
		modelName:        modelName,
		overwriteIfExist: overwriteIfExists,
		accessToken:      accessToken,
	}.download()
}

// downloader is a helper struct for downloading a model.
type downloader struct {
	modelPath        string
	modelName        string
	accessToken      string
	overwriteIfExist bool
}

func (d downloader) download() error {
	if err := d.ensureModelPath(); err != nil {
		return err
	}
	for _, filename := range modelsFiles {
		if err := d.downloadFile(filename); err != nil {
			return err
		}
	}
	return nil

}

func (d downloader) ensureModelPath() error {
	if info, err := os.Stat(d.modelPath); err == nil && info.IsDir() {
		return nil
	}
	if err := os.MkdirAll(d.modelPath, 0755); err != nil {
		return fmt.Errorf("error creating model path %#v: %w", d.modelPath, err)
	}
	return nil
}

func (d downloader) downloadFile(name string) (err error) {
	fPath := filepath.Join(d.modelPath, name)
	if info, err := os.Stat(fPath); !d.overwriteIfExist && err == nil && !info.IsDir() {
		log.Debug().Str("file", fPath).Msg("model file already exists, skipping download")
		return nil
	}

	url := d.bucketURL(name)
	log.Debug().Str("url", url).Str("destination", fPath).Msg("downloading")

	f, err := os.Create(fPath)
	if err != nil {
		return fmt.Errorf("error creating file %#v: %w", fPath, err)
	}
	defer func() {
		if e := f.Close(); e != nil && err == nil {
			err = fmt.Errorf("error closing file %#v: %w", fPath, e)
		}
	}()

	resp, err := d.httpGet(url)
	if err != nil {
		return fmt.Errorf("error getting %#v: %w", url, err)
	}
	defer func() {
		if e := resp.Body.Close(); e != nil && err == nil {
			err = fmt.Errorf("error closing %#v response body: %w", url, e)
		}
	}()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("%#v responded with %s", url, resp.Status)
	}

	prog := newDownloadProgress(int(resp.ContentLength))
	prog.Start()
	defer prog.Stop()

	_, err = io.Copy(f, io.TeeReader(resp.Body, prog))
	if err != nil {
		return fmt.Errorf("error downloading %#v to %#v: %w", url, fPath, err)
	}
	return nil
}

func (d downloader) httpGet(url string) (*http.Response, error) {
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}
	if d.accessToken != "" {
		req.Header.Set("Authorization", "Bearer "+d.accessToken)
	}
	return http.DefaultClient.Do(req)
}

func (d downloader) bucketURL(fileName string) string {
	return fmt.Sprintf(huggingFaceCoPrefix, d.modelName, defaultRevision, fileName)
}
