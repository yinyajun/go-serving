/*
* @Author: Yajun
* @Date:   2022/4/5 15:21
 */

package serving

import (
	"github.com/yinyajun/go-serving/model"
	"github.com/yinyajun/go-serving/params"
	"sync/atomic"
	"time"
	"unsafe"
)

type ModelConfig struct {
	Name  string      `validate:"required"`
	Path  string      `validate:"required"`
	Model model.Model `validate:"required"`
}

type servingModel struct {
	config    *ModelConfig
	meta      unsafe.Pointer
	startTime time.Time
}

func (s *servingModel) GetMeta() *params.Params {
	return (*params.Params)(atomic.LoadPointer(&(s.meta)))
}

type modelManager struct {
	paths map[string]*servingModel
	names map[string]*servingModel
}

func (f modelManager) Set(c *ModelConfig) error {
	m := servingModel{config: c, meta: unsafe.Pointer(&params.Params{})}
	if _, ok := f.paths[c.Path]; ok {
		return DuplicatedError{c.Path}
	}
	if _, ok := f.names[c.Name]; ok {
		return DuplicatedError{c.Name}
	}
	f.paths[c.Path] = &m
	f.names[c.Name] = &m
	return nil
}

func (f modelManager) GetModelByPath(path string) (*servingModel, bool) {
	m, ok := f.paths[path]
	return m, ok
}

func (f modelManager) GetModelByName(name string) (*servingModel, bool) {
	m, ok := f.names[name]
	return m, ok
}
