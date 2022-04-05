/*
* @Author: Yajun
* @Date:   2022/4/3 10:48
 */

package model

import (
	"errors"
	"gorgonia.org/tensor"
)

type InputCache map[interface{}]tensor.Tensor

func (b InputCache) Get(key interface{}) (tensor.Tensor, bool) {
	t, ok := b[key]
	return t, ok
}

func (b InputCache) Set(key interface{}, t tensor.Tensor) {
	b[key] = t
}

func NewInputs(features Features) (InputCache, error) {
	var (
		in    = make(InputCache)
		batch = -1
	)
	for name, val := range features {
		// todo: check batch size
		if batch != -1 && val.Shape()[0] != batch {
			return nil, errors.New("expected same batchsize")
		}
		batch = val.Shape()[0]
		in[name] = val
	}
	return in, nil
}
