/*
* @Author: Yajun
* @Date:   2022/4/3 17:04
 */

package model

import (
	"github.com/yinyajun/go-serving/column"
	"github.com/yinyajun/go-serving/layer"
	"github.com/yinyajun/go-serving/math"
	"github.com/yinyajun/go-serving/params"
	"gorgonia.org/tensor"
)

type Features map[string]tensor.Tensor

type Model interface {
	Predict(params.Meta, Features) (tensor.Tensor, error)
}

type LogisticRegression struct {
	name  string
	layer *layer.LinearModelLayer
}

func NewLR(name string, units int, columns column.DenseColumns) *LogisticRegression {
	return &LogisticRegression{
		name:  name,
		layer: layer.NewLinearModelLayer(units, columns)}
}

func (m *LogisticRegression) Predict(meta params.Meta, feats Features) (tensor.Tensor, error) {
	inputs, err := NewInputs(feats)
	if err != nil {
		return nil, err
	}
	logit, err := m.layer.Call(meta, inputs)
	if err != nil {
		return nil, err
	}
	return math.Sigmoid(logit)
}
