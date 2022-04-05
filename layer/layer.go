/*
* @Author: Yajun
* @Date:   2022/4/3 16:50
 */

package layer

import (
	"errors"
	"fmt"
	"sort"

	"github.com/yinyajun/go-serving/column"
	"github.com/yinyajun/go-serving/params"
	"gorgonia.org/tensor"
)

type inputLayer struct {
	columns column.DenseColumns
}

func NewInputLayer(cols column.DenseColumns) *inputLayer {
	sort.Sort(cols)
	return &inputLayer{columns: cols}
}

func (l *inputLayer) Call(m params.Meta, inputs column.Inputs) (tensor.Tensor, error) {
	tt := make([]tensor.Tensor, len(l.columns))
	for i, col := range l.columns {
		t, err := col.Transform(m, inputs)
		if err != nil {
			return nil, err
		}
		tt[i] = t
	}
	if len(tt) == 0 {
		return nil, errors.New("output tensor is empty")
	}
	outputs, err := tensor.Concat(1, tt[0], tt[1:]...)
	if err != nil {
		return nil, err
	}
	return outputs, nil
}

type LinearModelLayer struct {
	units   int
	columns column.DenseColumns
}

func NewLinearModelLayer(units int, cols column.DenseColumns) *LinearModelLayer {
	sort.IsSorted(cols)
	for i := range cols {
		if cols[i].Dimension() != units {
			msg := fmt.Sprintf("units expected dimension is %d, but %s provided is %d",
				units, cols[i].Name(), cols[i].Dimension())
			panic(msg)
		}
	}
	return &LinearModelLayer{columns: cols, units: units}
}

func (l *LinearModelLayer) Call(m params.Meta, inputs column.Inputs) (tensor.Tensor, error) {
	tt := make([]tensor.Tensor, len(l.columns))
	for i, col := range l.columns {
		t, err := col.Transform(m, inputs)
		if err != nil {
			return nil, err
		}
		tt[i] = t
	}
	if len(tt) == 0 {
		return nil, errors.New("output tensor is empty")
	}

	a, err := tensor.Stack(2, tt[0], tt[1:]...)
	if err != nil {
		return nil, err
	}
	outputs, err := tensor.Sum(a, 2)
	if err != nil {
		return nil, err
	}
	return outputs, nil
}
