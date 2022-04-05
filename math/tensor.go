/*
* @Author: Yajun
* @Date:   2022/4/3 17:26
 */

package math

import (
	"gorgonia.org/tensor"
)

func Sigmoid(logit tensor.Tensor) (tensor.Tensor, error) {
	// todo: check t.dType==float32
	neg, err := tensor.Neg(logit)
	if err != nil {
		return nil, err
	}

	e, err := tensor.Exp(neg)
	if err != nil {
		return nil, err
	}

	s, err := tensor.Add(float32(1), e)
	if err != nil {
		return nil, err
	}

	output, err := tensor.Div(float32(1), s)
	if err != nil {
		return nil, err
	}
	return output, nil
}

// Multiply 简化了tensor乘法，支持对b的最后一个维度做broadcast
func Multiply(a, b *tensor.Dense) (retVal *tensor.Dense, err error) {
	if a.Dims()-b.Dims() != 1 {
		return nil, DimensionErr
	}

	// 通过repeat，将b整成和a相同的shape
	repeat := a.Shape()[a.Dims()-1]

	bb, err := tensor.Repeat(b, b.Dims()-1, repeat)
	if err != nil {
		return nil, err
	}

	err = bb.Reshape(a.Shape()...)
	if err != nil {
		return nil, err
	}

	return a.Mul(bb.(*tensor.Dense))
}
