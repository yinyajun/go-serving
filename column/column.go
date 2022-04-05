/*
* @Author: Yajun
* @Date:   2022/4/1 10:38
 */

package column

import (
	"github.com/yinyajun/go-serving/params"
	"gorgonia.org/tensor"
)

type FeatureColumn interface {
	Name() string
	Transform(m params.Meta, inputs Inputs) (tensor.Tensor, error)
}
