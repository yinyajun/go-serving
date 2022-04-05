/*
* @Author: Yajun
* @Date:   2022/4/1 17:06
 */

package column

import "gorgonia.org/tensor"

type Inputs interface {
	Get(key interface{}) (tensor.Tensor, bool)
	Set(key interface{}, t tensor.Tensor)
}
