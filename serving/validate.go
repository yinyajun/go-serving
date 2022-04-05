/*
* @Author: Yajun
* @Date:   2022/4/5 10:53
 */

package serving

import (
	"github.com/go-playground/validator/v10"
)

var (
	valid = NewValidator()
)

func NewValidator() *validator.Validate {
	v := validator.New()
	return v
}
