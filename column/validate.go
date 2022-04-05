/*
* @Author: Yajun
* @Date:   2022/4/5 11:38
 */

package column

import "github.com/go-playground/validator/v10"

var (
	valid = NewValidator()
)

func NewValidator() *validator.Validate {
	v := validator.New()
	return v
}
