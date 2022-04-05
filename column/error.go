/*
* @Author: Yajun
* @Date:   2022/4/1 11:15
 */

package column

import (
	"errors"
	"fmt"
	"reflect"
)

var (
	InvalidCombineErr = errors.New("invalid combine type")
	DimensionErr      = errors.New("dimension mismatch")
)

type FieldNotFoundError struct {
	field string
}

func (e FieldNotFoundError) Error() string {
	return fmt.Sprintf("Field %s not found in inputs", e.field)
}

type FieldTypeError struct {
	field    string
	expected reflect.Type
	provided reflect.Type
}

func (e FieldTypeError) Error() string {
	return fmt.Sprintf("Expected Field type is %s, but provided is %s", e.expected, e.provided)
}

func assert(b bool) {
	if !b {
		panic("assertion failed")
	}
}
