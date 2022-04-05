/*
* @Author: Yajun
* @Date:   2022/4/2 17:49
 */

package serving

import (
	"errors"
	"fmt"
)

var (
	RegisterError = errors.New("must register before serving launched")
)

type DuplicatedError struct {
	name string
}

func (e DuplicatedError) Error() string {
	return fmt.Sprintf("Duplicated %s", e.name)
}

type NotMatchError struct {
	expected, provided string
}

func (e NotMatchError) Error() string {
	return fmt.Sprintf("Not Match, expected: %s, but provided: %s", e.expected, e.provided)
}

type NotFoundError struct {
	name  string
	field string
}

func (e NotFoundError) Error() string {
	return fmt.Sprintf("[%s] cannot find %s", e.field, e.name)
}

type EmptyDirError struct {
	dir string
}

func (e EmptyDirError) Error() string {
	return fmt.Sprintf("No files in dir %s", e.dir)
}

type UnregisteredError struct {
	name  string
	field string
}

func (e UnregisteredError) Error() string {
	return fmt.Sprintf("[%s] %s is unregistered", e.field, e.name)
}
