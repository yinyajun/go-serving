/*
* @Author: Yajun
* @Date:   2022/4/1 19:18
 */

package params

import "errors"

var (
	FooterInvalidLengthErr = errors.New("invalid footer length")
	HeaderInvalidLengthErr = errors.New("invalid header length")
	DataInvalidLengthErr   = errors.New("invalid data length")
	IndexInvalidLengthErr  = errors.New("invalid index length")
	InvalidMagicErr        = errors.New("invalid magic number")
)
