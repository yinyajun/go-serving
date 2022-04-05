/*
* @Author: Yajun
* @Date:   2022/4/1 19:13
 */

package column

import (
	"log"
	"reflect"
	"sort"

	"github.com/yinyajun/go-serving/params"
	"gorgonia.org/tensor"
)

type CategoricalColumn interface {
	FeatureColumn
	NumBuckets() int
}

type IdentityColumn struct {
	Field   string `validate:"required"`
	DefFeat string `validate:"required"`
	Buckets int    `validate:"required"`
}

func NewIdentityColumn(field, defFeat string, buckets int) *IdentityColumn {
	c := &IdentityColumn{
		Field:   field,
		DefFeat: defFeat,
		Buckets: buckets,
	}
	if err := valid.Struct(c); err != nil {
		log.Panicln(err)
	}
	return c
}

func (c *IdentityColumn) Name() string {
	return c.Field
}

func (c *IdentityColumn) Transform(m params.Meta, inputs Inputs) (tensor.Tensor, error) {
	// already transformed
	if t, ok := inputs.Get(c); ok {
		return t, nil
	}

	raw, ok := inputs.Get(c.Name())
	if !ok {
		return nil, FieldNotFoundError{c.Name()}
	}

	in, ok := raw.Data().([]string)
	if !ok {
		return nil, FieldTypeError{field: c.Name(), expected: reflect.TypeOf([]string{}), provided: reflect.TypeOf(in)}
	}
	indices := m.IndexLookup(c.Name(), c.DefFeat, in...)
	res := tensor.New(tensor.WithBacking(indices), tensor.WithShape(raw.Shape()...))
	inputs.Set(c, res)
	return res, nil
}

func (c *IdentityColumn) NumBuckets() int { return c.Buckets }

type BucketizedColumn struct {
	Field        string     `validate:"required"`
	DefValue     float32    `validate:"required"`
	OmittedValue float32    `validate:"required"`
	Boundaries   Boundaries `validate:"required"`
}

type Boundaries []float32

func (b Boundaries) Len() int           { return len(b) }
func (b Boundaries) Less(i, j int) bool { return b[i] < b[j] }
func (b Boundaries) Swap(i, j int)      { b[i], b[j] = b[j], b[i] }

func NewBucketizedColumn(field string, defVal, omittedVal float32, boundaries Boundaries) *BucketizedColumn {
	// todo: check sorted
	assert(sort.IsSorted(boundaries))
	assert(defVal != omittedVal)
	return &BucketizedColumn{
		Field:        field,
		DefValue:     defVal,
		OmittedValue: omittedVal,
		Boundaries:   boundaries,
	}
}

func (c *BucketizedColumn) Name() string { return c.Field }

func (c *BucketizedColumn) bucketized(target float32) int {
	return sort.Search(len(c.Boundaries), func(i int) bool { return c.Boundaries[i] > target })
}

func (c *BucketizedColumn) NumBuckets() int { return len(c.Boundaries) + 1 }

func (c *BucketizedColumn) Transform(m params.Meta, inputs Inputs) (tensor.Tensor, error) {
	// already transformed
	if t, ok := inputs.Get(c); ok {
		return t, nil
	}

	raw, ok := inputs.Get(c.Name())
	if !ok {
		return nil, FieldNotFoundError{c.Name()}
	}

	in, ok := raw.Data().([]float32)
	if !ok {
		return nil, FieldTypeError{field: c.Name(), expected: reflect.TypeOf([]string{}), provided: reflect.TypeOf(in)}
	}
	indices := make(params.Index, len(in))
	for i, k := range in {
		if k == c.OmittedValue {
			k = c.DefValue
		}
		indices[i] = c.bucketized(k)
	}
	res := tensor.New(tensor.WithBacking(indices), tensor.WithShape(raw.Shape()...))
	inputs.Set(c, res)
	return res, nil
}
