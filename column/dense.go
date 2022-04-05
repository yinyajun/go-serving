/*
* @Author: Yajun
* @Date:   2022/4/1 19:14
 */

package column

import (
	"math"

	math2 "github.com/yinyajun/go-serving/math"
	"github.com/yinyajun/go-serving/params"
	"gorgonia.org/tensor"
)

type DenseColumn interface {
	FeatureColumn
	Dimension() int
	CategoricalColumn() CategoricalColumn
}

type DenseColumns []DenseColumn

func (m DenseColumns) Less(i, j int) bool { return m[i].Name() < m[j].Name() }
func (m DenseColumns) Swap(i, j int)      { m[i], m[j] = m[j], m[i] }
func (m DenseColumns) Len() int           { return len(m) }

type Combiner int

const (
	Sum Combiner = iota
	Mean
	SqrtN
)

type EmbeddingColumn struct {
	column  CategoricalColumn
	weight  string
	dim     int
	combine Combiner
}

func NewEmbeddingColumn(col CategoricalColumn, weight string, dim int, comb Combiner) *EmbeddingColumn {
	return &EmbeddingColumn{
		column:  col,
		weight:  weight,
		dim:     dim,
		combine: comb,
	}
}

func (c *EmbeddingColumn) CategoricalColumn() CategoricalColumn { return c.column }

func (c *EmbeddingColumn) Dimension() int { return c.dim }

func (c *EmbeddingColumn) Name() string { return c.column.Name() + "_embedding" }

func (c *EmbeddingColumn) Transform(m params.Meta, inputs Inputs) (tensor.Tensor, error) {
	// already transformed
	if t, ok := inputs.Get(c); ok {
		return t, nil
	}

	index, ok := inputs.Get(c.column)
	var err error
	if !ok {
		index, err = c.column.Transform(m, inputs)
		if err != nil {
			return nil, err
		}
	}

	embeddings := m.EmbeddingLookup(c.column.Name(), index.Data().([]int)).(*tensor.Dense) // todo: can omit check?
	shape := append(index.Shape(), c.dim)
	err = embeddings.Reshape(shape...)
	if err != nil {
		return nil, err
	}

	if embeddings.Dims() <= 2 {
		return embeddings, nil
	}

	// embeddings.Dims() > 2, need pooling
	if c.weight != "" {
		_weight, ok := inputs.Get(c.weight)
		if ok {
			// todo : check weight shape and type
			weight := _weight.(*tensor.Dense)
			return weightedEmbeddingPooling(embeddings, weight, c.combine)
		}
	}
	return embeddingPooling(embeddings, c.combine)
}

func embeddingPooling(embeddings *tensor.Dense, combine Combiner) (tensor.Tensor, error) {
	shape := embeddings.Shape()
	sum, _ := embeddings.Sum(tensor.Range(tensor.Int, 1, len(shape)-1).([]int)...)

	size := 1
	for _, s := range shape[1 : len(shape)-1] {
		size *= s
	}

	switch combine {
	case Sum:
		return sum, nil
	case Mean:
		return sum.DivScalar(float32(size), true)
	case SqrtN:
		return sum.DivScalar(float32(math.Sqrt(float64(size))), true)
	default:
		panic(InvalidCombineErr)
	}
}

func weightedEmbeddingPooling(embeddings, weight *tensor.Dense, combine Combiner) (tensor.Tensor, error) {
	weightedEmbeddings, err := math2.Multiply(embeddings, weight)
	if err != nil {
		panic(err)
	}
	shape := weightedEmbeddings.Shape()
	along := tensor.Range(tensor.Int, 1, len(shape)-1).([]int)
	sum, _ := weightedEmbeddings.Sum(along...)

	switch combine {
	case Sum:
		return sum, nil
	case Mean:
		b, _ := weight.Sum(along...)
		return sum.Div(b)
	case SqrtN:
		b, _ := weight.PowScalar(float32(2), true)
		bb, _ := b.Sum(along...)
		bbb, _ := bb.PowScalar(float32(0.5), true)
		return sum.Div(bbb)
	default:
		panic(InvalidCombineErr)
	}
}
