/*
* @Author: Yajun
* @Date:   2022/4/1 19:26
 */

package params

import (
	"encoding/binary"
	"fmt"
	"log"
	"os"

	"github.com/yinyajun/go-serving/proto"
	"gorgonia.org/tensor"
)

const (
	FooterSize = 30
	Magic      = "go_serving"
)

type Index []int

type Meta interface {
	IndexLookup(fieldName, defaultFeat string, featNames ...string) Index
	EmbeddingLookup(fieldName string, index Index) tensor.Tensor
	GetTensor(fieldName string) tensor.Tensor
}

type stat struct {
	modelName  string
	version    uint64
	headSize   int64
	dataSize   int64
	indexSize  int64
	footerSize int64
}

type header struct {
	modelName string `validate:"required"`
	version   uint64 `validate:"required,len=10"`
}

type footer struct {
	dataOffset  uint64 `validate:"required,len=10"`
	indexOffset uint64 `validate:"required,len=10"`
	magicNumber []byte `validate:"required,len=10"`
}

type Params struct {
	file    string
	header  header
	footer  footer
	tensors map[string]*tensor.Dense
	index   map[string]*proto.Field
	stat    stat
}

func New(file string) *Params {
	m := &Params{
		file:    file,
		tensors: make(map[string]*tensor.Dense),
		index:   make(map[string]*proto.Field),
		stat:    stat{},
	}
	return m
}

func (m *Params) parseFooter(buf []byte) error {
	if len(buf) != FooterSize {
		return FooterInvalidLengthErr
	}

	footer := footer{}
	footer.dataOffset, _ = binary.Uvarint(buf[:10])
	footer.indexOffset, _ = binary.Uvarint(buf[10:20])
	footer.magicNumber = buf[20:30]
	if string(footer.magicNumber) != Magic {
		return InvalidMagicErr
	}
	m.footer = footer
	m.stat.footerSize = int64(len(buf))
	return nil
}

func (m *Params) parseHeader(buf []byte) error {
	header := header{}
	header.modelName = string(buf[:len(buf)-10])
	header.version, _ = binary.Uvarint(buf[len(buf)-10 : len(buf)])
	m.header = header
	m.stat.headSize = int64(len(buf))
	m.stat.modelName = header.modelName
	m.stat.version = header.version
	return nil
}

func (m *Params) parseData(buf []byte) error {
	data := new(proto.Data)
	if err := data.Unmarshal(buf); err != nil {
		return err
	}

	for k, v := range data.Data {
		m.tensors[k] = decode(v)
	}
	m.stat.dataSize = int64(len(buf))
	return nil
}

func (m *Params) parseIndex(buf []byte) error {
	index := new(proto.Index)
	if err := index.Unmarshal(buf); err != nil {
		return err
	}
	m.index = index.Embeddings
	m.stat.indexSize = int64(len(buf))
	return nil
}

func (m *Params) Load() (err error) {
	f, err := os.Open(m.file)
	if err != nil {
		return err
	}
	defer f.Close()

	stat, err := f.Stat()
	if err != nil {
		return err
	}
	// footer
	footerOffset := stat.Size() - FooterSize
	buf := make([]byte, FooterSize)
	n, err := f.ReadAt(buf, footerOffset)
	if err != nil {
		return err
	}
	if n != FooterSize {
		return FooterInvalidLengthErr
	}
	if err := m.parseFooter(buf); err != nil {
		return err
	}

	// header
	buf = make([]byte, m.footer.dataOffset)
	n, err = f.ReadAt(buf, 0)
	if err != nil {
		return err
	}
	if n != int(m.footer.dataOffset) {
		return HeaderInvalidLengthErr
	}
	if err := m.parseHeader(buf); err != nil {
		return err
	}
	// data
	buf = make([]byte, m.footer.indexOffset-m.footer.dataOffset)
	// todo: int64 overflow?
	n, err = f.ReadAt(buf, int64(m.footer.dataOffset))
	if err != nil {
		return err
	}
	if n != int(m.footer.indexOffset-m.footer.dataOffset) {
		return DataInvalidLengthErr
	}
	if err := m.parseData(buf); err != nil {
		return err
	}
	// index
	buf = make([]byte, footerOffset-int64(m.footer.indexOffset))
	n, err = f.ReadAt(buf, int64(m.footer.indexOffset))
	if err != nil {
		return err
	}
	if n != int(footerOffset-int64(m.footer.indexOffset)) {
		return IndexInvalidLengthErr
	}
	if err := m.parseIndex(buf); err != nil {
		return err
	}
	log.Printf("Load %s (%v) OK!", m.file, m.stat)
	return
}

func (m *Params) ModelName() string {
	return m.header.modelName
}

func (m *Params) Version() uint64 {
	return m.header.version
}

func (m *Params) EmbeddingLookup(fieldName string, index Index) tensor.Tensor {
	tt := m.tensors[fieldName]
	// lookup embedding depends on indices
	tensors := make([]tensor.Tensor, len(index))
	for i := 0; i < len(index); i++ {
		view, _ := tt.Slice(tensor.S(index[i]))
		tensors[i] = view.Materialize().(*tensor.Dense)
	}
	res, _ := tensor.Stack(0, tensors[0], tensors[1:]...)
	return res
}

func (m *Params) IndexLookup(fieldName, defaultFeat string, featNames ...string) Index {
	ii := m.index[fieldName]
	// get indices
	indices := make([]int, len(featNames))
	var (
		defaultFeatIdx int64 = -1
		featIdx        int64
		ok             bool
	)
	for i := 0; i < len(indices); i++ {
		featIdx, ok = ii.Records[featNames[i]]
		if ok {
			indices[i] = int(featIdx)
			continue
		}
		// find default featName in Field
		if defaultFeatIdx == -1 {
			defaultFeatIdx, ok = ii.Records[defaultFeat] // after check, defaultFeatName is definitely exists
		}
		indices[i] = int(defaultFeatIdx)
	}
	return indices
}

func (m *Params) GetTensor(fieldName string) tensor.Tensor {
	return m.tensors[fieldName]
}

func (m *Params) ShowTensors() {
	for k, v := range m.tensors {
		fmt.Println(k)
		fmt.Println(v)
	}
}

func (m *Params) ShowIndex() {
	for k, v := range m.index {
		fmt.Println(k)
		for i, j := range v.Records {
			fmt.Println("  ", i, j)
		}
	}
}

func decode(t *proto.Tensor) *tensor.Dense {
	var back tensor.ConsOpt

	switch t.GetDtype() {
	case proto.DataType_DT_FLOAT:
		back = tensor.WithBacking(t.GetFloatVal())
	case proto.DataType_DT_INT32:
		back = tensor.WithBacking(t.GetIntVal())
	case proto.DataType_DT_STRING:
		back = tensor.WithBacking(t.GetStringVal())
	}

	var shapes []int
	for _, i := range t.GetTensorShape() {
		shapes = append(shapes, int(i))
	}
	return tensor.New(back, tensor.WithShape(shapes...))
}
