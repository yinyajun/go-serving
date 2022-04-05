/*
* @Author: Yajun
* @Date:   2022/4/1 12:46
 */

package main

import (
	"fmt"
	"github.com/yinyajun/go-serving/column"
	"github.com/yinyajun/go-serving/model"
	"github.com/yinyajun/go-serving/serving"
	"gorgonia.org/tensor"
	"time"
)

func LRModel() model.Model {
	f1 := column.NewIdentityColumn("F1", "124", 2)
	f3 := column.NewEmbeddingColumn(f1, "", 3, column.Sum)
	return model.NewLR("LR", 3, []column.DenseColumn{f3, f3})
}

func main() {
	s := serving.New()
	s.Register(&serving.ModelConfig{
		Name:  "wide_deep",
		Path:  "/tmp/data/wide_deep",
		Model: LRModel(),
	})
	s.Launch()
	go s.Watch()

	go func() {
		for {
			batch := 2
			feats := map[string]tensor.Tensor{
				"F1": tensor.New(tensor.WithBacking([]string{"123", "124", "123", "125", "234", "126"}), tensor.WithShape(batch, 3)),
				"F2": tensor.New(tensor.WithBacking([]float32{13, -1, 19, -1}), tensor.WithShape(batch, 2)),
			}
			out, err := s.Request("wide_deep", feats)
			fmt.Println(err)
			fmt.Println(out)
			time.Sleep(3 * time.Second)
		}

	}()
	select {}
}
