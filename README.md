# go-serving

Mimic tensorflow serving


# model

```go
func LRModel() model.Model {
	f1 := column.NewIdentityColumn("F1", "124", 2)
	f3 := column.NewEmbeddingColumn(f1, "", 3, column.Sum)
	return model.NewLR("LR", 3, []column.DenseColumn{f3, f3})
}
```


# serving

```go
s := serving.New()
	s.Register(&serving.ModelConfig{
		Name:  "wide_deep",
		Path:  "/tmp/data/wide_deep",
		Model: LRModel(),
	})
	s.Launch()
	go s.Watch()
```