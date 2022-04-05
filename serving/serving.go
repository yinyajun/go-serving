/*
* @Author: Yajun
* @Date:   2022/4/1 21:15
 */

package serving

import (
	"log"
	"os"
	"path"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/fsnotify/fsnotify"
	"github.com/yinyajun/go-serving/model"
	"github.com/yinyajun/go-serving/params"
	"gorgonia.org/tensor"
)

type Serving struct {
	launched bool
	once     sync.Once
	lock     sync.RWMutex
	watcher  *fsnotify.Watcher
	models   modelManager
}

func New() *Serving {
	serving := &Serving{
		models: modelManager{
			paths: make(map[string]*servingModel),
			names: make(map[string]*servingModel),
		},
	}
	return serving
}

func (s *Serving) Register(c *ModelConfig) {
	if s.launched {
		log.Panicln(RegisterError)
	}
	if err := valid.Struct(c); err != nil {
		log.Panicln(err)
	}
	if err := s.models.Set(c); err != nil {
		log.Panicln(err)
	}
}

func (s *Serving) Launch() {
	s.once.Do(func() {
		s.launch()
	})
}

func (s *Serving) launch() {
	// init names
	for p, _ := range s.models.paths {
		file, err := LatestModel(p)
		if err != nil {
			log.Panicln(err)
		}
		if err = s.UpdateMeta(file); err != nil {
			log.Panicln(err)
		}
	}
	// watch
	watch, err := fsnotify.NewWatcher()
	if err != nil {
		log.Panicln(err)
	}
	for p, _ := range s.models.paths {
		err := watch.Add(p)
		if err != nil {
			log.Panicln(err)
		}
	}
	s.watcher = watch
	s.launched = true
}

func (s *Serving) Close() error {
	return s.watcher.Close()
}

func (s *Serving) Watch() {
	for {
		select {
		case ev := <-s.watcher.Events:
			if ev.Op&fsnotify.Create == fsnotify.Create {
				time.Sleep(500 * time.Millisecond) // ensure file completed
				err := s.UpdateMeta(ev.Name)
				if err != nil {
					log.Printf("Updated Failed: %s\n", err)
				}
			}
		case err := <-s.watcher.Errors:
			log.Println("Watch error:", err)
		}
	}
}

func (s *Serving) UpdateMeta(file string) error {
	meta := params.New(file)
	if err := meta.Load(); err != nil {
		return err
	}
	dir := path.Dir(file)
	m, ok := s.models.GetModelByPath(dir)
	if !ok {
		return UnregisteredError{name: dir, field: "paths"}
	}
	conf := m.config
	if conf.Name != meta.ModelName() {
		return NotMatchError{expected: conf.Name, provided: meta.ModelName()}
	}

	s.lock.Lock()
	atomic.StorePointer(&(m.meta), unsafe.Pointer(meta))
	m.startTime = time.Now()
	s.lock.Unlock()
	return nil
}

func (s *Serving) Request(name string, feats model.Features) (tensor.Tensor, error) {
	m, err := s.GetModel(name)
	if err != nil {
		return nil, err
	}
	out, err := m.config.Model.Predict(m.GetMeta(), feats) // todo: check meta
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (s *Serving) GetModel(name string) (*servingModel, error) {
	m, ok := s.models.GetModelByName(name)
	if !ok {
		return nil, NotFoundError{name: name, field: "models"}
	}
	return m, nil
}

func LatestModel(dir string) (string, error) {
	entries, err := os.ReadDir(dir)
	var file string
	if err != nil {
		return file, err
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		if file == "" {
			file = e.Name()
			continue
		}
		if e.Name() > file {
			file = e.Name()
		}
	}
	if file == "" {
		return file, EmptyDirError{dir: dir}
	}
	file = path.Join(dir, file)
	return file, nil
}
