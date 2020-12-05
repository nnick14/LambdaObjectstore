package cluster

import (
	"errors"
	"sync"

	"github.com/mason-leap-lab/infinicache/common/logger"
	"github.com/mason-leap-lab/infinicache/proxy/config"
	"github.com/mason-leap-lab/infinicache/proxy/global"
	"github.com/mason-leap-lab/infinicache/proxy/lambdastore"
	"github.com/mason-leap-lab/infinicache/proxy/types"
)

var (
	ErrOutOfBound = errors.New("instance is not active")
)

type Group struct {
	all     []*GroupInstance
	buff    []*GroupInstance
	idxBase int // The base for logic index. The base will increase every time "expire" is called.
	mu      sync.RWMutex

	log logger.ILogger
}

type GroupInstance struct {
	types.LambdaDeployment
	group *Group
	idx   GroupIndex
}

func (gins *GroupInstance) Idx() int {
	return gins.idx.Idx()
}

func (gins *GroupInstance) Instance() *lambdastore.Instance {
	ins, _ := gins.LambdaDeployment.(*lambdastore.Instance)
	return ins
}

type GroupIndex interface {
	Idx() int
}

type DefaultGroupIndex int

func (i DefaultGroupIndex) Idx() int {
	return int(i)
}

func (i *DefaultGroupIndex) Next() DefaultGroupIndex {
	return *i + 1
}

func (i *DefaultGroupIndex) NextN(n int) DefaultGroupIndex {
	return *i + DefaultGroupIndex(n)
}

func NewGroup(num int) *Group {
	return &Group{
		all:  make([]*GroupInstance, num, config.LambdaMaxDeployments),
		buff: make([]*GroupInstance, 0, config.LambdaMaxDeployments),
		log:  global.GetLogger("Group: "),
	}
}

func (g *Group) Len() int {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return len(g.all)
}

func (g *Group) StartIndex() DefaultGroupIndex {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return DefaultGroupIndex(g.idxBase)
}

func (g *Group) EndIndex() DefaultGroupIndex {
	g.mu.RLock()
	defer g.mu.RUnlock()

	return DefaultGroupIndex(g.idxBase + len(g.all))
}

func (g *Group) Expand(n int) (DefaultGroupIndex, error) {
	g.mu.Lock()
	defer g.mu.Unlock()

	g.all = g.checkCapacity(g.all, len(g.all)+n, true)
	g.all = g.all[0 : len(g.all)+n]
	return DefaultGroupIndex(g.idxBase + len(g.all)), nil
}

func (g *Group) Expire(n int) error {
	if n > len(g.all) {
		return errors.New("not enough instance to be expired")
	}

	// Force bin packing
	g.mu.Lock()
	g.buff = g.checkCapacity(g.buff, len(g.all)-n, false)
	g.buff = g.buff[:len(g.all)-n]
	copy(g.buff, g.all[n:len(g.all)])
	g.idxBase += n
	g.all, g.buff = g.buff, g.all
	g.mu.Unlock()

	g.log.Debug("expired %d of %d instances, new base %d, left %d\n", n, len(g.all)+n, g.idxBase, len(g.all))

	return nil
}

func (g *Group) All() []*lambdastore.Instance {
	g.mu.RLock()
	all := make([]*lambdastore.Instance, len(g.all))
	for i := 0; i < len(all); i++ {
		all[i] = g.all[i].Instance()
	}
	g.mu.RUnlock()

	return all
}

func (g *Group) SubGroup(start GroupIndex, end GroupIndex) []*GroupInstance {
	g.mu.RLock()
	sub := make([]*GroupInstance, end.Idx()-start.Idx())
	copy(sub, g.all[start.Idx()-g.idxBase:end.Idx()-g.idxBase])
	g.mu.RUnlock()
	return sub
}

func (g *Group) Reserve(idx GroupIndex, d types.LambdaDeployment) *GroupInstance {
	return &GroupInstance{d, g, idx}
}

func (g *Group) Set(ins *GroupInstance) {
	g.mu.RLock()
	g.all[ins.Idx()-g.idxBase] = ins
	g.mu.RUnlock()
}

func (g *Group) Instance(idx GroupIndex) *lambdastore.Instance {
	g.mu.RLock()
	ret := g.all[idx.Idx()-g.idxBase]
	g.mu.RUnlock()
	return ret.LambdaDeployment.(*lambdastore.Instance)
}

func (g *Group) checkCapacity(buff []*GroupInstance, num int, preserve bool) []*GroupInstance {
	if cap(buff) < num {
		newBuff := make([]*GroupInstance, 0, ((num-1)/cap(buff)+1)*cap(buff)) // Ceil num/cap * cap
		if preserve {
			copy(newBuff[:len(buff)], buff)
			return newBuff[:len(buff)]
		} else {
			return newBuff
		}
	}

	return buff
}
