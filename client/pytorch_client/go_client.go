package main

// Need to run: go build -o ecClient.so -buildmode=c-shared go_client.go

/*
#include <stdlib.h>
*/
import "C"
import (
	"flag"
	"strings"
	"unsafe"

	"github.com/mason-leap-lab/infinicache/client"
)

var (
	d        = flag.Int("d", 10, "data shard")
	p        = flag.Int("p", 2, "parity shard")
	c        = flag.Int("c", 10, "max concurrency")
	addrList = "127.0.0.1:6378"
	cli      *client.PooledClient
)

//export getFromCache
func getFromCache(cacheKeyC *C.char) *C.char {
	cacheKeyGo := C.GoString(cacheKeyC)

	reader, issue := cli.Get(cacheKeyGo)
	if issue != nil {
		return C.CString("-1")
	}

	buf, _ := reader.ReadAll()
	return C.CString(string(buf))
}

//export setInCache
func setInCache(cacheKeyC *C.char, inputDataC *C.char, arrayLen C.int) {
	cacheKeyGo := C.GoString(cacheKeyC)
	// valBytes := C.GoBytes(unsafe.Pointer(inputDataC), arrayLen)
	valBytes := unsafe.Slice((*byte)(unsafe.Pointer(inputDataC)), int(arrayLen))

	cli.Set(cacheKeyGo, valBytes)
}

//export initializeVars
func initializeVars() {
	flag.Parse()

	cli = client.NewPooledClient(strings.Split(addrList, ","), func(pc *client.PooledClient) {
		pc.NumDataShards = *d
		pc.NumParityShards = *p
		pc.Concurrency = *c
	})
}

//export close
func close() {
	if cli != nil {
		cli.Close()
		cli = nil
	}
}

func main() {}
