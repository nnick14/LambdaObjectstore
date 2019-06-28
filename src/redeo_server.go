package main

import (
	"fmt"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/lambda"
	"github.com/cornelk/hashmap"
	"github.com/wangaoone/redeo"
	"github.com/wangaoone/redeo/resp"
	"net"
	"strconv"
)

var (
	clientLis    net.Listener
	lambdaLis    net.Listener
	cMap         = make(map[int]chan interface{}) // client channel mapping table
	mappingTable = hashmap.New(1024)              // lambda store mapping table
	shard        = 13
	isPrint      = true
)

func main() {
	clientLis, _ = net.Listen("tcp", ":6378")
	lambdaLis, _ = net.Listen("tcp", ":6379")
	fmt.Println("start listening client face port 6378")
	fmt.Println("start listening lambda face port 6379")
	srv := redeo.NewServer(nil)
	lambdaSrv := redeo.NewServer(nil)

	// initial lambda store
	initial(lambdaSrv)

	// lambda handler
	//go lambdaHandler(lambdaStore)
	// lambda facing peeking response type
	//go myPeek(lambdaStore)

	// initial ec2 server and lambda store

	// Start serving (blocking)
	err := srv.MyServe(clientLis, cMap, mappingTable)
	if err != nil {
		fmt.Println(err)
	}
}

// initial lambda group
func initial(lambdaSrv *redeo.Server) {
	group := redeo.Group{Arr: make([]redeo.LambdaInstance, shard), C: make(chan redeo.Response, 1024*1024)}
	//group := make([]redeo.LambdaInstance, shard)
	for i := range group.Arr {
		node := newLambdaInstance("Lambda2SmallJPG")
		//node.Id = i
		myPrint("No.", i, "replication lambda store has registered")
		group.Arr[i] = *node
		go lambdaTrigger(node)
		// start a new server to receive conn from lambda store
		myPrint("start a new conn")
		node.Cn = lambdaSrv.Accept(lambdaLis)
		myPrint("lambda store has connected", node.Cn.RemoteAddr())
		// writer and reader
		node.W = resp.NewRequestWriter(node.Cn)
		node.R = resp.NewResponseReader(node.Cn)
		// lambda handler
		go lambdaHandler(node)
		fmt.Println(node.Alive)
	}
	mappingTable.Set(0, group)
	go groupReclaim(group)

}

func groupReclaim(group redeo.Group) {
	clientId := 0
	for {
		for i := 0; i < shard; i++ {
			obj := <-group.C
			id, _ := strconv.Atoi(obj.Id)
			myPrint("client id is ", id, "obj body is", obj.Body, "key is ", obj.Key)
			clientId = id
		}
		myPrint("lambda group has finished receive response from", shard, "lambdas")
		// send response body to client channel
		//cMap[id] <- obj.Body
		myPrint("client id is ", clientId)
		cMap[clientId] <- "1"
	}
}

func newLambdaInstance(name string) *redeo.LambdaInstance {
	return &redeo.LambdaInstance{
		//name:  "dataNode" + strconv.Itoa(id),
		Name:  name,
		Alive: false,
		C:     make(chan redeo.SetReq, 1024*1024),
		Peek:  make(chan redeo.Response, 1024*1024),
	}
}

// blocking on peekType, every response's type is bulk
func myPeek(l *redeo.LambdaInstance) {
	for {
		var obj redeo.Response
		// field 1 for client id
		field1, err := l.R.PeekType()
		if err != nil {
			fmt.Println("field1 err", err)
			return
		}
		switch field1 {
		case resp.TypeBulk:
			id, _ := l.R.ReadBulkString()
			obj.Id = id
		case resp.TypeError:
			err, _ := l.R.ReadError()
			fmt.Println("peek type err1 is", err)
		default:
			panic("unexpected response type")
		}
		// field 2 for obj key
		field2, err := l.R.PeekType()
		if err != nil {
			fmt.Println("field2 err", err)
			return
		}
		switch field2 {
		case resp.TypeBulk:
			key, _ := l.R.ReadBulkString()
			obj.Key = key
		case resp.TypeError:
			err, _ := l.R.ReadError()
			fmt.Println("peek type err2 is", err)
			return
		default:
			panic("unexpected response type")
		}
		// field 3 for obj body
		field3, err := l.R.PeekType()
		if err != nil {
			fmt.Println("field2 err", err)
			return
		}
		switch field3 {
		case resp.TypeBulk:
			body, _ := l.R.ReadBulkString()
			obj.Body = body
		case resp.TypeError:
			err, _ := l.R.ReadError()
			fmt.Println("peek type err2 is", err)
			return
		default:
			panic("unexpected response type")
		}
		// send obj to lambda helper channel
		l.Peek <- obj
	}
}

func lambdaHandler(l *redeo.LambdaInstance) {
	fmt.Println("conn is", l.Cn)
	for {
		select {
		case a := <-l.C: /*blocking on lambda facing channel*/
			// check lambda status first
			l.AliveLock.Lock()
			if l.Alive == false {
				myPrint("Lambda 2 is not alive, need to activate")
				// trigger lambda
				go lambdaTrigger(l)
			}
			l.AliveLock.Unlock()
			// req from client
			//myPrint("req from client is ", a.Cmd, a.Key, a.Val, a.Cid)
			// get channel id
			cid := strconv.Itoa(a.Cid)
			//myPrint("id is ", cid)
			// get cmd argument
			//argsCount := len(a.Argument.Args)
			//switch argsCount {
			//case 1: /*get or one argument cmd*/
			//	l.W.WriteCmdString(a.Cmd, a.Argument.Arg(0).String(), cid)
			//	err := l.W.Flush()
			//	if err != nil {
			//		fmt.Println("flush pipeline err is ", err)
			//	}
			//case 2: /*set or two argument cmd*/
			//	myPrint("obj length is ", len(a.Argument.Arg(1)))
			//	l.W.WriteCmdString(a.Cmd, a.Argument.Arg(0).String(), a.Argument.Arg(1).String(), cid)
			//	err := l.W.Flush()
			//	if err != nil {
			//		fmt.Println("flush pipeline err is ", err)
			//	}
			//	myPrint("write complete")
			//}
			//myPrint("obj length is ", len(a.Val))
			myPrint("val is", a.Val, "id is ", cid)
			l.W.MyWriteCmd(a.Cmd, cid, a.Key, a.Val)
			err := l.W.Flush()
			if err != nil {
				fmt.Println("flush pipeline err is ", err)
			}
			//myPrint("No.", l.Id, "replication writes complete")
			// lambda facing peeking response type
			go myPeek(l)
			//groupBy(l, a.Key)
		case obj := <-l.Peek: /*blocking on lambda facing receive*/
			//group, ok := mappingTable.Get(obj.Key)
			group, ok := mappingTable.Get(0)
			if ok == false {
				fmt.Println("get lambda instance failed")
				return
			}
			group.(redeo.Group).C <- obj
			// send response body to client channel
			//cMap[id] <- obj.Body
			//cMap[id] <- "1"
		}
	}
}

//func register(l *lambdaInstance) {
//	lambdaStoreMap[*l] = l.cn
//	myPrint("register lambda store", l.name)
//}

func lambdaTrigger(l *redeo.LambdaInstance) {
	l.Alive = true
	sess := session.Must(session.NewSessionWithOptions(session.Options{
		SharedConfigState: session.SharedConfigEnable,
	}))

	client := lambda.New(sess, &aws.Config{Region: aws.String("us-east-1")})

	_, err := client.Invoke(&lambda.InvokeInput{FunctionName: aws.String(l.Name)})
	if err != nil {
		fmt.Println("Error calling LambdaFunction", err)
	}

	myPrint("Lambda Deactivate")
	//l.AliveLock.Lock()
	l.Alive = false
	//l.AliveLock.Unlock()
}

func myPrint(a ...interface{}) {
	if isPrint == true {
		fmt.Println(a)
	}
}
