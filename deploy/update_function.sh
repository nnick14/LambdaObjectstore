#!/bin/bash

BASE=`pwd`/`dirname $0`
PREFIX="CacheNode"
KEY="lambda"
cluster=500
mem=3008

S3="infinistore-xiaolong"

cd $BASE/../lambda
echo "Compiling lambda code..."
GOOS=linux go build
echo "Compressing file..."
zip $KEY $KEY
echo "Putting code zip to s3"
aws s3api put-object --bucket ${S3} --key $KEY.zip --body $KEY.zip

echo "Updating lambda functions.."
go run $BASE/deploy_function.go -S3 ${S3} -code -config -prefix=$PREFIX -vpc -key=$KEY -to=$cluster -mem=$mem -timeout=$1
rm $KEY*
