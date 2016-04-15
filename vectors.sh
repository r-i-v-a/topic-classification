#!/bin/bash

workdir=$1
datadir=$2

echo 'making document vectors'
$workdir/vectors.py $datadir