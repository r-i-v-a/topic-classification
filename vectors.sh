#!/bin/bash

#$ -S /bin/bash
#$ -N riva_test
#$ -o /homes/eva/q/qnathans/topic-classification/log/sge.out
#$ -e /homes/eva/q/qnathans/topic-classification/log/sge.err
#$ -q all.q@<what to put here?>

workdir=$1
datadir=$2
cd $workdir

echo 'making document vectors'
$workdir/vectors.py $datadir