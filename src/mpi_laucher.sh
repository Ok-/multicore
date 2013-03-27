#!/bin/bash

BINROOT=/comptes/goualard-f/local/bin

host_name="$(hostname)"
MPIRUN="$BINROOT/mpirun -H $host_name,$host_name -n 2 optimization-mpi"

$($MPIRUN)
