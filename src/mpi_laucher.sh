#!/bin/bash

host_name="$(hostname)"
mpirun -H $host_name,$host_name -n 2 optimization-mpi
