#!/bin/sh

for path in $(cat $1); do
    echo ${path/\/eos\/uscms/}
done