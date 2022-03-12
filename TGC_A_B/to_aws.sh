#!/bin/bash
args=("$@")
FILE_NAME=${args[0]}
scp $FILE_NAME ec2-user@18.188.161.247:~/$FILE_NAME