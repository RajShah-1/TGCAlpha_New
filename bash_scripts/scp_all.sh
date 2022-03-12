#!/bin/bash
args=("$@")
FILE_NAME=${args[0]}

while IFS="" read -r p || [ -n "$p" ]
do
	scp -r $FILE_NAME ec2-user@$p:/home/ec2-user/$FILE_NAME
done < machinefile
echo "DONE!!!"