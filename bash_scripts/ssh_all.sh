#!/bin/bash
args=("$@")
CMD=${args[0]}

ips=()
while IFS="" read -r p || [ -n "$p" ]
do
	echo $p
	ips+=($p)
done < machinefile

for ip in "${arr[@]}"
do
	echo $ip
	ssh -o StrictHostKeyChecking=no $ip $CMD
done

echo "DONE!!!"