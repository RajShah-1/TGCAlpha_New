#!/bin/bash
args=("$@")
CMD=${args[0]}

while IFS="" read -r p || [ -n "$p" ]
do
	ssh -o StrictHostKeyChecking=no $p $CMD &
done < machinefile
wait
echo "DONE!!!"