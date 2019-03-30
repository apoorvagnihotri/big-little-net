#! /bin/bash 

folder=val
i=0
while read -r -a REPLY;
do
  i=$((i+1))
  echo $i $folder ${REPLY[1]}
  mkdir -p $folder/${REPLY[1]}
  mv $folder/images/${REPLY[0]} $folder/${REPLY[1]}/
done
