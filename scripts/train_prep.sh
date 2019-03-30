#! /bin/bash
folder=train
i=0
for sub_folder in $(ls $folder)
do
        i=$((i+1))
        echo $i $folder $sub_folder
        im_fl=$folder/$sub_folder/images
        mv $im_fl/*.JPEG $folder/$sub_folder
done

