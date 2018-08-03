#!/bin/bash

 #USAGE: color_balance  [-i input_dir] [-o output_dir] [-f framework]

 #OPTIONS:
# 
# -f 	  framework 	   Method used f='Image Magick', 'uwcorrect'
# -i 	  input_dir 	   Directory of flattened and cropped, non color corrected images 
# -o 	  output_dir 	   Directory where color_corrected files will be saved
while [ "$1" != "" ]; do
	case $1 in 
		-i | --input_dir)	shift 
			i=$1
		;;
		-o | --output_dir) shift
			o=$1
		;;
		-f | --framework) shift 
			f=$1
	esac
	shift
done


dialog --title 'Processing automatic image color balance' --gauge "Number of images processed..." 10 75 < <(
	# get total number of files in array
	ifiles=(`ls ${i}/*.jpg`)

	# set counter 
	count=0

	# get total number of files process
	total=${#ifiles[@]}
	
	#
	# Start loop
	#

	for img in "${ifiles[@]}"
	do 
	# calculate progress
	PCT=$(( 100*(++count)/total ))
	# update dialog box
cat <<EOF
XXX
$PCT
Images proccesed: $count
from: "$i"
to: "$o"
using: "$f"
XXX 
EOF
	
	# check what processing mehtod to use
	if [ "$f" = "uwcorrect" ]
	then
	sh /media/manu/DATAPART1/Dropbox/scripts/color_correction/uwcorrect.sh -h 15 ${i}/"${img##*/}" ${o}/"${img##*/}"
	else
	convert \( ${i}/"${img##*/}" -auto-level \) \
	\( +clone -colorspace gray +level-colors black,red \) \
	-compose screen -composite -channel rgb -auto-level -modulate 100,130,100 \
	${o}/"${img##*/}"
	fi

	
	done
)
