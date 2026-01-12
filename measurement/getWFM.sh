#!/bin/sh

##########
# scirpt to get waveform data from Tektronix oscilloscope
#
# #usage
# 1.make the oscilloscope connected to the LAN and get IP address
# (It is easy to get by DHCP) 
# 2.make your PC also connected to the same LAN
# 3.excute this command with following format
#  ./getWFM.sh (IP address) (comment)
# 4.then, waveform data for last triggered event
#   for each channel set in variable "ChList" will be saved in "wfmdata/run00000_1.isf"
# *Run No is automatically updated
# *png image will also be saved every 10 events
#  -this frequency can be changed by variable ImgTakeFreq)
#  -can force to get png file by option "--Image" (just after IP address)
# *Waveform data are saved in binary format.
#  -some headers in ascii format, then 10000 voltage data
#  -each voltage data has 2 byte size or 16bit precision
#
# actually operated oscilloscope model : 
#  *TDS3034B
#  *DPO3034 (variable ImageFileName should be changed to "image.png")
#
# last modified : 23rd Aug, 2012
# Maeda Yosuke, Kyoto University
# maeda_y (AT) scphys.kyoto-u.ac.jp
##########

# modify these variables if you need
ImgTakeFreq=1
ChList="1 2 3 4" #input ch No(1-4). when take data of multichannel, seprate numbers with space
ImageFileName="Image.png"

#default value
statefile="state.dat"
rawdatafilename="tmp.isf"
converteddatafilename="${rawdatafilename%isf}dat"
runNoFilename="runNo.dat"
data_dirname="wfmdata"
image_dirname="png"
log_dirname="log"

check_dir(){
    
    if [ $# -lt 1 ]
	then
	echo "check_dir : give directory name to check"
	exit 1
    fi
    
    dirname="${1}"
    
    if [ -e "${dirname}" ]
	then
	if [ -d "${dirname}" ]
	    then
	    :
	else
	    echo "no directory \"${dirname}\" slready exists."
	    exit 1
	fi
    else
	mkdir "${dirname}"
    fi
    
    return 0
}

if [ $# -lt 2 ]
    then
    echo "usage :"
    echo "${0} (IP) [--Image] (comment)"
    echo " *The option\"--Image\" forces to get the png image in the oscilloscope display."
    echo "  (image will be saved every ${ImgTakeFreq} events)"
    exit 1
fi

check_dir "${data_dirname}"
check_dir "${image_dirname}"
check_dir "${log_dirname}"

IP="${1}"
shift
if [ "${1}" = "--Image" ]
    then
    GetImage=0
    shift
else
    GetImage=1
fi
comment="$*"

if [ -e "${runNoFilename}" ]
    then
    runNo=`cat "${runNoFilename}"`
    runNo=`expr "${runNo}" + 1`
else
    touch "${runNoFilename}"
    runNo="0"
fi
runNoWithZero=`printf %05d "${runNo}"`
logfile="${log_dirname}/run${runNoWithZero}.log"
echo "${runNo}" > "${runNoFilename}"

export LANG=C date > "${logfile}"
echo "run${runNoWithZero} start -----" | tee -a "${logfile}"
echo "comment : ${comment}" | tee -a "${logfile}"
echo | tee -a "${logfile}"
echo | tee -a "${logfile}"

wget -nv --output-document="${statefile}" http://"${IP}"/?command=acquire:stopafter+sequence 2>&1 | tee -a "${logfile}"
wget -nv --output-document="${statefile}" http://"${IP}"/?command=acquire:state+1 2>&1 | tee -a "${logfile}"
#wget -nv --output-document="${statefile}" http://"${IP}"/?command=select:ch1+on 2>&1 | tee -a "${logfile}"
#wget -nv --output-document="${statefile}" http://"${IP}"/?command=select:ch2+on 2>&1 | tee -a "${logfile}"
#wget -nv --output-document="${statefile}" http://"${IP}"/?command=select:ch3+on 2>&1 | tee -a "${logfile}"
#wget -nv --output-document="${statefile}" http://"${IP}"/?command=select:ch4+on 2>&1 | tee -a "${logfile}"

# set readour file format (internal=binary,spreadsheet=csv)
wget -nv --output-document="${statefile}" http://"${IP}"/?command=save:waveform:fileformat+internal 2>&1 | tee -a "${logfile}"
#wget -nv --output-document="${statefile}" http://"${IP}"/?command=save:waveform:fileformat+spreadsheet 2>&1 | tee -a "${logfile}"


## wait until triggered

query="1"
while [ "${query}" -ne 0 ]
  do
  wget -nv --output-document="${statefile}" http://"${IP}"/?command=acquire:state? | tee -a "${logfile}"
  query=`cat "${statefile}"`
done


## save waveform data for each channel

for ch in `echo "${ChList}" | xargs`
  do
  wget -nv --output-document="${statefile}" http://"${IP}"/?command=select:ch"${ch}"+on 2>&1 | tee -a "${logfile}"
  wget -nv --output-document="${rawdatafilename}" http://"${IP}"/?wfmsend=get 2>&1 | tee -a "${logfile}"
  mv "${rawdatafilename}" wfmdata/run"${runNoWithZero}"_"${ch}".isf
done

#wget -nv --output-document="${statefile}" http://"${IP}"/?command=save:waveform:fileformat+spreadsheet 2>&1 | tee -a "${logfile}"
#for ch in `echo "${ChList}" | xargs`
#  do
#  wget -nv --output-document="${statefile}" http://"${IP}"/?command=select:ch"${ch}"+on 2>&1 | tee -a "${logfile}"
#  wget -nv --output-document="${rawdatafilename}" http://"${IP}"/?wfmsend=get 2>&1 | tee -a "${logfile}"
#  mv "${rawdatafilename}" wfmdata/run"${runNoWithZero}"_"${ch}".csv
#done

## save image

TakeImage=`expr "${runNo}" % "${ImgTakeFreq}"`
if [ "${TakeImage}" -eq 0 ] || [ "${GetImage}" -eq 0 ]
    then
    wget -nv --tries=1 http://"${IP}"/"${ImageFileName}"
    mv "${ImageFileName}" "${image_dirname}"/run"${runNoWithZero}".png
    echo "waveform image is saved in /run${runNoWithZero}.png"
fi

#wget -nv --output-document="${statefile}" http://"${IP}"/?command=acquire:stopafter+runstop 2>&1 | tee -a "${logfile}"
#wget -nv --output-document="${statefile}" http://"${IP}"/?command=acquire:state+1 2>&1 | tee -a "${logfile}"

echo | tee -a "${logfile}"
echo "----------" | tee -a "${logfile}"
export LANG=C date | tee -a "${logfile}"
echo | tee -a "${logfile}"
