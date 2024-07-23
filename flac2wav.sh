folder=$1
nthreads=8
find $folder -name *.flac -print0 | xargs -0 -n 1 -P $nthreads -I file ./flac2wav.sh file
