for i in {1..16}
do
    python /home/Adama/dataloading/src/performance/multiprocess-decode.py ${i}
    sleep 10s
done