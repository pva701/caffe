cat $1/description.txt | head -n 8079 > $1/train.txt
cat $1/description.txt | tail -n 1000 > $1/test.txt
