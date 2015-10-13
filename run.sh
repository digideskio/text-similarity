python similarity.py datasets/train/STS.input.MSRpar.txt
python similarity.py datasets/train/STS.input.MSRvid.txt
python similarity.py datasets/train/STS.input.SMTeuroparl.txt

echo
echo
echo 'Pearson correlation for each train file:'
echo
echo 'MSRpar:'
perl datasets/train/correlation.pl datasets/train/STS.gs.MSRpar.txt output/train/STS.output.MSRpar.txt 
echo 'MSRvid'
perl datasets/train/correlation.pl datasets/train/STS.gs.MSRvid.txt output/train/STS.output.MSRvid.txt
echo 'SMTeuroparl:'
perl datasets/train/correlation.pl datasets/train/STS.gs.SMTeuroparl.txt output/train/STS.output.SMTeuroparl.txt