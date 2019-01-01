T=./trans.txt
G=./gold.txt

perl ./tools/multi-bleu.perl ${G} < ${T} > bleu.txt
