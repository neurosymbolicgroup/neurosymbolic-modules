# runs supervised training, tries to solve tasks with policy rollouts for 30min, then repeats with policy gradient
python -u -m experiments.arc --forward_only &> out/arc_fw.out
# same thing but with bidirectional
python -u -m experiments.arc &> out/arc_bidir.out
