python -u -m experiments.twenty_four --dataset_gen
python -u -m experiments.twenty_four --forward_only --epochs 2 &> out/double_and_add_fw_only.out
python -u -m experiments.double_and_add_plot_gen
