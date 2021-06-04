python -u -m experiments.double_and_add --epochs 2 &> out/double_and_add_bidir.out
python -u -m experiments.double_and_add --forward_only --epochs 50 &> out/double_and_add_fw_only.out
python -u -m experiments.double_and_add_plot_gen
