python -u -m experiments.twenty_four --dataset_gen
echo "Dataset gen done, now training model to out/twenty_four_exmaple_d2_fw.out"
python -u -m experiments.twenty_four --forward_only --depth 2 --run-supervised --run-policy_gradient &> out/twenty_four_example_d2_fw.out
echo "Training model done"
python -u -m experiments.twenty_four_analytics
