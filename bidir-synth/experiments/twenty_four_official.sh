# first, generate data if needed.
python -u -m experiments.twenty_four --dataset_gen
# next, train five different mixed supervised models for forward only and bidir
python -u -m experiments.twenty_four --run_supervised --forward_only &> out/24_official_sv_fw_1.out
python -u -m experiments.twenty_four --run_supervised --forward_only &> out/24_official_sv_fw_2.out
python -u -m experiments.twenty_four --run_supervised --forward_only &> out/24_official_sv_fw_3.out
python -u -m experiments.twenty_four --run_supervised --forward_only &> out/24_official_sv_fw_4.out
python -u -m experiments.twenty_four --run_supervised --forward_only &> out/24_official_sv_fw_5.out
python -u -m experiments.twenty_four --run_supervised &> out/24_official_sv_bidir_1.out
python -u -m experiments.twenty_four --run_supervised &> out/24_official_sv_bidir_2.out
python -u -m experiments.twenty_four --run_supervised &> out/24_official_sv_bidir_3.out
python -u -m experiments.twenty_four --run_supervised &> out/24_official_sv_bidir_4.out
python -u -m experiments.twenty_four --run_supervised &> out/24_official_sv_bidir_5.out

# TODO: IMPORTANT. Now, before running the RL experiments, you need to fill in
# the model ID's used in the five seeds for the policy gradient fine-tuning
# Go to line 83 of experiments/twenty_four.py and fill in the model ID's that are in the out files from the supervised models.
# then uncomment the following experiments and run them

# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 1 --seed 1 &> out/24_official_d1_fw_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 1 --seed 2 &> out/24_official_d1_fw_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 1 --seed 3 &> out/24_official_d1_fw_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 1 --seed 4 &> out/24_official_d1_fw_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 1 --seed 5 &> out/24_official_d1_fw_5.out

# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 2 --seed 1 &> out/24_official_d2_fw_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 2 --seed 2 &> out/24_official_d2_fw_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 2 --seed 3 &> out/24_official_d2_fw_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 2 --seed 4 &> out/24_official_d2_fw_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 2 --seed 5 &> out/24_official_d2_fw_5.out

# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 3 --seed 1 &> out/24_official_d3_fw_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 3 --seed 2 &> out/24_official_d3_fw_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 3 --seed 3 &> out/24_official_d3_fw_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 3 --seed 4 &> out/24_official_d3_fw_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 3 --seed 5 &> out/24_official_d3_fw_5.out

# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 4 --seed 1 &> out/24_official_d4_fw_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 4 --seed 2 &> out/24_official_d4_fw_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 4 --seed 3 &> out/24_official_d4_fw_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 4 --seed 4 &> out/24_official_d4_fw_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --forward_only --depth 4 --seed 5 &> out/24_official_d4_fw_5.out

# same thing, but bidirectional
# python -u -m experiments.twenty_four --run_policy_gradient --depth 1 --seed 1 &> out/24_official_d1_bidir_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 1 --seed 2 &> out/24_official_d1_bidir_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 1 --seed 3 &> out/24_official_d1_bidir_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 1 --seed 4 &> out/24_official_d1_bidir_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 1 --seed 5 &> out/24_official_d1_bidir_5.out

# python -u -m experiments.twenty_four --run_policy_gradient --depth 2 --seed 1 &> out/24_official_d2_bidir_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 2 --seed 2 &> out/24_official_d2_bidir_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 2 --seed 3 &> out/24_official_d2_bidir_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 2 --seed 4 &> out/24_official_d2_bidir_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 2 --seed 5 &> out/24_official_d2_bidir_5.out

# python -u -m experiments.twenty_four --run_policy_gradient --depth 3 --seed 1 &> out/24_official_d3_bidir_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 3 --seed 2 &> out/24_official_d3_bidir_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 3 --seed 3 &> out/24_official_d3_bidir_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 3 --seed 4 &> out/24_official_d3_bidir_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 3 --seed 5 &> out/24_official_d3_bidir_5.out

# python -u -m experiments.twenty_four --run_policy_gradient --depth 4 --seed 1 &> out/24_official_d4_bidir_1.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 4 --seed 2 &> out/24_official_d4_bidir_2.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 4 --seed 3 &> out/24_official_d4_bidir_3.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 4 --seed 4 &> out/24_official_d4_bidir_4.out
# python -u -m experiments.twenty_four --run_policy_gradient --depth 4 --seed 5 &> out/24_official_d4_bidir_5.out

# # TODO: uncomment this too. analyze out files to extract accuracies after policy gradient models train
# # run data analytics on the results. will print out accuracies.
# python -u -m experiments.twenty_four_analytics
