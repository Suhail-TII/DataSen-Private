#!/bin/bash

# echo "Running TABLE 1..."
# python table1_ds.py \
#         --ft_path DataSentinel_Models/detector_large/checkpoint-5000/ \
#         --data_num 100 \
#         --save_dir results/table1 \
#         --repo_dir .

# echo "Running TABLE 2..."
# python table2_ds.py \
#         --ft_path DataSentinel_Models/detector_large/checkpoint-5000/ \
#         --data_num 100 \
#         --save_dir results/table2 \
#         --repo_dir . \
#         --pleak_mode combined

# echo "Training SSFTD..."

# python train_ssftd.py \
#     --variant ssftd \
#     --save_dir results/ssftd_checkpoint \
#     --repo_dir .

# echo "Training SSFTD_G..."
# # We were rejected access to Meta-Llama-3-8B-Instruct, so we use Meta-Llama-3.1-8B-Instruct instead.
# python train_ssftd.py \
#     --variant ssftd_g \
#     --save_dir results/ssftd_g_checkpoint \
#     --repo_dir . \
#     --backend_model_path meta-llama/Meta-Llama-3.1-8B-Instruct

# echo "Reproducing Table 3..."

# python table3_ds.py \
#     --ft_path DataSentinel_Models/detector_large/checkpoint-5000/ \
#     --ssftd_path results/ssftd_checkpoint \
#     --ssftd_g_path results/ssftd_g_checkpoint \
#     --baselines evd,nllmd,ssftd,ssftd_g,kad,promptguard,datasentinel \
#     --save_dir results/table3 \
#     --repo_dir .

# echo "Running TABLE 4(a) col 2 — LLaMA2-7B as detection LLM (~3 hrs)"
# python train_datasentinel.py \
#     --detection_model meta-llama/Llama-2-7b-hf \
#     --backend_model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --save_dir results/ds_detect_llama2 \
#     --repo_dir .

# echo "Running TABLE 4(a) col 3 — LLaMA3-8B-Instruct as detection LLM (~3 hrs)"
# python train_datasentinel.py \
#     --detection_model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --backend_model meta-llama/Meta-Llama-3.1-8B-Instruct \
#     --save_dir results/ds_detect_llama3 \
#     --repo_dir .

# echo "Running TABLE 4(b) col 1 — Mistral-7B as backend LLM (~3 hrs)"
# python train_datasentinel.py \
#     --detection_model mistralai/Mistral-7B-v0.1 \
#     --backend_model mistralai/Mistral-7B-v0.1 \
#     --save_dir results/ds_backend_mistral \
#     --repo_dir .

# echo "Running TABLE 4(b) col 2 — LLaMA2-7B as backend LLM (~3 hrs)"
# python train_datasentinel.py \
#     --detection_model mistralai/Mistral-7B-v0.1 \
#     --backend_model meta-llama/Llama-2-7b-hf \
#     --save_dir results/ds_backend_llama2 \
#     --repo_dir .

# python table4_ds.py \
#     --ckpt_mistral7b   DataSentinel_Models/detector_large/checkpoint-5000/ \
#     --ckpt_llama2_det  results/ds_detect_llama2/final \
#     --ckpt_llama3_det  results/ds_detect_llama3/final \
#     --ckpt_mistral_bk  results/ds_backend_mistral/final \
#     --ckpt_llama2_bk   results/ds_backend_llama2/final \
#     --data_num 100 \
#     --save_dir results/table4

# python table5_ds.py \
#     --ft_path DataSentinel_Models/detector_large/checkpoint-5000/ \
#     --repo_dir . \
#     --data_num 100 \
#     --save_dir results/table5

# python train_datasentinel_min.py \
#     --detection_model mistralai/Mistral-7B-v0.1 \
#     --save_dir results/ds_min \
#     --repo_dir .

# python table6_ds.py \
#     --ckpt_minimax DataSentinel_Models/detector_large/checkpoint-5000/ \
#     --ckpt_min     results/ds_min/final \
#     --repo_dir     . \
#     --data_num     100 \
#     --save_dir     results/table6

# ------------------------------------------------------------
# PromptLocate
# The checkpoint name 7_1000_naive_minimax_mistral_3_1_backend_llama3_revise_beta_1_balance_finetune decodes as: 
# 7 tasks, 1000 training samples
# naive attack used during inner-max training
# minimax optimization (same as DataSentinel's formulation)
# mistral_3_1 detector, backend_llama3 for attack generation
# balance_finetune = the segment augmentation that makes it the oracle
# ------------------------------------------------------------

python table1_pl.py \
    --detect_ft_path DataSentinel_Models/detector_large/checkpoint-5000/ \
    --locate_ft_path 7_1000_naive_minimax_mistral_3_1_backend_llama3_revise_beta_1_balance_finetune/checkpoint-500 \


# python table6_pl.py \
#     --detect_ft_path DataSentinel_Models/detector_large/checkpoint-5000/ \
#     --locate_ft_path 7_1000_naive_minimax_mistral_3_1_backend_llama3_revise_beta_1_balance_finetune/checkpoint-500/ \
#     --data_num 100 \
#     --seed 42 \
#     --output_json results_table6.json

# python smoke_test_attackers.py \
#  --detect_ft_path DataSentinel_Models/detector_large/checkpoint-5000/ \
#  --locate_ft_path 7_1000_naive_minimax_mistral_3_1_backend_llama3_revise_beta_1_balance_finetune/checkpoint-500 \

   

# ## EXTENSION WORK FOR THESIS
# # Game theoritic attacks
# python game_theoretic_attacks.py \
#     --attack all \
#     --ckpt_detector DataSentinel_Models/detector_large/checkpoint-5000/ \
#     --repo_dir . \
#     --data_num 100 \
#     --save_dir results/game_theoretic_attacks 2>&1 | tee results/game_theoretic_attacks.log

# python gt_semantic_attack.py \
#     --ckpt_detector DataSentinel_Models/detector_large/checkpoint-5000/ \
#     --paraphrase_model meta-llama/Meta-Llama-3-8B-Instruct \
#     --repo_dir . --data_num 100 \
#     --num_paraphrases 20 --mw_rounds 100 --mw_eta 0.3 \
#     --ablation \
#     --save_dir results/gt_semantic