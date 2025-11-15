#!/bin/bash
# Script to create all GitHub issues
# Run with: bash .github/issues/create_all_issues.sh

gh issue create --title "[TASK] Wire Ensemble World Model End-to-End" --body-file .github/issues/v0.2_01_wire_ensemble_world_model_endt.md --label "task,enhancement,v0.2"
sleep 2  # Avoid rate limiting

gh issue create --title "[TASK] Add Ensemble Support to Evaluation Script" --body-file .github/issues/v0.2_02_add_ensemble_support_to_evalua.md --label "task,enhancement,v0.2,evaluation"
sleep 2  # Avoid rate limiting

gh issue create --title "[TASK] Implement Encoder/Decoder Modules" --body-file .github/issues/v0.3_01_implement_encoder_decoder_modu.md --label "task,enhancement,v0.3,architecture"
sleep 2  # Avoid rate limiting

gh issue create --title "[TASK] Create Autoencoder Training Script" --body-file .github/issues/v0.3_02_create_autoencoder_training_sc.md --label "task,enhancement,v0.3,training"
sleep 2  # Avoid rate limiting

gh issue create --title "[TASK] Add Visual Observations to Environment" --body-file .github/issues/v0.4_01_add_visual_observations_to_env.md --label "task,enhancement,v0.4,environment"
sleep 2  # Avoid rate limiting

gh issue create --title "[TASK] Implement Stochastic World Model" --body-file .github/issues/v0.5_01_implement_stochastic_world_mod.md --label "task,enhancement,v0.5,uncertainty"
sleep 2  # Avoid rate limiting

