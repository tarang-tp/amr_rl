.PHONY: smoke pretrain train-fast train-full eval clean

DEVICE ?= cpu
EC50   := checkpoints/ec50_predictor.pt

smoke:
	python scripts/smoke_test.py

pretrain:
	python -m resistance.pretraining.pretrain_resistance \
		--mock --epochs 30 --device $(DEVICE) \
		--output checkpoints/resistance_pretrained.pt

train-fast:
	python scripts/train.py \
		--config config.yaml \
		--total_timesteps 200000 \
		--seed 42 \
		--device $(DEVICE)

train-full: pretrain
	python -m training.agents.fixed_resistance_agent \
		--resistance_mode static --total_timesteps 1000000 \
		--seed 42 --device $(DEVICE)
	python scripts/train.py --config config.yaml \
		--total_timesteps 1000000 --seed 42 --device $(DEVICE) \
		--ec50_predictor $(EC50)
	python scripts/train.py --config config.yaml \
		--total_timesteps 1000000 --seed 7  --device $(DEVICE) \
		--ec50_predictor $(EC50)
	python scripts/train.py --config config.yaml \
		--total_timesteps 1000000 --seed 21 --device $(DEVICE) \
		--ec50_predictor $(EC50)

eval:
	python scripts/evaluate.py \
		--config config.yaml \
		--policy_path checkpoints/best/best_model.zip \
		--fixed_policy_path checkpoints/fixed_static_best/best_model.zip \
		--adversary_path checkpoints/adversary_final.pt \
		--training_history runs/training_history.json \
		--output_dir results/ \
		--n_episodes 200

clean:
	rm -rf runs/ checkpoints/ results/
