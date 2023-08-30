# multi_view_zero_shot_open_intent_induction

## Train MBD Classifier
1. Modify dataset.py line 39 <path_to_dataset> to local path
2. Train MBD classifier
```
python main.py -e 100 -l 5e-6 -w 1e-2 -s 20 -m 0.0
```
## Load Bert layer in MBD Classifier
1. Modify MBD_Model.py line 55 <path_to_saved_model_pkl_file> to local path
2. Modify utils.py load_pretrained_arcface function


## Run Inference using official code and our model
1. Uncomment one of the experiments you want to run in either </inference/configs/run-open-intent-induction-baselines.jsonnet> or </inference/configs/run-intent-clustering-baselines.jsonnet> 
2. To Run Intent clustering (Task 1) tye the underneath code
```
python3 -m sitod.run_experiment \
--data_root_dir dstc11 \
--experiment_root_dir results \
--config configs/run-intent-clustering-baselines.jsonnet
```
3. To Run Intent Induction (Task 2) tye the underneath code
```
python3 -m sitod.run_experiment \
--data_root_dir dstc11 \
--experiment_root_dir results \
--config configs/run-open-intent-induction-baselines.jsonnet
```

### TBD
