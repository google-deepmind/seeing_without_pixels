# Seeing without Pixels: Perception from Camera Trajectories
[**Seeing without Pixels: Perception from Camera Trajectories**](https://arxiv.org/abs/2511.21681)
Zihui Xue, Kristen Grauman, Dima Damen, Andrew Zisserman, Tengda Han
arXiv, 2024.
[project page](https://sites.google.com/view/seeing-without-pixels/) | [arxiv](https://arxiv.org/abs/2511.21681) | [bibtex](#citation)

##  Setup
```bash
conda env create -f environment.yml -n camformer
```
**Data**: download and unzip `final_data.zip` ([link](https://drive.google.com/file/d/1zYV8LJQBeeHjNvYgIFD5HGygWqbZ2g97/view?usp=sharing)) to the project folder.

## Pretraining

### Training
```bash
# EgoExo4D
python train.py --dataset egoexo4d_pretrain_longseq --take_duration 8 --sample_dur --encode_pose 11

# DynPose-100K
python train.py --dataset dynpose_pretrain --method gt --encode_pose 2  # original pose
python train.py --dataset dynpose_pretrain --method vipe --encode_pose 2    # vipe pose
```

### Eval
```bash
# EgoExo4D
python train.py --dataset egoexo4d_pretrain_longseq --scenario all --test --num_gpus 1 --batch_size 1000 --take_duration 8 --sample_dur --encode_pose 11 --ckpt ~/final_data/checkpoints/egoexo4d/bs1024_sampleddur8_pose11/checkpoints/best-epoch=498-val_loss=5.6122.ckpt

# Nymeria
python train.py --dataset nymeria_pretrain --test --num_gpus 1 --batch_size 1000 --encode_pose 11 --ckpt ~/final_data/checkpoints/egoexo4d/bs1024_sampleddur16_pose11/checkpoints/best-epoch=272-val_loss=5.6991.ckpt --text_column a/b/c/d

# DynPose-100K
python train.py --dataset dynpose_pretrain --batch_size 5000 --test --num_gpus 1 --eval_data samevideo  --method gt --ckpt ~/final_data/checkpoints/dynpose100k/v1_gt/checkpoints/best-epoch=188-val_loss=6.2618.ckpt   # original pose
python train.py --dataset dynpose_pretrain --method vipe --batch_size 5000 --test --num_gpus 1 --eval_data samevideo --ckpt ~/final_data/checkpoints/dynpose100k/v1_vipe/checkpoints/best-epoch=315-val_loss=6.0542.ckpt    # vipe pose
```

The generated features will be saved in `final_data/retrieval_features/ours/`, Run `python baselines/compute_metrics.py` to read the retrieval numbers.


## Downstream
### Scenario classification
```bash
python train_ar.py --dataset egoexo4d_scenario --num_gpus 1 --batch_size 128 --take_duration 16 --test_take_duration 16 --sample_dur --encode_pose 11 --init_ckpt ~/final_data/checkpoints/egoexo4d/bs1024_sampleddur16_pose11/checkpoints/best-epoch=272-val_loss=5.6991.ckpt
```

### Scenario classification (subset, 4-second clip)
```bash
# no init
python train_ar.py --dataset egoexo4d_scenario_subset --num_gpus 1 --batch_size 128 --encode_pose 2 --umeyama_transform --method gt/megasam/vipe/pi3
# init with our pretrained ckpt
python train_ar.py --dataset egoexo4d_scenario_subset --num_gpus 1 --batch_size 128 --encode_pose 2 --init_ckpt ~/data/logs/egoexo4d_pretrain_longseq/bs1024_dur4_pose2_sr4_new/checkpoints/best-epoch=469-val_loss=5.6075.ckpt --method gt/megasam/vipe/pi3
# inference
python train_ar.py --dataset egoexo4d_scenario_subset --ckpt /home/sherryxue_google_com/data/logs/egoexo4d_scenario_subset_8label/encodepose11_lr1e-4wd1e-3/vipe/checkpoints/best-epoch=182-val_acc=0.6158.ckpt --encode_pose 11 --method vipe --umeyama_transform --test
```

### Keystep recognition (downsampled 5FPS clip)
```bash
python tasks/extract_features.py --task cls_subset --window_size 20 --window_stride 2 --init_ckpt ~/data/logs/egoexo4d_pretrain_longseq/bs1024_dur4_pose2_sr4_new/checkpoints/best-epoch=469-val_loss=5.6075.ckpt --umeyama_transform --method gt/megasam/vipe/pi3
```
Then run `python downstream/linearSVM.py` to read the linear SVM classification numbers.


## Citation

If you find our work inspiring or use our codebase in your research, please consider giving a star ⭐ and a citation.

```
@article{xue2025seeing,
  title={Seeing without Pixels: Perception from Camera Trajectories},
  author={Xue, Zihui and Grauman, Kristen and Damen, Dima and Zisserman, Andrew and Han, Tengda},
  journal={arXiv preprint arXiv:2511.21681},
  year={2025}
}
```
