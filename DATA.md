# Data Release

This repository includes the metadata CSVs used by the loaders. The larger artifacts, including checkpoints, precomputed retrieval features, and camera-pose trajectories, are hosted on Google Drive.

No video frames are included in this release.

Download: [Google Drive folder](https://drive.google.com/drive/folders/199GH6yGt4ce_hs4bCVBiRTkQzybtxtF6?usp=sharing)

## Repository Files

The following metadata files are included directly in the repository:

| Dataset | Location | Contents |
|---------|----------|----------|
| Ego-Exo4D | `data_files/egoexo4d/` | Train, validation, Pi3-pose metadata, scenario labels, and 5-way MCQ test metadata |
| DynPose-100K | `data_files/dynpose100k/` | Train, validation, and 5-way MCQ test metadata |
| Nymeria | `data_files/nymeria/eval1000/` | Four 5-way MCQ evaluation splits, one per narration type |

These files are read automatically by the dataset loaders.

## Hosted Artifacts

| File | Size | Contents |
|------|------|----------|
| `camformer_retrieval_features.zip` | about 145 MB | Precomputed `frames.pt` and `text.pt` embeddings for all released retrieval splits |
| `camformer_checkpoints.zip` | about 30 MB | Released CamFormer encoder checkpoints |
| `egoexo4d_poses.tar` | about 15 GB | Ego-Exo4D Aria camera-pose trajectories |
| `egoexo4d_pi3_poses.tar` | about 305 MB | Ego-Exo4D Pi3-estimated camera-pose trajectories |
| `dynpose100k_poses.tar` | about 2.3 GB | DynPose-100K original and ViPE-estimated camera-pose trajectories |
| `nymeria_poses.tar` | about 162 MB | Nymeria evaluation camera-pose trajectories |

## Precomputed Retrieval Features

The precomputed features reproduce the released retrieval scores without loading a checkpoint or downloading pose trajectories.

From the repository root:

```bash
unzip camformer_retrieval_features.zip
```

Expected layout:

```text
retrieval_features/
  egoexo4d/
    frames.pt
    text.pt
  dynpose_original/
    frames.pt
    text.pt
  dynpose_vipe/
    frames.pt
    text.pt
  nymeria_a/
    frames.pt
    text.pt
  nymeria_b/
    frames.pt
    text.pt
  nymeria_c/
    frames.pt
    text.pt
  nymeria_d/
    frames.pt
    text.pt
```

Example:

```bash
python eval_retrieval.py retrieval_features/egoexo4d
```

The reported paper metric is `Motion->Text MCQ acc`.

## Checkpoints

From the repository root:

```bash
unzip camformer_checkpoints.zip
```

Expected layout:

```text
checkpoints/
  egoexo4d_dur8.pt
  egoexo4d_dur16.pt
  dynpose100k_original.pt
  dynpose100k_vipe.pt
```

Checkpoint usage:

| Checkpoint | Evaluation setting | Pose encoding |
|------------|--------------------|---------------|
| `egoexo4d_dur8.pt` | Ego-Exo4D retrieval | `rel9d_grav` |
| `egoexo4d_dur16.pt` | Nymeria zero-shot transfer | `rel9d_grav` |
| `dynpose100k_original.pt` | DynPose-100K, original poses | `rel9d` |
| `dynpose100k_vipe.pt` | DynPose-100K, ViPE poses | `rel9d` |

## Camera-Pose Trajectories

Trajectory archives should be extracted outside the repository. The commands below use `~/camformer_data`, but any location is fine.

```bash
export CAMFORMER_DATA=~/camformer_data
mkdir -p "$CAMFORMER_DATA"
```

Extract the needed archives:

```bash
tar -C "$CAMFORMER_DATA" -xf egoexo4d_poses.tar
tar -C "$CAMFORMER_DATA" -xf egoexo4d_pi3_poses.tar
tar -C "$CAMFORMER_DATA" -xf dynpose100k_poses.tar
tar -C "$CAMFORMER_DATA" -xf nymeria_poses.tar
```

Expected layout:

```text
$CAMFORMER_DATA/
  data/
    egoexo4d_pretrain/
      train_presr50_v2/
      val_presr50_v2/
    dynpose100k_pretrain/
      cameras/
      vipe_poses/
    nymeria_eval/
      presr50/
  pi3_preds/
```

Set the corresponding environment variables:

```bash
export EGOEXO4D_PRETRAIN_TRAJ_DIR="$CAMFORMER_DATA/data/egoexo4d_pretrain"
export DYNPOSE_DATA_DIR="$CAMFORMER_DATA"
export NYMERIA_DATA_DIR="$CAMFORMER_DATA"

# Required only for --use_pi3_pose.
export EGOEXO4D_PI3_TRAJ_DIR="$CAMFORMER_DATA/pi3_preds"
```

## Nymeria Text Columns

| Flag | Narration type |
|------|----------------|
| `--text_column a` | Body posture |
| `--text_column b` | Hands and arms motion |
| `--text_column c` | Legs and feet motion |
| `--text_column d` | Focus of attention |

## Notes

The trajectory archives contain derived camera-pose data, not original videos. Use of the released metadata and trajectories should follow the terms of the underlying datasets.

The code still supports fallback paths from the original internal layout, but setting the environment variables above is recommended for a fresh checkout.
