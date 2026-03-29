# SDTR_STGAT

這是一份只包含 `STGAT` 訓練所需內容的自包含 bundle，可直接上傳到 H100 訓練平台。

## 內容

- `train_predictor.py`
- `stgat_model.py`
- `data_loader.py`
- `environment-h100.yml`
- `create_nano5_env.sh`
- `job.sb`
- `submit_nano5_stgat.slurm`
- `run_h100.sh`
- `data/`
  - `adjacency_matrix.npy`
  - `edge_index.npy`
  - `edge_lengths_osrm.npy`
  - `edge_lengths.npy`
  - `node_demand.npy`
  - `node_supply.npy`
  - `edge_speeds.npy`
  - `time_meta.csv`
  - `zone_info.csv`

## Nano5 建議用法

這份 bundle 已經特別整理成適合臺灣國網中心 iService Nano5（H100）用 Slurm 送件的形式。

### 1. 上傳 bundle

把整個 `SDTR_STGAT/` 上傳到 Nano5 的工作目錄，例如：

```bash
/work/<your_account>/SDTR_STGAT
```

### 2. 建立環境

登入 Nano5 後，先在 bundle 目錄下建立 conda 環境：

```bash
bash create_nano5_env.sh
```

這個腳本會使用 Nano5 手冊建議的 `ml load miniconda3/24.11.1` 與 `conda run` 工作流。

### 3. 用 Slurm 提交正式訓練

最推薦直接用：

```bash
sbatch -A <PROJECT_ID> submit_nano5_stgat.slurm
```

如果你習慣傳統檔名，也可以直接：

```bash
sbatch -A <PROJECT_ID> job.sb
```

預設會用 `normal` partition、1 個節點、1 張 H100。

如果你只想先做 smoke test，可改用：

```bash
sbatch -A <PROJECT_ID> -p dev --time=02:00:00 submit_nano5_stgat.slurm
```

### 4. 查看狀態

```bash
squeue -u $USER
sacct -X
```

### 5. 取回結果

訓練結果會寫到：

```bash
SDTR_STGAT/runs/nano5_<job_id>/
```

把裡面的 `stgat_best.pt`、`stgat_meta.json` 等檔案下載回原專案的 `runs/` 即可接回原本流程。

## 其他一般用法

如果平台支援 conda：

```bash
conda env create -f environment-h100.yml
conda activate sdtr-stgat-h100
bash run_h100.sh
```

如果平台本身已提供 CUDA 版 PyTorch 映像：

```bash
bash run_h100.sh
```

## 預設訓練命令

`run_h100.sh` 預設等價於：

```bash
python train_predictor.py \
  --data-dir ./data \
  --log-dir ./runs \
  --device auto \
  --precision bf16 \
  --batch-size 16
```

訓練資料切分固定採用：

- 每月 `1-20` 日：`train`
- 每月 `21-24` 日：`val`
- 每月 `25` 日之後：`test`

跨到不同切分區間的預測窗口會自動略過，避免 train / val / test 混到同一筆 target。

如果你想改參數，可以直接把額外參數接在後面：

```bash
bash run_h100.sh --epochs 100 --batch-size 32
```

## 路徑說明

這個專案已經把訓練需要的資料放進 `data/`，並且 `run_h100.sh` 會自動用腳本所在目錄組出絕對路徑，所以不需要依賴原本 `d:/STDR` 的工作目錄。

只要整個 `SDTR_STGAT/` 目錄結構保持不變，就不會因為換平台而出現相對路徑問題。
