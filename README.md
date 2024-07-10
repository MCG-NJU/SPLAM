# Sub-path Linear Approximation Model

Official Repository of the paper: [Accelerating Image Generation with Sub-path Linear Approximation Model](https://arxiv.org/abs/2404.13903)

Project Page: https://subpath-linear-approx-model.github.io/


## Usage

### Environment Setting

Install [diffusers](https://github.com/huggingface/diffusers) library from source:

```bash
git clone https://github.com/huggingface/diffusers
cd diffusers
pip install -e .
```

Install required packages:

```bash
pip install -r requirements.txt
```

Initialize an [🤗Accelerate](https://github.com/huggingface/accelerate/) environment with:

```bash
accelerate config
```

### Example of Lanching a Training

The following uses the [Conceptual Captions 12M (CC12M)](https://github.com/google-research-datasets/conceptual-12m) dataset as an example, and for illustrative purposes only. For best results you may consider large and high-quality text-image datasets such as [LAION](https://laion.ai/blog/laion-400-open-dataset/). You may also need to search the hyperparameter space according to the dataset you use.

```bash
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="path/to/saved/model"

accelerate launch train_splam_distill_sd_wds.py \
    --pretrained_teacher_model=$MODEL_NAME \
    --output_dir=$OUTPUT_DIR \
    --mixed_precision=fp16 \
    --resolution=512 \
    --learning_rate=8e-6 --loss_type="huber" --ema_decay=0.95 --adam_weight_decay=0.0 \
    --max_train_steps=1000 \
    --max_train_samples=4000000 \
    --dataloader_num_workers=8 \
    --train_shards_path_or_url="pipe:curl -L -s https://huggingface.co/datasets/laion/conceptual-captions-12m-webdataset/resolve/main/data/{00000..01099}.tar?download=true" \
    --validation_steps=200 \
    --checkpointing_steps=200 --checkpoints_total_limit=10 \
    --train_batch_size=12 \
    --gradient_checkpointing --enable_xformers_memory_efficient_attention \
    --gradient_accumulation_steps=1 \
    --use_8bit_adam \
    --resume_from_checkpoint=latest \
    --report_to=wandb \
    --seed=453645634 \
    --push_to_hub
```


## BibTex
```bibtex
@misc{xu2024acceleratingimagegenerationsubpath,
      title={Accelerating Image Generation with Sub-path Linear Approximation Model}, 
      author={Chen Xu and Tianhui Song and Weixin Feng and Xubin Li and Tiezheng Ge and Bo Zheng and Limin Wang},
      year={2024},
      eprint={2404.13903},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
}
```
