## [TouchScale: 500 Hours of Human Vision and Touch for Visual–Tactile Learning](https://touch-scale.github.io/)

Dayou Li1,*, Hao Wang2,*, Qianqian Yang3,*, Zihao Zhu1,*, Haoquan Fang4, Ziyao Zeng5, Yan Han6, Zihan Wang7, Yan Wang8, Baoru Huang9, Dilin Wang10, Kenji Shimada3, Yiyue Luo11, Manling Li12, Teresa Lv13, Mustafa Mukadam11, Rakesh Ranjan10, Ruohan Zhang4, Qi He6, Changliu Liu3, Xu Chen11, Marco Pavone4,8, Bangya Liu7, Jiachen Li14, Masayoshi Tomizuka15, Zhiwen Fan1,†

1Texas A&M University   2Google DeepMind   3CMU   4Stanford University   5Yale University   6Microsoft   7Overfit Lab   8NVIDIA   9University of Liverpool   10Meta   11University of Washington   12Northwestern University   13Sony   14Georgia Tech   15UC Berkeley   
*Equal contribution   †Corresponding author



![arXiv](https://img.shields.io/badge/Arxiv-2610.10288-b31b1b.svg?logo=arXiv) ![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)
![Home Page](https://img.shields.io/badge/Project-Website-green.svg) ![Blog](https://img.shields.io/badge/Blog-Overfit%20Lab-orange.svg)



This repository is the official code release for TouchScale, a 500-hour egocentric visual–tactile dataset of everyday human manipulation, recorded with a wearable RGB-D camera, two wrist cameras, and bimanual tactile gloves.

  


![TouchScale video wall: twenty egocentric recordings of gloved hands manipulating objects](assets/touchscale_wall.webp)  
*500 hours across laboratories, kitchens, workbenches, and everyday environments.*

## Overview

TouchScale is a large-scale egocentric visual-tactile dataset of everyday
manipulation, recorded in laboratories, kitchens, workbenches, and other
everyday environments. A single wearable setup records:

- a head-mounted **RGB-D camera** for the overall interaction,
- two **wrist-mounted RGB cameras** for close views of hand-object contact (both at 30 Hz),
- bimanual **tactile gloves**, each with 880 taxels across the five fingers and
palm at under 2 mm spatial resolution, recording normal and shear force.

Mid-training on TouchScale teaches a policy to predict how touch will change
before robot post-training. In real-robot experiments this raised task success
from 22.5% to 57.5% on contact-rich tasks (soft/hard sorting, bottle-cap removal,
test-tube transfer, whiteboard wiping). Scaling the data also improves zero-shot
tactile contact prediction and egocentric action recognition.

## Repository contents


| Path                   | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ---------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[qc/](qc/)`           | The automatic data-quality pipeline used to screen recordings: a timestamp sync gate across all six sensor streams, and a VLM-assisted check for missing tactile signal and baseline noise. See `[qc/README.md](qc/README.md)`.                                                                                                                                                                                                                                                               |
| `[n0-vtla/](n0-vtla/)` | The visual–tactile–action policy (a derivative of [NeoteAI's N0-VTLA](https://github.com/neoteai/N0-VTLA)) with the TouchScale additions: action-free **mid-training** of the tactile branch on human TouchScale data, and **post-training** on your own robot recordings (data conversion, normalization scripts, launcher, real-robot inference notes). Start with `[n0-vtla/docs/MID_TRAIN.md](n0-vtla/docs/MID_TRAIN.md)` and `[n0-vtla/docs/POST_TRAIN.md](n0-vtla/docs/POST_TRAIN.md)`. |




## Quick start (QC)

```bash
cd qc
pip install -r requirements.txt          # requires ffmpeg / ffprobe
python check_sync.py --root /path/to/recordings
cp .env.example .env                     # add a Gemini API key for the tactile QC stage
python run_qc.py --data /path/to/recordings --out qc_out
```



## Quick start (N0-VTLA: mid-training and post-training)

Checkpoints are expected to live in a `checkpoints/` folder next to `n0-vtla/`. Install the environment as in `[n0-vtla/docs/INSTALL.md](n0-vtla/docs/INSTALL.md)`, then:

```bash
cd n0-vtla
hf download NeoteAI/n0-vtla-base --local-dir ../checkpoints/n0-vtla-base     # original N0-VTLA weights

# 1) Mid-train the tactile branch on TouchScale recordings (details: docs/MID_TRAIN.md)
python scripts/build_per_task_scale_normalization.py --raw-root /path/to/touchscale_raw --out per_task_scale.json --workers 16
VTLA_ITW_RAW_ROOT=/path/to/touchscale_raw VTLA_ITW_NORMALIZATION=$PWD/per_task_scale.json \
  VTLA_PRETRAINED_CHECKPOINT=$PWD/../checkpoints/n0-vtla-base bash train_stage1.sh

# 2) Convert your robot recordings, then post-train (details: docs/POST_TRAIN.md)
#    INIT=original   starts from the original checkpoint
#    INIT=touchscale starts from the original checkpoint + the mid-training result of step 1
INIT=touchscale DATASET=/path/to/canonical_dataset ASSET_ID=my_asset bash scripts/posttrain.sh
```



## License

Code in this repository is released under the [MIT License](LICENSE), with one exception: the `n0-vtla/` directory is derived from [NeoteAI's N0-VTLA](https://github.com/neoteai/N0-VTLA) and remains under its original [CC BY-SA 4.0 license](n0-vtla/LICENSE) (third-party notices in `n0-vtla/NOTICE` and `n0-vtla/LICENSES/`). The N0-VTLA model weights are governed by the Gemma Terms of Use.

## Citation

If you find TouchScale useful in your research, please cite:

```bibtex
@misc{li2026touchscale500hourshuman,
      title={TouchScale: 500 Hours of Human Vision and Touch for Visual-Tactile Learning},
      author={Dayou Li and Hao Wang and Qianqian Yang and Zihao Zhu and Haoquan Fang and Ziyao Zeng and Yan Han and Zihan Wang and Yan Wang and Baoru Huang and Dilin Wang and Kenji Shimada and Yiyue Luo and Manling Li and Teresa Lv and Mustafa Mukadam and Rakesh Ranjan and Ruohan Zhang and Qi He and Changliu Liu and Xu Chen and Marco Pavone and Bangya Liu and Jiachen Li and Masayoshi Tomizuka and Zhiwen Fan},
      year={2026},
      eprint={2610.10288},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.10288},
}
```

