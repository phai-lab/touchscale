<h2 align="center"><a href="https://touch-scale.github.io/">TouchScale: 500 Hours of Human Vision and Touch for Visual–Tactile Learning</a></h2>

<h5 align="center">

[![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-b31b1b.svg?logo=arXiv)](https://touch-scale.github.io/) [![Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-blue)](https://huggingface.co/datasets/2077AIDataFoundation/TouchScale)
[![Home Page](https://img.shields.io/badge/Project-Website-green.svg)](https://touch-scale.github.io/) [![Blog](https://img.shields.io/badge/Blog-Overfit%20Lab-orange.svg)](https://www.overfitlab.ai/research/touchscale)
</h5>

<div align="center">
This repository is the official code release for TouchScale, a 500-hour egocentric visual–tactile dataset of everyday human manipulation,
recorded with a wearable RGB-D camera, two wrist cameras, and bimanual tactile gloves.
</div>
<br>

<p align="center">
  <a href="https://touch-scale.github.io/"><img src="assets/touchscale_wall.webp" width="100%" alt="TouchScale video wall: twenty egocentric recordings of gloved hands manipulating objects"></a>
  <br>
  <em>500 hours across laboratories, kitchens, workbenches, and everyday environments.</em>
</p>

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

| Path | Description |
|---|---|
| [`qc/`](qc/) | The automatic data-quality pipeline used to screen recordings: a timestamp sync gate across all six sensor streams, and a VLM-assisted check for missing tactile signal and baseline noise. See [`qc/README.md`](qc/README.md). |

## Quick start (QC)

```bash
cd qc
pip install -r requirements.txt          # requires ffmpeg / ffprobe
python check_sync.py --root /path/to/recordings
cp .env.example .env                     # add a Gemini API key for the tactile QC stage
python run_qc.py --data /path/to/recordings --out qc_out
```

## License

Code in this repository is released under the [MIT License](LICENSE).
