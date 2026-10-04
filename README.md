# TouchScale

**500 hours of human vision and touch for visual-tactile learning.**

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
