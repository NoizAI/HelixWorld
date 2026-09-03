<p align="center">
  <img src="assets/banner.jpg" alt="HelixWorld — Sound and vision, born together" width="100%">
</p>

# HelixWorld

**HelixWorld** is a real-time interactive audio-visual world model from [Noiz AI](https://noiz.ai).

Give it an image and a prompt. Walk forward or turn around — picture and sound update together. The spatial field turns with the camera. Audio is not a soundtrack laid on afterwards.

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  <a href="https://huggingface.co/NoizAI/HelixWorld-preview"><img src="https://img.shields.io/badge/Hugging%20Face-ffcc4d.svg?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://helixworld.org/"><img src="https://img.shields.io/badge/Demo-helixworld.org-5b8def.svg?style=flat-square" alt="Live demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>
</p>

## News

- **2026-09-03.** We release **HelixWorld Preview v1** inference code and a preview checkpoint on [Hugging Face](https://huggingface.co/NoizAI/HelixWorld-preview). You can also roam an interactive world in the browser at [helixworld.org](https://helixworld.org/).

## Highlights

| | HelixWorld |
| --- | --- |
| Roaming / camera navigation | Yes |
| Joint audio-video | Yes |
| Spatial sound field | Follows viewpoint |
| Interactive demo | [helixworld.org](https://helixworld.org/) |
| Offline inference | Preview v1 |
| Full model, training, and report | Coming soon |

## Code

### Preview v1

- Linux, Python 3.11, CUDA 12.x
- NVIDIA GPU, 80 GB VRAM recommended
- System `ffmpeg` and `ffprobe`

A reviewed BF16 single-GPU run uses about 70 GB of VRAM.

```bash
git clone https://github.com/NoizAI/HelixWorld.git
cd HelixWorld

conda create -n helixworld-preview python=3.11 -y
conda activate helixworld-preview
python -m pip install -r requirements.txt
python download_models.py
```

```text
models/
├── text_encoder/gemma-3-12b/
└── weights/
    └── model.safetensors
```

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh \
  --image /path/to/first_frame.png \
  --prompt-file examples/prompt.json \
  --actions "W:5,right:5,stop:5" \
  --perspective first_person \
  --num-frames 121 \
  --output-dir outputs/demo
```

Edit [`examples/prompt.json`](examples/prompt.json), or pass `--video-prompt`, `--audio-prompt`, and `--av-prompt`. Output is a clean MP4 in `<output-dir>/release/native/`.

Actions: `W` `A` `S` `D` `left` `right` `up` `down` `stop`. Combine with `+` (`W+D:8`). The number after `:` is latent transitions; 121 frames have 15. The last segment may omit a duration.

## Citation

```bibtex
@misc{helixworld2026,
  title        = {HelixWorld: A Real-Time Interactive Audio-Visual World Model},
  author       = {{Noiz AI}},
  year         = {2026},
  howpublished = {\url{https://github.com/NoizAI/HelixWorld}},
  note         = {Preview v1 released; technical report, full model, and training code forthcoming}
}
```

## License

Code is [Apache 2.0](LICENSE). The weight license is published with the [Preview v1 checkpoint](https://huggingface.co/NoizAI/HelixWorld-preview).
