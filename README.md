<p align="center">
  <img src="assets/banner.jpg" alt="HelixWorld — Sound and vision, born together" width="100%">
</p>

# HelixWorld

<p align="center">
  English | <a href="README_zh.md">简体中文</a>
</p>

**HelixWorld** is a real-time interactive audio-visual world model from [Noiz AI](https://noiz.ai).

Give it an image and a prompt. Walk forward or turn around — picture and sound update together. The spatial field turns with the camera. Audio is not a soundtrack laid on afterwards.

<p align="center">
  <a href="https://arxiv.org/abs/2609.38123"><img src="https://img.shields.io/badge/arXiv-2609.38123-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv"></a>
  <a href="https://huggingface.co/NoizAI/HelixWorld-preview"><img src="https://img.shields.io/badge/Hugging%20Face-ffcc4d.svg?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face"></a>
  <a href="https://huggingface.co/datasets/NoizAI/Helixbench"><img src="https://img.shields.io/badge/Benchmark-HelixBench-ffcc4d.svg?style=flat-square&logo=huggingface&logoColor=black" alt="HelixBench"></a>
  <a href="https://helixworld.org/"><img src="https://img.shields.io/badge/Demo-helixworld.org-5b8def.svg?style=flat-square" alt="Live demo"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>
</p>

## News

- **2026-10-12.** We release **HelixBench**, our benchmark for interactive audio-visual world models, on [Hugging Face](https://huggingface.co/datasets/NoizAI/Helixbench). It evaluates whether synthesized sound fields track dynamic viewpoint motion.
- **2026-09-29.** The **HelixWorld** technical report is now on [arXiv](https://arxiv.org/abs/2609.38123).
- **2026-09-03.** We release **HelixWorld Preview v1** inference code and a preview checkpoint on [Hugging Face](https://huggingface.co/NoizAI/HelixWorld-preview). You can also roam an interactive world in the browser at [helixworld.org](https://helixworld.org/).

## Showcase

Continuous generation across 8 scenes. The keys on the left show movement; the keys on the right show camera control.

<table>
  <tr>
    <td align="center" width="50%"><img src="assets/cases/03_savannah.webp" alt="Savannah Ride" width="100%"><br><b>Savannah Ride</b></td>
    <td align="center" width="50%"><img src="assets/cases/04_crystal.webp" alt="Crystal Cave" width="100%"><br><b>Crystal Cave</b></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/cases/05_studio.webp" alt="Sketch Studio" width="100%"><br><b>Sketch Studio</b></td>
    <td align="center" width="50%"><img src="assets/cases/06_vineyard.webp" alt="Vineyard at Dusk" width="100%"><br><b>Vineyard at Dusk</b></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/cases/07_paris.webp" alt="Pencil Paris" width="100%"><br><b>Pencil Paris</b></td>
    <td align="center" width="50%"><img src="assets/cases/08_countryside.webp" alt="Country Lane" width="100%"><br><b>Country Lane</b></td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="assets/cases/01_sunlit_street.webp" alt="Sunlit Street" width="100%"><br><b>Sunlit Street</b></td>
    <td align="center" width="50%"><img src="assets/cases/02_fountain.webp" alt="Fountain Plaza" width="100%"><br><b>Fountain Plaza</b></td>
  </tr>
</table>

## Highlights

### 1. Audio-Visual Synchronization

HelixWorld is a real-time interactive audio-visual world model in which visual scenes and camera-grounded spatial stereo sound co-evolve natively under user interaction. In a drumming example with a visible pause, HelixWorld's transients subside during the pause and resume with the motion, whereas the post-hoc dubbing baseline AudioX produces a transient during the pause.

![Audio-visual synchronization during drumming](assets/figures/05_av_sync_drums.png)

### 2. Camera-Aligned Spatial Audio

Continuous 6-DoF camera poses are injected into the video branch via Projective Relative Positional Encoding (PRoPE), and cross-modal attention binds visual geometry with acoustic features, naturally transferring camera ego-motion into realistic stereo panning. On [HelixBench](https://huggingface.co/datasets/NoizAI/Helixbench), which evaluates whether synthesized sound fields track dynamic viewpoint motion, HelixWorld surpasses both EchoWM and cascaded video-to-audio (V2A) dubbing pipelines in spatial-acoustic consistency.

![Spatial-acoustic consistency on HelixBench](assets/figures/06_spatial_score.png)

### 3. Real-Time Inference on a Single GPU

On a single NVIDIA H800 GPU, HelixWorld achieves a steady-state real-time factor (RTF) of 0.77 at 768×512 and 24 FPS, including video and audio decoding. On the WBench navigation split, it outperforms most open-source silent baselines in visual quality and control responsiveness.

![Inference speed and visual quality](assets/figures/07_speed_vs_quality.png)

### 4. User Study

In a blind pairwise user study with audio, HelixWorld is preferred over all baselines by a clear margin, winning the majority of votes in every matchup.

![User-study preferences](assets/figures/08_user_study.png)

Under the same first frame and control inputs, HelixWorld retains scene structure during navigation and motion reversal:

![Visual comparison under interactive control](assets/figures/09_visual_comparison.jpg)

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
@article{ke2026helixworld,
  title={HelixWorld: A Real-time Interactive Audio-Visual World Model},
  author={Ke, Lei and Pan, Jiahao and Tian, Zeyue and Wang, Jiaming and Huang, Haoyuan and Wu, Kam Man and Fang, Pengjun and Liu, Hongyu and Qi, Chenyang and Wang, Lin and others},
  journal={arXiv preprint arXiv:2609.38123},
  year={2026}
}
```

## License

Code is [Apache 2.0](LICENSE). The weight license is published with the [Preview v1 checkpoint](https://huggingface.co/NoizAI/HelixWorld-preview).

## Community

Join our user groups to share results, ask questions, and get the latest updates.

<table align="center">
  <tr>
    <td align="center"><img src="assets/qr_wechat.jpg" alt="WeChat group QR code" width="200"><br><b>WeChat</b></td>
    <td align="center"><img src="assets/qr_discord.png" alt="Discord QR code" width="200"><br><b>Discord</b></td>
  </tr>
</table>
