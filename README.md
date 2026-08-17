<p align="center">
  <img src="assets/banner.jpg" alt="HelixWorld 1.0 — Sound and vision, born together" width="100%">
</p>

# HelixWorld

**HelixWorld 1.0** is a real-time interactive audio-visual world model from [Noiz AI](https://noiz.ai).

Give it an image and a prompt. Walk forward or turn around — picture and sound update together. The spatial field turns with the camera. Audio is not a soundtrack laid on afterwards.

> Weights, inference code, and the technical report ship in the coming weeks.

<p align="center">
  <img src="https://img.shields.io/badge/arXiv-Coming%20Soon-b31b1b.svg?style=flat-square&logo=arxiv&logoColor=white" alt="arXiv">
  <img src="https://img.shields.io/badge/Hugging%20Face-Coming%20Soon-ffcc4d.svg?style=flat-square&logo=huggingface&logoColor=black" alt="Hugging Face">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg?style=flat-square" alt="License"></a>
</p>

## Highlights

| | HelixWorld 1.0 |
| --- | --- |
| Roaming / camera navigation | Yes |
| Real-time interaction | Yes |
| Joint audio-video | Yes |
| Spatial sound field | Follows viewpoint |
| Weights and code | Coming soon |

## Code

Coming Soon.

## Method

Four stages. Details, figures, and ablations ship with the report.

```text
spatial AV data  →  joint generation + action
                 →  causal rollout
                 →  distillation for realtime
```

1. **Spatial AV data.** First-person real-world video (on-location sound) plus game-engine captures with known geometry and listener pose.
2. **Joint generation + action.** The model predicts the next picture and sound from the current state and the user's action.
3. **Causal rollout.** Interaction cannot look ahead. The generator is causal, and trained to keep going from its own history.
4. **Realtime.** Joint generation is distilled so action, decode, and AV output can run as a pipeline.

## Citation

```bibtex
@misc{helixworld2026,
  title        = {HelixWorld: A Real-Time Interactive Audio-Visual World Model},
  author       = {{Noiz AI}},
  year         = {2026},
  howpublished = {\url{https://github.com/NoizAI/HelixWorld}},
  note         = {Weights and code forthcoming}
}
```

## License

Code is [Apache 2.0](LICENSE). The weight license will be published with the checkpoint.
