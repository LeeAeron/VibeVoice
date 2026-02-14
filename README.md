# 🚀 VibeVoice: A Frontier Long Conversational Text-to-Speech Engine, Portable Version
![VibeVoice](img/VibeVoice.png)

[![Release](https://img.shields.io/github/release/LeeAeron/F5-TTSx.svg)](https://github.com/LeeAeron/VibeVoice/releases/latest)


## 🔧 About
**VibeVoice** is a novel framework designed for generating **expressive**, **long-form**, **multi-speaker** conversational audio, such as podcasts, from text. It addresses significant challenges in traditional Text-to-Speech (TTS) systems, particularly in scalability, speaker consistency, and natural turn-taking.

A core innovation of VibeVoice is its use of continuous speech tokenizers (Acoustic and Semantic) operating at an ultra-low frame rate of 7.5 Hz. These tokenizers efficiently preserve audio fidelity while significantly boosting computational efficiency for processing long sequences. VibeVoice employs a [next-token diffusion](https://arxiv.org/abs/2412.08635) framework, leveraging a Large Language Model (LLM) to understand textual context and dialogue flow, and a diffusion head to generate high-fidelity acoustic details.

The model can synthesize speech up to **90 minutes** long with up to **4 distinct speakers**, surpassing the typical 1-2 speaker limits of many prior models. 


## ✅ Models
| Model | Context Length | Generation Length |  Weight |
|-------|----------------|----------|----------|
| VibeVoice-1.5B | 64K | ~90 min | [HF link](https://huggingface.co/microsoft/VibeVoice-1.5B) |
| VibeVoice-Large| 32K | ~45 min | [HF link](https://huggingface.co/microsoft/VibeVoice-Large) |
| VibeVoice-Q8| 32K | ~45 min | [HF link](https://huggingface.co/FabioSarracino/VibeVoice-Large-Q8) |


## ⚙️ Installation
VibeVoice uses Python 3.11 and Torch 2.5.1 Cuda 12.1 or 2.7.1 Cuda 12.8.

VibeVoice supports GTX and RTX cards, including GTX10xx/16xx and RTX 20xx–50xx.

### 🖥️ Windows Portable Installation

This project provided with only *.bat installer/re-installer/starter/updater file, that will download and install all components and build fully portable VibeVoice.

➤ Please Note:
    - I'm supporting only nVidia GTX10xx/16xx and RTX20xx-50xx GPUs.
    - This installer is intended for those running Windows 10 or higher. 
    - Application functionality for systems running Windows 7 or lower is not guaranteed.

- Download the VibeVoice .bat installer for Windows in [Releases](https://github.com/LeeAeron/VibeVoice/releases).
- Place the BAT-file in any folder in the root of any partition with a short Latin name without spaces or special characters and run it.
- Select INSTALL (3) entry. Choose desired Torch version to fit your GPU model. .bat file will download, unpack and configure all needed environment.
- After installing, select START (1) or START (2) depending mode you need. .bat will launch Browser, and loads necessary files, models (at forst time).
- UPDATE (4) option updates project from this Git for new version (if it will be).


## ⚙️ New Features:
- added VibeVoice Q8 model. Works with supported GPUs (RTX20xx-50xx)
- added Ruacent stress marks setup/processing
- added additional voices download option (choosable)
- added pre-uploaded reference voice pack (838 voices)
- added 4Bit on-the-fly quantize mode to run VibeVoice Large model on PCs with LowVRAM and 32GB+ RAM
- added GPU auto-detection and acceleration auto-select (SDPA/FlashAttention2)
- added autoplay selection


## 📺 Credits

* [LeeAeron](https://github.com/LeeAeron) — additional code, modding, reworking, repository, Hugginface space, features, installer/launcher, reference audios.
* [NerualDreming](@nerual_dreming) — projecting, first installler version
* [Slait](@ruweb24) - Q8 base code support, help with additional code


## 📝 License

The **VibeVoice** code is released under MIT License. 
