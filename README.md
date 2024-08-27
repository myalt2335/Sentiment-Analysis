# Quick message
Hello! I created this as a side project and have since moved on to other things. This project will likely receive few or no updates in the future. It has always been, and will remain, open source, so I hope others can continue to build on what I’ve started here.
# Sentiment-Analysis
Sentiment + Emotion detector bot (It used to detect sarcasm but there's no real good easy way to do that)
# Base Model
In the base model folder is a basic model you can use for whatever purpose.
# Interactive Model
In the Interactive model folder is a model coded with the ability to test it built in already. You can simply download it and run it (If you have everything downloaded) then it'll prompt you to give it your message and it'll analyze it.

# Requirements

This script uses various AI models and dependencies, which require careful version management. Follow the steps below to ensure the correct versions are installed:

1. First, uninstall any conflicting versions of the required libraries:

```bash
pip uninstall -y numpy Pillow gymnasium shimmy facenet-pytorch stable-baselines3 torch torchaudio
```

2. Then, install the correct versions of the libraries:

```bash
pip install numpy<2.0 Pillow==10.2.0 gymnasium==0.29.1 stable-baselines3==2.3.2 shimmy==1.0.0 torch==2.2.0 torchaudio==2.2.0 facenet-pytorch==2.6.0
```

If you don’t already have the modules listed above, this step will install them directly.
# Processing Power
It appears to be lightweight despite the use of multiple AI models but my computer could just be strong. I don't know what specs I should recommend for using this.
# Notes
To detect sarcasm I used a single AI model I found on huggingface. Take it's detections with a grain of salt, just about any bit of text will trip it. (You can ignore this if your using the latest version where its removed) also if your reading this please star this repository please please I beg of you
