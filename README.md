# Quick message
Hello! I created this as a side project and have since moved on to other things. This project will likely receive few or no updates in the future. It has always been, and will remain, open source, so I hope others can continue to build on what I’ve started here.
# Sentiment-Analysis
Sentiment/Emotion detector bot (It used to detect sarcasm but there's no real good easy way to do that)
# Base Model
No Base Model is currently available, I don't feel like shaving down my code. Sorry.
# Interactive Model
In the Interactive model folder is a model coded with the ability to test it built in already. You can simply download it and run it (If you have every requirement downloaded) then it'll prompt you to give it your message and it'll analyze it.

# Requirements

This script uses various AI models and dependencies, which require careful version management. Follow the steps below to ensure the correct versions are installed:

1. You can ensure everything is downloaded with

```bash
pip install nltk textblob transformers afinn torch emoji rich
```

If you don’t already have the modules listed above, this step will install them directly.
# 11/20/24 update
If your reading this, its possible some of the modules required by this script aren't listed above even though they should be. Maybe I personally have this error because I removed a lot of my modules but if you encounter ANY errors at all, its most likely a module related error.
# Processing Power
It appears to be lightweight despite the use of multiple AI models but my computer could just be strong. I don't know what specs I should recommend for using this.
# Notes
To detect sarcasm I used a single AI model I found on huggingface. Take it's detections with a grain of salt, just about any bit of text will trip it. (You can ignore this if your using the latest version where its removed) also if your reading this please star this repository please please I beg of you
