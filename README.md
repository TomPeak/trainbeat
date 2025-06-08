# Drum Generation on my Own LLM for MidiDrumSeq per Style
Tensorflow for compliling CPU for Apple Silicon
```
uv venv --python 3.12
source .venv/bin/activate
git clone https://github.com/tensorflow/tensorflow.git
git checkout v2.16.1
cd tensorflow
bazel build //tensorflow/tools/pip_package:wheel --repo_env=USE_PYWRAP_RULES=1 --repo_env=WHEEL_NAME=tensorflow_cpu
```
## Traningdata
```
wget https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip
mkdir groove
unzip groove-v1.0.0-midionly.zip -d groove
```