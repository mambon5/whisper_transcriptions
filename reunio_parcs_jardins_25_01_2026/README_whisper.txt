## Whisper model files in custom `ggml` format

The [original Whisper PyTorch models provided by OpenAI](https://github.com/open
ai/whisper/blob/main/whisper/__init__.py#L17-L30)
are converted to custom `ggml` format in order to be able to load them in C/C++.
Conversion is performed using the [convert-pt-to-ggml.py](convert-pt-to-ggml.py)
 script.

There are three ways to obtain `ggml` models:

### 1. Use [download-ggml-model.sh](download-ggml-model.sh) to download pre-conv
erted models

Example download:

```text
$ ./download-ggml-model.sh base.en
Downloading ggml model base.en ...
models/ggml-base.en.bin          100%[==========================================
===>] 141.11M  5.41MB/s    in 22s
Done! Model 'base.en' saved in 'models/ggml-base.en.bin'
You can now use it like this:

  $ ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav
```

### 2. Manually download pre-converted models

`ggml` models are available from the following locations:

- https://huggingface.co/ggerganov/whisper.cpp/tree/main

### 3. Convert with [convert-pt-to-ggml.py](convert-pt-to-ggml.py)

Download one of the [models provided by OpenAI](https://github.com/openai/whispe
r/blob/main/whisper/__init__.py#L17-L30) and generate the `ggml` files using the
 [convert-pt-to-ggml.py](convert-pt-to-ggml.py) script.

Example conversion, assuming the original PyTorch files have been downloaded int
o `~/.cache/whisper`. Change `~/path/to/repo/whisper/` to the location for your 
copy of the Whisper source:

```bash
mkdir models/whisper-medium
python models/convert-pt-to-ggml.py ~/.cache/whisper/medium.pt ~/path/to/repo/wh
isper/ ./models/whisper-medium
mv ./models/whisper-medium/ggml-model.bin models/ggml-medium.bin
rmdir models/whisper-medium
```

## Available models

| Model               | Disk    | SHA                                        |
| ------------------- | ------- | ------------------------------------------ |
| tiny                | 75 MiB  | `bd577a113a864445d4c299885e0cb97d4ba92b5f` |
| tiny.en             | 75 MiB  | `c78c86eb1a8faa21b369bcd33207cc90d64ae9df` |
| base                | 142 MiB | `465707469ff3a37a2b9b8d8f89f2f99de7299dac` |
| base.en             | 142 MiB | `137c40403d78fd54d454da0f9bd998f78703390c` |
| small               | 466 MiB | `55356645c2b361a969dfd0ef2c5a50d530afd8d5` |
| small.en            | 466 MiB | `db8a495a91d927739e50b3fc1cc4c6b8f6c2d022` |
| small.en-tdrz       | 465 MiB | `b6c6e7e89af1a35c08e6de56b66ca6a02a2fdfa1` |
| medium              | 1.5 GiB | `fd9727b6e1217c2f614f9b698455c4ffd82463b4` |
| medium.en           | 1.5 GiB | `8c30f0e44ce9560643ebd10bbe50cd20eafd3723` |
| large-v1            | 2.9 GiB | `b1caaf735c4cc1429223d5a74f0f4d0b9b59a299` |
| large-v2            | 2.9 GiB | `0f4c8e34f21cf1a914c59d8b3ce882345ad349d6` |
| large-v2-q5_0       | 1.1 GiB | `00e39f2196344e901b3a2bd5814807a769bd1630` |
| large-v3            | 2.9 GiB | `ad82bf6a9043ceed055076d0fd39f5f186ff8062` |
| large-v3-q5_0       | 1.1 GiB | `e6e2ed78495d403bef4b7cff42ef4aaadcfea8de` |
| large-v3-turbo      | 1.5 GiB | `4af2b29d7ec73d781377bfd1758ca957a807e941` |
| large-v3-turbo-q5_0 | 547 MiB | `e050f7970618a659205450ad97eb95a18d69c9ee` |

Models are multilingual unless the model name includes `.en`. Models ending in `
-q5_0` are [quantized](../README.md#quantization). Models ending in `-tdrz` supp
ort local diarization (marking of speaker turns) using [tinydiarize](https://git
hub.com/akashmjn/tinydiarize). More information about models is available [upstr
eam (openai/whisper)](https://github.com/openai/whisper#available-models-and-lan
guages). The list above is a subset of the models supported by the [download-ggm
l-model.sh](download-ggml-model.sh) script, but many more are available at https
://huggingface.co/ggerganov/whisper.cpp/tree/main and elsewhere.

## Model files for testing purposes

The model files prefixed with `for-tests-` are empty (i.e. do not contain any we
ights) and are used by the CI for
testing purposes. They are directly included in this repository for convenience 
and the Github Actions CI uses them to
run various sanitizer tests.

## Fine-tuned models

There are community efforts for creating fine-tuned Whisper models using extra t
raining data. For example, this
[blog post](https://huggingface.co/blog/fine-tune-whisper) describes a method fo
r fine-tuning using Hugging Face (HF)
Transformer implementation of Whisper. The produced models are in slightly diffe
rent format compared to the original
OpenAI format. To read the HF models you can use the [convert-h5-to-ggml.py](con
vert-h5-to-ggml.py) script like this:

```bash
git clone https://github.com/openai/whisper
git clone https://github.com/ggml-org/whisper.cpp

# clone HF fine-tuned model (this is just an example)
git clone https://huggingface.co/openai/whisper-medium

# convert the model to ggml
python3 ./whisper.cpp/models/convert-h5-to-ggml.py ./whisper-medium/ ./whisper .
```

## Distilled models

Initial support for https://huggingface.co/distil-whisper is available.

Currently, the chunk-based transcription strategy is not implemented, so there c
an be sub-optimal quality when using the distilled models with `whisper.cpp`.

```bash
# clone OpenAI whisper and whisper.cpp
git clone https://github.com/openai/whisper
git clone https://github.com/ggml-org/whisper.cpp

# get the models
cd whisper.cpp/models
git clone https://huggingface.co/distil-whisper/distil-medium.en
git clone https://huggingface.co/distil-whisper/distil-large-v2

# convert to ggml
python3 ./convert-h5-to-ggml.py ./distil-medium.en/ ../../whisper .
mv ggml-model.bin ggml-medium.en-distil.bin

python3 ./convert-h5-to-ggml.py ./distil-large-v2/ ../../whisper .
mv ggml-model.bin ggml-large-v2-distil.bin
```






























# whisper.cpp

![whisper.cpp](https://user-images.githubusercontent.com/1991296/235238348-05d0f
6a4-da44-4900-a1de-d0707e75b763.jpeg)

[![Actions Status](https://github.com/ggml-org/whisper.cpp/workflows/CI/badge.sv
g)](https://github.com/ggml-org/whisper.cpp/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://ope
nsource.org/licenses/MIT)
[![Conan Center](https://shields.io/conan/v/whisper-cpp)](https://conan.io/cente
r/whisper-cpp)
[![npm](https://img.shields.io/npm/v/whisper.cpp.svg)](https://www.npmjs.com/pac
kage/whisper.cpp/)

Stable: [v1.7.6](https://github.com/ggml-org/whisper.cpp/releases/tag/v1.7.6) / 
[Roadmap](https://github.com/orgs/ggml-org/projects/4/)

High-performance inference of [OpenAI's Whisper](https://github.com/openai/whisp
er) automatic speech recognition (ASR) model:

- Plain C/C++ implementation without dependencies
- Apple Silicon first-class citizen - optimized via ARM NEON, Accelerate framewo
rk, Metal and [Core ML](#core-ml-support)
- AVX intrinsics support for x86 architectures
- [VSX intrinsics support for POWER architectures](#power-vsx-intrinsics)
- Mixed F16 / F32 precision
- [Integer quantization support](#quantization)
- Zero memory allocations at runtime
- [Vulkan support](#vulkan-gpu-support)
- Support for CPU-only inference
- [Efficient GPU support for NVIDIA](#nvidia-gpu-support)
- [OpenVINO Support](#openvino-support)
- [Ascend NPU Support](#ascend-npu-support)
- [Moore Threads GPU Support](#moore-threads-gpu-support)
- [C-style API](https://github.com/ggml-org/whisper.cpp/blob/master/include/whis
per.h)
- [Voice Activity Detection (VAD)](#voice-activity-detection-vad)

Supported platforms:

- [x] Mac OS (Intel and Arm)
- [x] [iOS](examples/whisper.objc)
- [x] [Android](examples/whisper.android)
- [x] [Java](bindings/java/README.md)
- [x] Linux / [FreeBSD](https://github.com/ggml-org/whisper.cpp/issues/56#issuec
omment-1350920264)
- [x] [WebAssembly](examples/whisper.wasm)
- [x] Windows ([MSVC](https://github.com/ggml-org/whisper.cpp/blob/master/.githu
b/workflows/build.yml#L117-L144) and [MinGW](https://github.com/ggml-org/whisper
.cpp/issues/168))
- [x] [Raspberry Pi](https://github.com/ggml-org/whisper.cpp/discussions/166)
- [x] [Docker](https://github.com/ggml-org/whisper.cpp/pkgs/container/whisper.cp
p)

The entire high-level implementation of the model is contained in [whisper.h](in
clude/whisper.h) and [whisper.cpp](src/whisper.cpp).
The rest of the code is part of the [`ggml`](https://github.com/ggml-org/ggml) m
achine learning library.

Having such a lightweight implementation of the model allows to easily integrate
 it in different platforms and applications.
As an example, here is a video of running the model on an iPhone 13 device - ful
ly offline, on-device: [whisper.objc](examples/whisper.objc)

https://user-images.githubusercontent.com/1991296/197385372-962a6dea-bca1-4d50-b
f96-1d8c27b98c81.mp4

You can also easily make your own offline voice assistant application: [command]
(examples/command)

https://user-images.githubusercontent.com/1991296/204038393-2f846eae-c255-4099-a
76d-5735c25c49da.mp4

On Apple Silicon, the inference runs fully on the GPU via Metal:

https://github.com/ggml-org/whisper.cpp/assets/1991296/c82e8f86-60dc-49f2-b048-d
2fdbd6b5225

## Quick start

First clone the repository:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
```

Navigate into the directory:

```
cd whisper.cpp
```

Then, download one of the Whisper [models](models/README.md) converted in [`ggml
` format](#ggml-format). For example:

```bash
sh ./models/download-ggml-model.sh base.en
```

Now build the [whisper-cli](examples/cli) example and transcribe an audio file l
ike this:

```bash
# build the project
cmake -B build
cmake --build build -j --config Release

# transcribe an audio file
./build/bin/whisper-cli -f samples/jfk.wav
```

---

For a quick demo, simply run `make base.en`.

The command downloads the `base.en` model converted to custom `ggml` format and 
runs the inference on all `.wav` samples in the folder `samples`.

For detailed usage instructions, run: `./build/bin/whisper-cli -h`

Note that the [whisper-cli](examples/cli) example currently runs only with 16-bi
t WAV files, so make sure to convert your input before running the tool.
For example, you can use `ffmpeg` like this:

```bash
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

## More audio samples

If you want some extra audio samples to play with, simply run:

```
make -j samples
```

This will download a few more audio files from Wikipedia and convert them to 16-
bit WAV format via `ffmpeg`.

You can download and run the other models as follows:

```
make -j tiny.en
make -j tiny
make -j base.en
make -j base
make -j small.en
make -j small
make -j medium.en
make -j medium
make -j large-v1
make -j large-v2
make -j large-v3
make -j large-v3-turbo
```

## Memory usage

| Model  | Disk    | Mem     |
| ------ | ------- | ------- |
| tiny   | 75 MiB  | ~273 MB |
| base   | 142 MiB | ~388 MB |
| small  | 466 MiB | ~852 MB |
| medium | 1.5 GiB | ~2.1 GB |
| large  | 2.9 GiB | ~3.9 GB |

## POWER VSX Intrinsics

`whisper.cpp` supports POWER architectures and includes code which
significantly speeds operation on Linux running on POWER9/10, making it
capable of faster-than-realtime transcription on underclocked Raptor
Talos II. Ensure you have a BLAS package installed, and replace the
standard cmake setup with:

```bash
# build with GGML_BLAS defined
cmake -B build -DGGML_BLAS=1
cmake --build build -j --config Release
./build/bin/whisper-cli [ .. etc .. ]
```

## Quantization

`whisper.cpp` supports integer quantization of the Whisper `ggml` models.
Quantized models require less memory and disk space and depending on the hardwar
e can be processed more efficiently.

Here are the steps for creating and using a quantized model:

```bash
# quantize a model with Q5_0 method
cmake -B build
cmake --build build -j --config Release
./build/bin/quantize models/ggml-base.en.bin models/ggml-base.en-q5_0.bin q5_0

# run the examples as usual, specifying the quantized model file
./build/bin/whisper-cli -m models/ggml-base.en-q5_0.bin ./samples/gb0.wav
```

## Core ML support

On Apple Silicon devices, the Encoder inference can be executed on the Apple Neu
ral Engine (ANE) via Core ML. This can result in significant
speed-up - more than x3 faster compared with CPU-only execution. Here are the in
structions for generating a Core ML model and using it with `whisper.cpp`:

- Install Python dependencies needed for the creation of the Core ML model:

  ```bash
  pip install ane_transformers
  pip install openai-whisper
  pip install coremltools
  ```

  - To ensure `coremltools` operates correctly, please confirm that [Xcode](http
s://developer.apple.com/xcode/) is installed and execute `xcode-select --install
` to install the command-line tools.
  - Python 3.11 is recommended.
  - MacOS Sonoma (version 14) or newer is recommended, as older versions of MacO
S might experience issues with transcription hallucination.
  - [OPTIONAL] It is recommended to utilize a Python version management system, 
such as [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for this ste
p:
    - To create an environment, use: `conda create -n py311-whisper python=3.11 
-y`
    - To activate the environment, use: `conda activate py311-whisper`

- Generate a Core ML model. For example, to generate a `base.en` model, use:

  ```bash
  ./models/generate-coreml-model.sh base.en
  ```

  This will generate the folder `models/ggml-base.en-encoder.mlmodelc`

- Build `whisper.cpp` with Core ML support:

  ```bash
  # using CMake
  cmake -B build -DWHISPER_COREML=1
  cmake --build build -j --config Release
  ```

- Run the examples as usual. For example:

  ```text
  $ ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

  ...

  whisper_init_state: loading Core ML model from 'models/ggml-base.en-encoder.ml
modelc'
  whisper_init_state: first run on a device may take a while ...
  whisper_init_state: Core ML model loaded

  system_info: n_threads = 4 / 10 | AVX = 0 | AVX2 = 0 | AVX512 = 0 | FMA = 0 | 
NEON = 1 | ARM_FMA = 1 | F16C = 0 | FP16_VA = 1 | WASM_SIMD = 0 | BLAS = 1 | SSE
3 = 0 | VSX = 0 | COREML = 1 |

  ...
  ```

  The first run on a device is slow, since the ANE service compiles the Core ML 
model to some device-specific format.
  Next runs are faster.

For more information about the Core ML implementation please refer to PR [#566](
https://github.com/ggml-org/whisper.cpp/pull/566).

## OpenVINO support

On platforms that support [OpenVINO](https://github.com/openvinotoolkit/openvino
), the Encoder inference can be executed
on OpenVINO-supported devices including x86 CPUs and Intel GPUs (integrated & di
screte).

This can result in significant speedup in encoder performance. Here are the inst
ructions for generating the OpenVINO model and using it with `whisper.cpp`:

- First, setup python virtual env. and install python dependencies. Python 3.10 
is recommended.

  Windows:

  ```powershell
  cd models
  python -m venv openvino_conv_env
  openvino_conv_env\Scripts\activate
  python -m pip install --upgrade pip
  pip install -r requirements-openvino.txt
  ```

  Linux and macOS:

  ```bash
  cd models
  python3 -m venv openvino_conv_env
  source openvino_conv_env/bin/activate
  python -m pip install --upgrade pip
  pip install -r requirements-openvino.txt
  ```

- Generate an OpenVINO encoder model. For example, to generate a `base.en` model
, use:

  ```
  python convert-whisper-to-openvino.py --model base.en
  ```

  This will produce ggml-base.en-encoder-openvino.xml/.bin IR model files. It's 
recommended to relocate these to the same folder as `ggml` models, as that
  is the default location that the OpenVINO extension will search at runtime.

- Build `whisper.cpp` with OpenVINO support:

  Download OpenVINO package from [release page](https://github.com/openvinotoolk
it/openvino/releases). The recommended version to use is [2024.6.0](https://gith
ub.com/openvinotoolkit/openvino/releases/tag/2024.6.0). Ready to use Binaries of
 the required libraries can be found in the [OpenVino Archives](https://storage.
openvinotoolkit.org/repositories/openvino/packages/2024.6/)

  After downloading & extracting package onto your development system, set up re
quired environment by sourcing setupvars script. For example:

  Linux:

  ```bash
  source /path/to/l_openvino_toolkit_ubuntu22_2023.0.0.10926.b4452d56304_x86_64/
setupvars.sh
  ```

  Windows (cmd):

  ```powershell
  C:\Path\To\w_openvino_toolkit_windows_2023.0.0.10926.b4452d56304_x86_64\setupv
ars.bat
  ```

  And then build the project using cmake:

  ```bash
  cmake -B build -DWHISPER_OPENVINO=1
  cmake --build build -j --config Release
  ```

- Run the examples as usual. For example:

  ```text
  $ ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav

  ...

  whisper_ctx_init_openvino_encoder: loading OpenVINO model from 'models/ggml-ba
se.en-encoder-openvino.xml'
  whisper_ctx_init_openvino_encoder: first run on a device may take a while ...
  whisper_openvino_init: path_model = models/ggml-base.en-encoder-openvino.xml, 
device = GPU, cache_dir = models/ggml-base.en-encoder-openvino-cache
  whisper_ctx_init_openvino_encoder: OpenVINO model loaded

  system_info: n_threads = 4 / 8 | AVX = 1 | AVX2 = 1 | AVX512 = 0 | FMA = 1 | N
EON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 0 | SSE3
 = 1 | VSX = 0 | COREML = 0 | OPENVINO = 1 |

  ...
  ```

  The first time run on an OpenVINO device is slow, since the OpenVINO framework
 will compile the IR (Intermediate Representation) model to a device-specific 'b
lob'. This device-specific blob will get
  cached for the next run.

For more information about the OpenVINO implementation please refer to PR [#1037
](https://github.com/ggml-org/whisper.cpp/pull/1037).

## NVIDIA GPU support

With NVIDIA cards the processing of the models is done efficiently on the GPU vi
a cuBLAS and custom CUDA kernels.
First, make sure you have installed `cuda`: https://developer.nvidia.com/cuda-do
wnloads

Now build `whisper.cpp` with CUDA support:

```
cmake -B build -DGGML_CUDA=1
cmake --build build -j --config Release
```

or for newer NVIDIA GPU's (RTX 5000 series):
```
cmake -B build -DGGML_CUDA=1 -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build -j --config Release
```

## Vulkan GPU support
Cross-vendor solution which allows you to accelerate workload on your GPU.
First, make sure your graphics card driver provides support for Vulkan API.

Now build `whisper.cpp` with Vulkan support:
```
cmake -B build -DGGML_VULKAN=1
cmake --build build -j --config Release
```

## BLAS CPU support via OpenBLAS

Encoder processing can be accelerated on the CPU via OpenBLAS.
First, make sure you have installed `openblas`: https://www.openblas.net/

Now build `whisper.cpp` with OpenBLAS support:

```
cmake -B build -DGGML_BLAS=1
cmake --build build -j --config Release
```

## Ascend NPU support

Ascend NPU provides inference acceleration via [`CANN`](https://www.hiascend.com
/en/software/cann) and AI cores.

First, check if your Ascend NPU device is supported:

**Verified devices**
| Ascend NPU                    | Status  |
|:-----------------------------:|:-------:|
| Atlas 300T A2                 | Support |

Then, make sure you have installed [`CANN toolkit`](https://www.hiascend.com/en/
software/cann/community) . The lasted version of CANN is recommanded.

Now build `whisper.cpp` with CANN support:

```
cmake -B build -DGGML_CANN=1
cmake --build build -j --config Release
```

Run the inference examples as usual, for example:

```
./build/bin/whisper-cli -f samples/jfk.wav -m models/ggml-base.en.bin -t 8
```

*Notes:*

- If you have trouble with Ascend NPU device, please create a issue with **[CANN
]** prefix/tag.
- If you run successfully with your Ascend NPU device, please help update the ta
ble `Verified devices`.

## Moore Threads GPU support

With Moore Threads cards the processing of the models is done efficiently on the
 GPU via muBLAS and custom MUSA kernels.
First, make sure you have installed `MUSA SDK rc4.2.0`: https://developer.mthrea
ds.com/sdk/download/musa?equipment=&os=&driverVersion=&version=4.2.0

Now build `whisper.cpp` with MUSA support:

```
cmake -B build -DGGML_MUSA=1
cmake --build build -j --config Release
```

or specify the architecture for your Moore Threads GPU. For example, if you have
 a MTT S80 GPU, you can specify the architecture as follows:

```
cmake -B build -DGGML_MUSA=1 -DMUSA_ARCHITECTURES="21"
cmake --build build -j --config Release
```

## FFmpeg support (Linux only)

If you want to support more audio formats (such as Opus and AAC), you can turn o
n the `WHISPER_FFMPEG` build flag to enable FFmpeg integration.

First, you need to install required libraries:

```bash
# Debian/Ubuntu
sudo apt install libavcodec-dev libavformat-dev libavutil-dev

# RHEL/Fedora
sudo dnf install libavcodec-free-devel libavformat-free-devel libavutil-free-dev
el
```

Then you can build the project as follows:

```bash
cmake -B build -D WHISPER_FFMPEG=yes
cmake --build build
```

Run the following example to confirm it's working:

```bash
# Convert an audio file to Opus format
ffmpeg -i samples/jfk.wav jfk.opus

# Transcribe the audio file
./build/bin/whisper-cli --model models/ggml-base.en.bin --file jfk.opus
```

## Docker

### Prerequisites

- Docker must be installed and running on your system.
- Create a folder to store big models & intermediate files (ex. /whisper/models)

### Images

We have two Docker images available for this project:

1. `ghcr.io/ggml-org/whisper.cpp:main`: This image includes the main executable 
file as well as `curl` and `ffmpeg`. (platforms: `linux/amd64`, `linux/arm64`)
2. `ghcr.io/ggml-org/whisper.cpp:main-cuda`: Same as `main` but compiled with CU
DA support. (platforms: `linux/amd64`)
3. `ghcr.io/ggml-org/whisper.cpp:main-musa`: Same as `main` but compiled with MU
SA support. (platforms: `linux/amd64`)

### Usage

```shell
# download model and persist it in a local folder
docker run -it --rm \
  -v path/to/models:/models \
  whisper.cpp:main "./models/download-ggml-model.sh base /models"
# transcribe an audio file
docker run -it --rm \
  -v path/to/models:/models \
  -v path/to/audios:/audios \
  whisper.cpp:main "whisper-cli -m /models/ggml-base.bin -f /audios/jfk.wav"
# transcribe an audio file in samples folder
docker run -it --rm \
  -v path/to/models:/models \
  whisper.cpp:main "whisper-cli -m /models/ggml-base.bin -f ./samples/jfk.wav"
```

## Installing with Conan

You can install pre-built binaries for whisper.cpp or build it from source using
 [Conan](https://conan.io/). Use the following command:

```
conan install --requires="whisper-cpp/[*]" --build=missing
```

For detailed instructions on how to use Conan, please refer to the [Conan docume
ntation](https://docs.conan.io/2/).

## Limitations

- Inference only

## Real-time audio input example

This is a naive example of performing real-time inference on audio from your mic
rophone.
The [stream](examples/stream) tool samples the audio every half a second and run
s the transcription continuously.
More info is available in [issue #10](https://github.com/ggml-org/whisper.cpp/is
sues/10).
You will need to have [sdl2](https://wiki.libsdl.org/SDL2/Installation) installe
d for it to work properly.

```bash
cmake -B build -DWHISPER_SDL2=ON
cmake --build build -j --config Release
./build/bin/whisper-stream -m ./models/ggml-base.en.bin -t 8 --step 500 --length
 5000
```

https://user-images.githubusercontent.com/1991296/194935793-76afede7-cfa8-48d8-a
80f-28ba83be7d09.mp4

## Confidence color-coding

Adding the `--print-colors` argument will print the transcribed text using an ex
perimental color coding strategy
to highlight words with high or low confidence:

```bash
./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/gb0.wav --print-co
lors
```

<img width="965" alt="image" src="https://user-images.githubusercontent.com/1991
296/197356445-311c8643-9397-4e5e-b46e-0b4b4daa2530.png">

## Controlling the length of the generated text segments (experimental)

For example, to limit the line length to a maximum of 16 characters, simply add 
`-ml 16`:

```text
$ ./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -ml 
16

whisper_model_load: loading model from './models/ggml-base.en.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1
 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 pr
ocessors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.850]   And so my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:04.140]   Americans, ask
[00:00:04.140 --> 00:00:05.660]   not what your
[00:00:05.660 --> 00:00:06.840]   country can do
[00:00:06.840 --> 00:00:08.430]   for you, ask
[00:00:08.430 --> 00:00:09.440]   what you can do
[00:00:09.440 --> 00:00:10.020]   for your
[00:00:10.020 --> 00:00:11.000]   country.
```

## Word-level timestamp (experimental)

The `--max-len` argument can be used to obtain word-level timestamps. Simply use
 `-ml 1`:

```text
$ ./build/bin/whisper-cli -m ./models/ggml-base.en.bin -f ./samples/jfk.wav -ml 
1

whisper_model_load: loading model from './models/ggml-base.en.bin'
...
system_info: n_threads = 4 / 10 | AVX2 = 0 | AVX512 = 0 | NEON = 1 | FP16_VA = 1
 | WASM_SIMD = 0 | BLAS = 1 |

main: processing './samples/jfk.wav' (176000 samples, 11.0 sec), 4 threads, 1 pr
ocessors, lang = en, task = transcribe, timestamps = 1 ...

[00:00:00.000 --> 00:00:00.320]
[00:00:00.320 --> 00:00:00.370]   And
[00:00:00.370 --> 00:00:00.690]   so
[00:00:00.690 --> 00:00:00.850]   my
[00:00:00.850 --> 00:00:01.590]   fellow
[00:00:01.590 --> 00:00:02.850]   Americans
[00:00:02.850 --> 00:00:03.300]  ,
[00:00:03.300 --> 00:00:04.140]   ask
[00:00:04.140 --> 00:00:04.990]   not
[00:00:04.990 --> 00:00:05.410]   what
[00:00:05.410 --> 00:00:05.660]   your
[00:00:05.660 --> 00:00:06.260]   country
[00:00:06.260 --> 00:00:06.600]   can
[00:00:06.600 --> 00:00:06.840]   do
[00:00:06.840 --> 00:00:07.010]   for
[00:00:07.010 --> 00:00:08.170]   you
[00:00:08.170 --> 00:00:08.190]  ,
[00:00:08.190 --> 00:00:08.430]   ask
[00:00:08.430 --> 00:00:08.910]   what
[00:00:08.910 --> 00:00:09.040]   you
[00:00:09.040 --> 00:00:09.320]   can
[00:00:09.320 --> 00:00:09.440]   do
[00:00:09.440 --> 00:00:09.760]   for
[00:00:09.760 --> 00:00:10.020]   your
[00:00:10.020 --> 00:00:10.510]   country
[00:00:10.510 --> 00:00:11.000]  .
```

## Speaker segmentation via tinydiarize (experimental)

More information about this approach is available here: https://github.com/ggml-
org/whisper.cpp/pull/1058

Sample usage:

```py
# download a tinydiarize compatible model
./models/download-ggml-model.sh small.en-tdrz

