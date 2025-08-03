# Experimental code for Action-Agnostic Point-Level Supervision for Temporal Action Detection

## Prerequisites

* Python 3.10
* Poetry 1.7.1
* NVIDIA CUDA 11.8

To install Python dependencies, run `poetry install`.

## Configuration

Parameters can be given as a nested dictionary in the YAML format. See an example config
file in `configs/` for details.

## Dataset preparation

The dataset directory must be structured as follows:

```text
dataset_root/
    ├── gt.json            <- ActivityNet-style ground-truth file (the name is arbitrary)
    ├── aapl_labels.csv    <- CSV file containing AAPL labels
    └── features/          <- Directory containing NPY files of snippet features
        ├── <video_id_1>.npy
        ├── <video_id_2>.npy
        ├── <video_id_3>.npy
        ...
```

The path and name of `dataset_root` are arbitrary. You need to set the path to
`dataset.dataset_root` in the config. To run the code with datasets other than
THUMOS'14, you will also need to add entries to `DATASET_CFG_PRESETS` in
`src/aapl/config/presets/datasets.py`, and `AAPL_DEFAULT` and `AAPL_DATASET_DEFAULT` in
`src/aapl/config/presets/aapl.py`.

### Snippet feature files

The snippet features for one video must be saved as a single array of the shape `(T,
D)`, where `T` is the number of snippets in the video and `D` is the number of
dimensions for one snippet feature. The file name must be `<video_id>.npy`, where
`<video_id>` must be replaced with the key specifying the video in the ground-truth
file.

### AAPL label file format

AAPL labels are supposed to be given as a CSV file with the following columns:

* `video_id` (string corresponding to the keys in the ActivityNet-style ground-truth file)
* `timestamp` (float representing the timestamp (in seconds) of the label)
* `action_label` (string corresponding to labels in the ActivityNet-style ground-truth file)

The first line must be the header (`video_id,timestamp,action_label`). Set the path of
this file relative to the dataset root to `dataset.sparse_label_file` in the config.

## Run the code

```shell
poetry run python main.py --cfg ${config_path}
```

## License

The software is distributed under the following license:

```text
Copyright © 2024 Shuhei M. Yoshida

The software, including all associated documentation and files (the "Software"), is accompanied by a
research publication by the same author(s) (the "Publication") and is provided for research purposes
only ("Research Purposes"), which include reproducing and evaluating the Publication and conducting
research within the same field or discipline as the Publication (the "Related Research"). By using
the Software, you agree to the following terms:

1. You are permitted to use, copy, reproduce, or modify the Software, or create any derivative
   software from the Software (the “Derivative Software”) for Research Purposes only.

2. Distribution of the Software or the Derivative Software is permitted only when accompanied by a
   research publication reporting the Related Research. You must include this license, the following
   copyright notice, and a reference to the Publication in all copies of the Software and the
   Derivative Software.

                          Copyright <YEAR> <COPYRIGHT HOLDER> 

3. You have no rights in the Software and the Derivative Software other than those specified herein.
   The Software and the Derivative Software must not be used, copied, reproduced, modified,
   distributed, or redistributed for any purposes other than Research Purposes, including but not
   limited to commercial or business purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO, ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT,
MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS OF THE SOFTWARE AND THE PUBLICATION BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```
