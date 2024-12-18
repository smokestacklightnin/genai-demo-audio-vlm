# Vision Language Model and Multimodal Audio-Language Examples
--------------
## General Information and Setup
This repo provides a jupyter notebook and conda environment suited to running a simple panel app demonstrator for two VLMs (Molmo and Aria) and the Qwen2-Audio audio-language model.

Before starting, you will want to clone the models from the following sources:

- Qwen2-Audio-7B-Instruct: https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct
- Molmo-7B-D-0924: https://huggingface.co/allenai/Molmo-7B-D-0924
- Aria: https://huggingface.co/rhymes-ai/Aria

The setup here assumes a NVIDIA GPU - these are not neccessarily hard and fast requirements, but the other possible combinations are not addressed here. While this notebook provides a 4-bit (bitsandbytes based) load option for Molmo-7B to save VRAM (roughly 12.5-14GB VRAM for simple model+inference), the standard-precision load for Molmo-7B and Qwen2-Audio will be most comfortable on a 24GB VRAM card. Meanwhile, while Aria only has ~4B parameters active per token at inference, the full model is quite large, so you need 50+GB VRAM just to load the thing (e.g. 4x T4, 3xL4). Depending on your hardware, the initial model load and first inference may be slow, but after that you should see reasonable speed.

Start by creating your conda environment with

`conda env create -f environment.yaml`

And await the full build - then `conda activate {env name}`

Then pull up the notebook in jupyter - you'll need to update 3 file references (all in cell 5):

`model_path` should be set to the path to the containing folder for Molmo-7B-D0924
e.g.

`model_path = "/path/to/my/folder/Molmo-7B-D-0924"`

And then there are equivalent paths that should be updated for `model_id_or_path` in each of the 'case' options: Aria, and Qwen2, in the same pattern as above.

------------

## Running and Considerations

With the paths set and the conda environment active, simply choose 'Restart Kernel and Run All' from the `Kernel` menu dropdown in jupyter and use the resulting UI.  

In the current notebook iteration, we enforce/assume a single image or audio track at a time - you can drag-and-drop files into the filedrop widget and the UI will create previews of images or an audio track to play for audio files. Files can be swapped/dropped/replaced at any time, but at the time of an actual message/inference request it will verify that the attached file type matches the model being used - images vs audio. It will also not respond if there is no image or audio file attached.

You may load a model up at any time, and swap between them at any time - simply select the model you want to use and press `Load Model` - it will fully unload any pre-existing models/data in your GPU and will load the new model.  Please don't select a different model after loading unless you want to actually use another - the inference requests use these buttons to decide which mode to be in - so if you load Qwen2-Audio and then select molmo without loading it will complain about not having image files attached, even though it's really an audio model that's loaded.

There is currently a quirky bug relating to the panel ChatInterface buttons with this setup - 'sometimes' but not always: after sending a message, the buttons will all go gray/noninteracting and will not recover, so you're unable to clear/undo/rerun etc - if you're running in a notebook those buttons can be pulled up again in a new cell via, e.g., `chat_interface._buttons['clear']` and this will work, but that doesn't solve the problem in any `.servable()` scenario. My working theory is there's a race-condition issue with tokenizers and the panel UI - tokenizers naturally sets itself up for parallelism unless it's specifically set in an environment variable to not do so - if this becomes a recurring issue, we may want to set that value, but it will come at the cost of processing speeds. For now, a simple sleep(0.1) before each large return seems to mostly skirt the issue, though it's less assured.