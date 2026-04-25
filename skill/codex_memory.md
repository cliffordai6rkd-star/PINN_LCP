# Codex Memory - CNN Learning Project

## Project Agreement

This is a learning project for understanding how CNN layers transform images.

The student wants to write the code personally. Codex should act as a teacher, not as the main programmer.

Rules:

- Use Chinese by default.
- Do not directly write complete implementation code for the student's core CNN/data-loading logic.
- If the student asks for full code, refuse politely and provide guidance, pseudocode, debugging steps, or a checklist.
- Help step by step, using small goals.
- Review the student's code and explain errors clearly.
- Prefer conceptual explanation before implementation details.

The project root has `SKILL.md` with these coaching rules.

## Current Project Goal

The current goal is to build the data-loading foundation before writing CNN layers.

The intended dataset structure is:

```text
dataset_root/
  episode_0001/
    colors/
      000001_ee_cam_color.jpg
      000001_side_cam_color.jpg
      000001_third_person_cam_color.jpg
  episode_0002/
    colors/
      ...
```

Important correction discovered today:

- The folder is named `colors`, not `color`.

## Files In Progress

Main files:

- `dataset/dataloader.py`
- `dataset/imgloader.py`

There is also another open file:

- `dataloader/dataloader.py`

Be careful not to confuse `dataset/dataloader.py` with `dataloader/dataloader.py`.

## Completed So Far

### 1. Episode Path Collection

The student wrote a method that:

- Takes `self.path` as the dataset root.
- Uses `os.listdir(dataset_path)` to list components.
- Uses `os.path.join(dataset_path, role)` to build full paths.
- Uses `os.path.isdir(role_path)` to keep only episode folders.
- Appends each episode path into a list.
- Returns the list.

Recommended improvement already discussed:

- Use sorted listing so episodes appear in order:

```text
components = sorted(os.listdir(dataset_path))
```

Expected output:

```text
[
  ".../episode_0001",
  ".../episode_0002",
  ...
]
```

### 2. Colors Path Collection

The student wanted episode handling to happen inside the function, not passed from outside.

The intended method behavior:

```text
get_color_path or get_color_paths:
  1. Call self.get_episode_path()
  2. Loop over every episode_path
  3. Join episode_path with "colors"
  4. Check os.path.isdir(color_path)
  5. Append valid color_path to color_paths
  6. Return color_paths after the loop
```

Important variable distinction:

```text
episode_paths = list of all episode paths
episode_path = one episode path during the loop
color_paths = list of all colors paths
color_path = one colors path during the loop
```

Common mistake fixed:

- `self.get_episode_path` is the method object.
- `self.get_episode_path()` runs the method and returns the list.

Common mistake fixed:

- `return color_paths` must be outside the `for` loop, otherwise only the first episode is processed.

### 3. Image Path Dictionary

The student built a dictionary like:

```text
{
  "episode_0054": {
    "000514_ee_cam_color": ".../episode_0054/colors/000514_ee_cam_color.jpg",
    "000514_side_cam_color": ".../episode_0054/colors/000514_side_cam_color.jpg",
    "000514_third_person_cam_color": ".../episode_0054/colors/000514_third_person_cam_color.jpg"
  }
}
```

The intended method behavior:

```text
get_img_dict:
  1. Create img_dict = {}
  2. Get color_paths by calling self.get_color_path()
  3. For each color_path:
     - episode_path = parent folder of color_path
     - episode_name = basename of episode_path
     - create img_dict[episode_name] = {}
  4. List files inside color_path and sort them
  5. For each img_name:
     - build img_path with os.path.join(color_path, img_name)
     - check os.path.isfile(img_path)
     - split img_name with os.path.splitext(img_name)
     - img_key should be the filename without extension
     - img_ext should be the extension, such as ".jpg"
     - if img_ext is allowed, store:
       img_dict[episode_name][img_key] = img_path
  6. Return img_dict
```

Important correction:

- `os.path.splitext("123.jpg")` returns:

```text
("123", ".jpg")
```

So:

- `[0]` is the key.
- `[1]` is the extension.

Important correction:

- Pillow image files have extensions like `.jpg`, not `jpg`.

Suggested future improvement:

```text
Use img_ext.lower()
Allow [".jpg", ".jpeg", ".png", ".bmp"]
```

The student currently prints many lines like:

```text
color ext: .jpg
```

This is too noisy. Next time, suggest removing this normal-case print or only printing skipped files.

### 4. Reading One Image

The student moved to `dataset/imgloader.py` and began implementing image loading.

Libraries:

- Install package: `pillow`
- Import in code: `from PIL import Image`
- Install/import NumPy: `import numpy as np`

Important correction:

- Do not write `import pillow`.
- Pillow is installed as `pillow`, but imported as `PIL`.

Image loading steps:

```text
1. Get image_path from img_dict
2. Open image_path with Image.open(image_path)
3. Convert image object to RGB with image.convert("RGB")
4. Optionally resize with image.resize((64, 64))
5. Convert to NumPy array with np.array(image)
6. Normalize with array / 255.0
7. Return image_array
```

Important distinction:

```text
Image.open(...) uses uppercase Image from PIL.
image.convert(...) uses lowercase image, the opened image object.
image.resize(...) also uses lowercase image.
```

Common mistakes fixed:

- `PIL.Image.open(...)` caused `NameError` because `PIL` was not imported.
- `Image.convert("RGB")` caused `AttributeError` because `convert` belongs to the image object, not the module.
- `image.convert("RGB")` before `image = Image.open(image_path)` caused `UnboundLocalError`.

The student successfully printed a normalized NumPy image array with values like:

```text
0.29803922
0.40784314
...
```

This means image reading and normalization are working.

Next suggested verification:

Print only:

```text
image_array.shape
image_array.dtype
image_array.min()
image_array.max()
```

Expected if resized to 64x64 RGB:

```text
shape: (64, 64, 3)
min: 0.0
max: <= 1.0
```

## Python Environment Fix

There was an environment issue.

Initial symptoms:

- `python` command was missing.
- `/usr/bin/python3` was used.
- `pip install pillow` failed with:

```text
externally-managed-environment
```

Diagnosis:

- `conda info --envs` showed `cnn` active.
- But `which python3` and `which pip` pointed to:

```text
/usr/bin/python3
/usr/bin/pip
```

Further diagnosis:

- `/home/hirol/miniconda3/envs/cnn/bin/python` did not exist.
- The `cnn` conda environment existed but was nearly empty and did not contain Python.

Fix:

```text
conda install -n cnn python=3.11 pip
conda deactivate
conda activate cnn
hash -r
```

After that, packages were successfully installed:

```text
pip install pillow
pip install numpy
```

Important future instruction:

- Do not run project scripts with `/usr/bin/python3`.
- Use:

```text
python dataset/imgloader.py
```

after activating `cnn`.

Check environment with:

```text
which python
which pip
python -c "import sys; print(sys.executable)"
```

Expected:

```text
/home/hirol/miniconda3/envs/cnn/bin/python
/home/hirol/miniconda3/envs/cnn/bin/pip
/home/hirol/miniconda3/envs/cnn/bin/python
```

## Current Next Step

Next time, continue from `dataset/imgloader.py`.

Recommended next small goal:

1. Stop printing the whole image array.
2. Make the image-loading method return `image_array`.
3. Print only summary info:

```text
shape
dtype
min
max
```

4. Confirm image shape is suitable for the first CNN layer.

After that, begin the first CNN layer:

```text
convolution layer
```

Before coding convolution, ask the student:

- What is the input image shape?
- What kernel size should be used?
- How many filters should the first layer have?
- Should padding be used?
- Should stride be 1?

Keep teaching step by step and avoid writing complete implementation code.
