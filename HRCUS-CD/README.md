# HRCUS-CD Dataset

This directory is reserved for the HRCUS-CD dataset used by AERNet.

The dataset files are not included in this repository. Please obtain the dataset from the original AERNet repository:

https://github.com/zjd1836/AERNet

After downloading and extracting the dataset, place the files in the following structure:

```text
HRCUS-CD/
├── train/
│   ├── A/
│   │   ├── 00000.tif
│   │   └── ...
│   ├── B/
│   │   ├── 00000.tif
│   │   └── ...
│   └── label/
│       ├── 00000.tif
│       └── ...
├── val/
│   ├── A/
│   ├── B/
│   └── label/
└── test/
    ├── A/
    ├── B/
    └── label/
```

Each sample contains two temporal remote sensing images and one binary change label:

- `A`: image from the first time phase
- `B`: image from the second time phase
- `label`: pixel-level building change mask

The file names in `A`, `B`, and `label` should correspond one-to-one, for example:

```text
HRCUS-CD/train/A/00000.tif
HRCUS-CD/train/B/00000.tif
HRCUS-CD/train/label/00000.tif
```
