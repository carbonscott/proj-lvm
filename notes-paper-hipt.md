# Scaling Vision Transformers to Gigapixel Images via Hierarchical Self-Supervised Learning (HIPT)

## Data

- One scan is 150000 × 150000 pixels (book)
- broken down into 4096×4096 sections (section)
- further broken down into 256x256 areas (sentence)
- 16x16 as the visual window (word)

256x256 areas are extracted from non-overlapping areas in the WSI (the whole
scan at the 150000x150000 resoultion).
