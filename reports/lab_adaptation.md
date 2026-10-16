# Workshop lab adaptation, version 5

## Data and training

The updated checkpoint was fine-tuned from the previously published workshop
model on CPU for 12 epochs at 640 pixels, using batch size 16, AdamW, and
`lr0=0.00015`. The training split contains 530 replay images from the original
dataset plus 192 workshop images: 48 manually checked source frames and the
three supplied brightness/contrast, horizontal-flip, and vertical-flip copies
for each frame. Flip labels were transformed with their images. Validation
contains 5 workshop frames and 169 original-dataset images. The held-out test
splits were left unchanged.

Across train, validation, and test, 63 of the 113 uploaded source frames have
labels. The remaining 50 were not auto-labeled or used for this training run.
The label package contains 192 training labels, 5 validation labels, and 10
test labels; it does not include image files.

## Results

| Evaluation split | Previously published model mAP50 / mAP50-95 | Updated model mAP50 / mAP50-95 |
| --- | ---: | ---: |
| Workshop holdout, 10 images | 67.5% / 27.0% | 87.4% / 35.0% |
| Original 920-image test | 93.4% / 75.9% | 93.0% / 75.8% |

On the workshop holdout, updated class mAP50-95 is 25.3% for pens, 49.7% for
screwdrivers, and 29.9% for screws. At the default confidence threshold of
0.25, inference found 7 of 8 labeled objects and produced no boxes on the 6
empty workshop images after the complete-screwdriver edge filter. The holdout
is small, so the improvement is encouraging but not a guarantee on new scenes.

A metrics file and annotated holdout preview are included in the repository.
The label archive pairs with the user's Drive images and their supplied
augmentation variants; the archive contains labels only, not photos.
