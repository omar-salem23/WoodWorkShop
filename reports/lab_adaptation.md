# Workshop lab model adaptation

## Training

- Model: YOLO11n, fine-tuned from `models/best.pt` on CPU.
- Training: 12 epochs; best mixed-domain validation checkpoint was selected.
- Training set: 120 selected workshop images (30 source frames plus three
  training-only image transforms per frame) and 530 sampled original-dataset
  images. The source frames have manually checked labels for pens, complete
  handled screwdrivers, and screws. Confirmed workshop-only backgrounds have
  no boxes.
- Validation: 5 workshop images plus 169 sampled original-dataset images.
- Test: 10 workshop images (8 objects and 6 backgrounds) plus the original
  held-out test set. No checkpoint selection used either test set.

The new data was kept outside the public Git repository; only the model,
results summary, and prediction preview are published here.

## Results

| Evaluation split | Original model mAP50 / mAP50-95 | Adapted model mAP50 / mAP50-95 |
| --- | ---: | ---: |
| Workshop holdout, 640 px | 52.6% / 24.6% | 67.5% / 27.0% |
| Original 920-image test | 93.7% / 78.1% | 93.4% / 75.9% |

For the workshop test, the adapted model had class mAP50-95 of 26.2% for pens,
39.8% for screwdrivers, and 14.9% for screws. It produced no detections at
confidence 0.25 on the 6 empty workshop photos. The very small screw fixed in a
wood panel received less than 0.05 confidence, below the default display
threshold. Raising image size to 1024 did not improve its confidence enough to
make it reliable. The heldout set is very small, so these figures are an early
domain check rather than a strong estimate of future accuracy.

Inference through `predict.py` removes screwdriver boxes within 2% of an image
edge. This follows the requested rule to count only complete, handled
screwdrivers; complete tools placed very close to an edge can also be
suppressed. Set `--screwdriver-edge-margin 0` to disable the filter.

The mixed-domain validation mAP50 / mAP50-95 peaked at 96.3% / 79.3%. The new
workshop validation scored 66.6% / 33.5%; its small size also limits how much
can be inferred from that number.
