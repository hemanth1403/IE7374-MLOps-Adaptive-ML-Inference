# Bias and Slice Performance Report

This report evaluates model-performance disparities across workload slices.
For this project, the slices are scene-complexity buckets:
- simple: <= 2 objects
- moderate: 3-7 objects
- complex: >= 8 objects

A slice is flagged when:
- latency is more than 100% worse than that model's best slice
- throughput is more than 50% worse than that model's best slice

## Summary

### yolo_large
- Best latency: 609.2328 ms
- Best throughput: 1.6414 FPS
- Flagged slices: none

### yolo_nano
- Best latency: 75.0869 ms
- Best throughput: 13.3179 FPS
- Flagged slices: none

### yolo_small
- Best latency: 133.1992 ms
- Best throughput: 7.5076 FPS
- Flagged slices: none

## Interpretation

This bias analysis is based on workload slices rather than demographic groups.
In this project, slice-based disparity matters because different scene complexities
can cause materially different latency and throughput behavior across YOLO tiers.
These results support adaptive routing rather than relying on one fixed model.
