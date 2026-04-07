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
- Best latency: 404.6514 ms
- Best throughput: 2.4713 FPS
- Flagged slices: none

### yolo_nano
- Best latency: 64.8514 ms
- Best throughput: 15.4199 FPS
- Flagged slices: none

### yolo_small
- Best latency: 121.5394 ms
- Best throughput: 8.2278 FPS
- Flagged slices: none

## Interpretation

This bias analysis is based on workload slices rather than demographic groups.
In this project, slice-based disparity matters because different scene complexities
can cause materially different latency and throughput behavior across YOLO tiers.
These results support adaptive routing rather than relying on one fixed model.
