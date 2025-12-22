import numpy as np


def draw_lines(canvases: np.ndarray, start_points: np.ndarray, end_points: np.ndarray, sample_count: int, width: float, width_pass_count: int):
  deltas = end_points - start_points
  directions = deltas / np.linalg.norm(deltas, axis=-1, keepdims=True)
  counter_directions = np.stack((directions[:, 1], -directions[:, 0]), axis=-1)
  for width_pass_offset in np.linspace(counter_directions * width / 2, -counter_directions * width / 2, width_pass_count):
    offset_start_points = start_points + width_pass_offset
    offset_end_points = end_points + width_pass_offset
    for points in np.linspace(offset_start_points, offset_end_points, sample_count):
      discretized = points.round()
      indices = discretized.astype(np.long)
      in_bounds = (indices[:, 0] >= 0) & (indices[:, 0] < canvases.shape[1]) & (indices[:, 1] >= 0) & (indices[:, 1] < canvases.shape[2])

      s = ((directions * (discretized - offset_start_points)).sum(-1) / np.square(directions).sum(-1))[:, None]
      discretized_distance_from_line = np.linalg.norm(discretized - offset_start_points - s * directions, axis=-1)
      intensity = np.clip(0.5 - discretized_distance_from_line, min=0) * 2
      
      arange = np.arange(canvases.shape[0])[in_bounds]
      indices = indices[in_bounds]
      canvases[arange, indices[:, 0], indices[:, 1]] = np.maximum(intensity[in_bounds], canvases[arange, indices[:, 0], indices[:, 1]])
