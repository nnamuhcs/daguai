"""TensorBoard writer with explicit JSONL-only mode for offline test machines.

Production training still uses TensorBoard.  Set DAGUAI_ALLOW_JSON_LOGGER=1 only
if TensorBoard cannot be installed; this does NOT change the PPO optimizer.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys


class JSONLWriter:
    def __init__(self, log_dir):
        self.folder = Path(log_dir)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream = (self.folder / 'metrics-fallback.jsonl').open('a', encoding='utf-8')
        print('TensorBoard unavailable: writing JSONL metrics (DAGUAI_ALLOW_JSON_LOGGER=1)',
              file=sys.stderr, flush=True)

    def add_scalar(self, tag, scalar_value, global_step=None, **_):
        self.stream.write(json.dumps({'kind': 'scalar', 'tag': tag,
                                      'value': float(scalar_value), 'step': global_step},
                                     allow_nan=False) + '\n')

    def add_text(self, tag, text_string, global_step=None, **_):
        self.stream.write(json.dumps({'kind': 'text', 'tag': tag,
                                      'value': str(text_string), 'step': global_step}) + '\n')

    def flush(self):
        self.stream.flush()
        os.fsync(self.stream.fileno())

    def close(self):
        self.flush()
        self.stream.close()


def make_writer(log_dir):
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(str(log_dir))
    except ImportError as error:
        if os.environ.get('DAGUAI_ALLOW_JSON_LOGGER') == '1':
            return JSONLWriter(log_dir)
        raise ImportError('TensorBoard is required for training. Install training/requirements.txt '
                          'or explicitly set DAGUAI_ALLOW_JSON_LOGGER=1 for JSONL-only logging') from error
