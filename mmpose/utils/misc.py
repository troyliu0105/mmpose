import mmcv
import os
from mmcv.utils import print_log


def update_data_root(cfg, logger=None):
    """Update data root according to env MMDET_DATASETS.

    If set env MMDET_DATASETS, update cfg.data_root according to
    MMDET_DATASETS. Otherwise, using cfg.data_root as default.

    Args:
        cfg (mmcv.Config): The model config need to modify
        logger (logging.Logger | str | None): the way to print msg
    """
    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    if 'MMDET_DATASETS' in os.environ:
        dst_root = os.environ['MMDET_DATASETS']
        print_log(f'MMDET_DATASETS has been set to be {dst_root}.'
                  f'Using {dst_root} as data root.')
    else:
        return

    assert isinstance(cfg, mmcv.Config), \
        f'cfg got wrong type: {type(cfg)}, expected mmcv.Config'

    def update(cfg, src_str, dst_str):
        if isinstance(cfg, list):
            for it in cfg:
                update(it, src_str, dst_str)
        elif isinstance(cfg, (dict, mmcv.ConfigDict)):
            for k, v in cfg.items():
                if isinstance(v, (mmcv.ConfigDict, list)):
                    update(cfg[k], src_str, dst_str)
                if isinstance(v, str) and src_str in v:
                    cfg[k] = v.replace(src_str, dst_str)

    update(cfg.data, cfg.data_root, dst_root)
    cfg.data_root = dst_root
