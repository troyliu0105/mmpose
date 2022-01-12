# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class LoadImageFromFile:
    """Loading image(s) from file.

    Required key: "image_file".

    Added key: "img".

    Args:
        to_float32 (bool): Whether to convert the loaded image to a float32
            numpy array. If set to False, the loaded image is an uint8 array.
            Defaults to False.
        color_type (str): Flags specifying the color type of a loaded image,
          candidates are 'color', 'grayscale' and 'unchanged'.
        channel_order (str): Order of channel, candidates are 'bgr' and 'rgb'.
        file_client_args (dict): Arguments to instantiate a FileClient.
            See :class:`mmcv.fileio.FileClient` for details.
            Defaults to ``dict(backend='disk')``.
    """

    def __init__(self,
                 to_float32=False,
                 color_type='color',
                 channel_order='rgb',
                 file_client_args=dict(backend='disk')):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.channel_order = channel_order
        self.file_client_args = file_client_args.copy()
        self.file_client = None

    def __call__(self, results):
        """Loading image(s) from file."""
        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)

        image_file = results['image_file']

        if isinstance(image_file, (list, tuple)):
            imgs = []
            for image in image_file:
                img_bytes = self.file_client.get(image)
                img = mmcv.imfrombytes(
                    img_bytes,
                    flag=self.color_type,
                    channel_order=self.channel_order)
                if self.to_float32:
                    img = img.astype(np.float32)
                if img is None:
                    raise ValueError(f'Fail to read {image}')
                imgs.append(img)
            results['img'] = imgs
        else:
            img_bytes = self.file_client.get(image_file)
            img = mmcv.imfrombytes(
                img_bytes,
                flag=self.color_type,
                channel_order=self.channel_order)
            if self.to_float32:
                img = img.astype(np.float32)
            if img is None:
                raise ValueError(f'Fail to read {image_file}')
            results['img'] = img

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f'file_client_args={self.file_client_args})')
        return repr_str


@PIPELINES.register_module()
class LoadImageAsThreeChannelGrayFromFile(LoadImageFromFile):
    """Loading image from file. return as a 3 channels grayscale image

    """

    def __init__(self, to_float32=False):
        super().__init__(to_float32=to_float32,
                         color_type="grayscale",
                         channel_order="bgr")

    def __call__(self, results):
        results = super().__call__(results)
        img = results['img']
        if img.ndim == 2:
            img = np.expand_dims(img, -1)
        else:
            assert img.ndim == 3 and img.shape[-1] == 1
        img = np.repeat(img, 3, axis=-1)
        results['img'] = img
        return results