"""
multipage_tiff_helper.py
This is a helper file for i/o of our simulated chips as multipage tiff file.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import TiffFile
from tifffile import TiffPage
from tifffile import TiffWriter


def read_multipage_tiff(sim_chip_path: str | Path) -> tuple[list[str], list[np.ndarray]]:
    """reads sim chip stored in a multipage tiff file.
    Returns list of page descriptions, and list of numpy arrays of sim data.

    Args:
        sim_chip_path (str | Path): path to tiff file.

    Returns:
        description_list (List[str]): TIFFTAG_IMAGEDESCRIPTION of each page of the tiff.
                                      len of this list is the number of pages in the tiff.
        data_list (List[np.ndarray]): array data from each page of the ome tiff.
                                      len of this list is the number of pages in the tiff.
                                      Note numpy arrays need not be of same dtype or shape.
    """
    description_list = []
    data_list = []
    with TiffFile(Path(sim_chip_path)) as temp_read:
        for page in temp_read.pages:
            assert isinstance(page, TiffPage)
            description_list.append(page.description)
            data_list.append(page.asarray())
    return description_list, data_list


def write_multipage_tiff(
    output_path: str | Path,
    description_list: list[str],
    data_list: list[np.ndarray]
    ) -> None:
    """write multi-page tiff file.
    The first page is expected to be a 4-channel uint8 array, to be interpreted as RGBA.
    Args:
        output_path (str | Path): path to output tiff file. # should end with .tif
        description_list (List[str]): TIFFTAG_IMAGEDESCRIPTION of each page of the tiff.
                                       len of this list is the number of pages in the tiff.
        data_list (List[np.ndarray]):  array data from each page of the tiff.
                                       len of this list is the number of pages in the tiff.
                                       Note numpy arrays need not be of same dtype or shape.
    Returns:
        None
    Raises:
        ValueError(msg): if data written into any tiff page is not a 2D or 3D numpy array
    """
    with TiffWriter(Path(output_path)) as temp_write:
        temp_write.write(data_list[0], photometric='rgb', extrasamples=['UNASSALPHA'],
                         description=description_list[0])
        for description, data in zip(description_list[1:], data_list[1:], strict=False):
            if len(data.shape) == 3:
                num_channels = data.shape[0]
            elif len(data.shape) == 2:
                num_channels = 1
            else:
                msg = f'data array dim {data.shape}, but only 2D or 3D numpy array expected for writing to tiff page'
                raise ValueError(msg)
            temp_write.write(data, photometric='minisblack', planarconfig='SEPARATE',
                extrasamples=['UNSPECIFIED'] * (num_channels - 1),
                             description=description)


"""
The sim chip tiff has 5 pages:
page 0: for visualization only. Its description is a jsonstr of all metadata
page 1: slant img array
page 2: ground img array
page 3: slant bdd mask
page 4: ground bdd mask
"""
