"""
sim_chip.py
Defines the SimChip class.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from typing import Self

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from pydantic import ConfigDict
from pydantic import model_validator
import json

from multipage_tiff_helper import read_multipage_tiff
from base_metadata import BaseMetadata

__all__ = ['SimChip']

# TODO: cater additional visualization pages (single-pol VV, scatterer center points)
# TODO: field for array/list of scatterer center points/values
# TODO: StringEnum for channel names

# TODO: change everything to np.complex128 when in memory, and np.complex64 only when writing out to tiff file.


class SimChip(BaseMetadata):
    """standardized class for a simulated SAR image chip and its corresponding metadata
    Args:
    slant_img : simulated slant image chip np.complex64 array.
                shape (num_channels, sim_metadata.slant_pfa_raster_info.num_rows, sim_metadata.slant_pfa_raster_info.num_cols)
                Each channel is one polarimetric channel, typically ["HH","HV","VH","VV"]
    ground_img : simulated ground image chip np.complex64 array
                 shape (num_channels, sim_metadata.ground_grid_params.num_rows, sim_metadata.ground_grid_params.num_cols)
                 Each channel is one polarimetric channel, typically ["HH","HV","VH","VV"]
    slant_bdd_mask : mask generated using cad .bdd file. np.uint8 array.
                     shape is (slant_img.shape[1], slant_img.shape[2])
    ground_bdd_mask : mask generated using cad .bdd file. np.uint8 array.
                      shape is (ground_img.shape[1], ground_img.shape[2])
                    entries in bdd masks take the following values:
                    ground_bdd_mask[np.logical_and(~layover_mask, ~shadow_mask)] = 96  # 64 + 32
                    ground_bdd_mask[np.logical_and(~layover_mask, shadow_mask)] = 64  # 64
                    ground_bdd_mask[np.logical_and(layover_mask, ~shadow_mask)] = 224  # 128 + 64 + 32
                    ground_bdd_mask[np.logical_and(layover_mask, shadow_mask)] = 192  # 128 + 64
    """
    sim_metadata : str
    slant_img : NDArray[np.complex128]  # (C, H, W)
    ground_img : NDArray[np.complex128]  # (C, H, W)
    slant_bdd_mask : NDArray[np.uint8]  # (H, W)
    ground_bdd_mask : NDArray[np.uint8]  # (H, W)
    model_config = ConfigDict(arbitrary_types_allowed=True)
    rtol_for_np_allclose : float = 1e-7  # because single precision float used in sim tiffs


    @model_validator(mode='after')
    def check_shape(self) -> Self:
        if self.slant_img.shape[1:3] != self.slant_bdd_mask.shape:
            msg = f'num rows/cols mismatch: slant_img.shape = {self.slant_img.shape}, but slant_bdd_mask.shape = {self.slant_bdd_mask.shape}'
            raise ValueError(msg)
        if self.ground_img.shape[1:3] != self.ground_bdd_mask.shape:
            msg = f'num rows/cols mismatch: ground_img.shape = {self.ground_img.shape}, but ground_bdd_mask.shape = {self.ground_bdd_mask.shape}'
            raise ValueError(msg)
        return self

    @model_validator(mode='after')
    def check_fftshift(self) -> Self:
        dc_slant = np.abs(np.sum(self.slant_img, axis=(-2, -1)))
        checkerboard_flag_slant = np.zeros(shape=self.slant_img.shape[1:], dtype=np.bool_)
        checkerboard_flag_slant[0::2, 1::2] = True
        checkerboard_flag_slant[1::2, 0::2] = True
        high_freq_slant = np.abs(np.sum(self.slant_img[:, checkerboard_flag_slant], axis=-1))
        np.testing.assert_array_less(high_freq_slant, dc_slant,
            err_msg='at least one channel of slant_img has dc weaker than high-freq; suspected fftshift issue (phase history not in corners)')
        dc_ground = np.abs(np.sum(self.ground_img, axis=(-2, -1)))
        checkerboard_flag_ground = np.zeros(shape=self.ground_img.shape[1:], dtype=np.bool_)
        checkerboard_flag_ground[0::2, 1::2] = True
        checkerboard_flag_ground[1::2, 0::2] = True
        high_freq_ground = np.abs(np.sum(self.ground_img[:, checkerboard_flag_ground], axis=-1))
        np.testing.assert_array_less(high_freq_ground, dc_ground,
            err_msg='at least one channel of ground_img has dc weaker than high-freq; suspected fftshift issue (phase history not in corners)')
        return self

    @staticmethod
    def image_description_tiff_tags() -> list[str]:
        """
        The sim chip tiff has 5 pages:
        page 0: for visualization only
        page 1: slant img array
        page 2: ground img array
        page 3: slant bdd mask
        page 4: ground bdd mask
        Thus we set the TIFFTAG_IMAGEDESCRIPTION tag for each page as the following.
        For page 0, jsonstr dump of sim_metadata
        For pages 1 to 4, simply the strings as below.
        Returns:
            tags1to4
        """
        tags1to4 = ['slant_img', 'ground_img', 'slant_bdd_mask', 'ground_bdd_mask']
        return tags1to4

    @staticmethod
    def from_sim_tiff(sim_chip_path: str | Path) -> SimChip:
        description_list, data_list = read_multipage_tiff(sim_chip_path)
        sim_metadata = description_list[0]
        slant_img = data_list[1].astype(np.complex128)
        if len(slant_img.shape) == 2:
            slant_img = slant_img[np.newaxis, :]
        ground_img = data_list[2].astype(np.complex128)
        if len(ground_img.shape) == 2:
            ground_img = ground_img[np.newaxis, :]

        return SimChip(
            sim_metadata=sim_metadata,
            slant_img=slant_img,
            ground_img=ground_img,
            slant_bdd_mask=data_list[3],
            ground_bdd_mask=data_list[4],
        )



if __name__ == '__main__':

    simchip_tifpath = ''  # type in a path to a TIF file here

    sim_chip = SimChip.from_sim_tiff(simchip_tifpath)
    print(sim_chip)

