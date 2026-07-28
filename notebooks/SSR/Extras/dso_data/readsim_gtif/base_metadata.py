from __future__ import annotations

import difflib
import json
from pathlib import Path
from typing import Any
from typing import Self

import numpy as np
from pydantic import BaseModel
from pydantic import Field

class BaseMetadata(BaseModel):
    """BaseMetadata
    """
    rtol_for_np_allclose: float = Field(default=1e-14, exclude=True)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            err_msg = f'{other} not of type {type(self)}; == comparison cannot be made'
            raise TypeError(err_msg)
        temp_bool = True
        for field_name, self_field_value in self:
            other_field_value = other.__getattribute__(field_name)
            if field_name == 'rtol_for_np_allclose':  # ignore this field when comparing for equality
                curr_bool = True
            elif isinstance(self_field_value, (float, np.ndarray)):
                rtol = min(self.rtol_for_np_allclose, other.rtol_for_np_allclose)
                curr_bool = np.allclose(self_field_value, other_field_value, rtol=rtol, atol=0, equal_nan=True)
            else:
                curr_bool = self_field_value == other_field_value
            temp_bool = temp_bool and curr_bool
        return temp_bool

    __hash__ = None

    def diagnose_comparison(self, other: object) -> bool:
        if not isinstance(other, type(self)):
            err_msg = f'{other} not of type {type(self)}; == comparison cannot be made'
            raise TypeError(err_msg)
        temp_bool = True
        differing_fields_list = []
        for field_name, self_field_value in self:
            other_field_value = other.__getattribute__(field_name)
            if field_name == 'rtol_for_np_allclose':
                curr_bool = True
            elif isinstance(self_field_value, (float, np.ndarray)):
                rtol = min(self.rtol_for_np_allclose, other.rtol_for_np_allclose)
                curr_bool = np.allclose(self_field_value, other_field_value, rtol=rtol, atol=0, equal_nan=True)
                if not curr_bool:
                    np.testing.assert_allclose(self_field_value, other_field_value, rtol=rtol, atol=0, equal_nan=True,
                                               err_msg=f'{field_name} differs')
            else:
                curr_bool = self_field_value == other_field_value
            if not curr_bool:
                print(f'{field_name} differs: {self_field_value}, {other_field_value}')
                differing_fields_list.append(field_name)
            temp_bool = temp_bool and curr_bool
        diff = difflib.ndiff(self.to_json_str().splitlines(keepends=True), other.to_json_str().splitlines(keepends=True))
        print(''.join(diff), end='')
        print(f'\ndiffering fields: {differing_fields_list}')
        return temp_bool

    def to_json(self,
                save_path_meta : Path | str,		   # str to json path to save the metadata
                ) -> None:
        meta_dict = self.model_dump()
        with Path(save_path_meta).open(mode='w') as f:
            json.dump(meta_dict, f, indent=4, cls=NpEncoder)

    def to_json_str(self) -> str:
        meta_dict = self.model_dump()
        meta_jsonstr = json.dumps(meta_dict, indent=4, cls=NpEncoder)
        return meta_jsonstr

    @classmethod
    def reformat_json_loaded_dict(cls,
            meta_dict : dict,  # dict output from json.loads(validated_jsonstr)
        ) -> dict:
        """Re-implement this method as necessary for child classes. See below from_validated_jsonstr.
        Returns:
            meta_dict (dict): reformatted dict that can be passed as an argument to BaseModel.model_validate().
        """
        return meta_dict

    @classmethod
    def from_validated_jsonstr(cls,
            meta_jsonstr : str  # json string loaded from validated json.
        ) -> Self:
        """
        The intended input to this function is from a validated json file, e.g.
        with Path(save_path_meta).open(mode='r') as f:
            meta_jsonstr = json.dumps(json.load(f), indent=4)
        Or from a jsonstr loaded from the output of BaseMetadata.to_json_str().
        Returns:
            class object constructed using BaseModel.model_validate().
        """
        meta_dict = json.loads(meta_jsonstr)
        return cls.model_validate(cls.reformat_json_loaded_dict(meta_dict))

    @classmethod
    def from_validated_json(cls,
            save_path_meta : Path | str,    # str to json path of saved metadata
        ) -> Self:
        with Path(save_path_meta).open(mode='r') as f:
            meta_jsonstr = json.dumps(json.load(f), indent=4)
        return cls.from_validated_jsonstr(meta_jsonstr)
