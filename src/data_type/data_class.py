from typing import NamedTuple, Optional
import numpy as np
import torch
from sklearn.model_selection import train_test_split

######Data classes used for NMMR estimation (partially adopted from https://github.com/beamlab-hsph/Neural-Moment-Matching-Regression/blob/main/src/data/ate/data_class.py)######

class NMMRTrainDataSet_h(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: Optional[np.ndarray]

class NMMRTrainDataSet_q(NamedTuple):
    treatment: np.ndarray
    treatment_target:np.ndarray
    treatment_proxy: np.ndarray
    outcome_proxy: np.ndarray
    outcome: np.ndarray
    backdoor: Optional[np.ndarray]


class NMMRTestDataSet_h(NamedTuple):
    treatment: np.ndarray
    outcome_proxy: np.ndarray
    backdoor: np.ndarray

class NMMRTestDataSet_q(NamedTuple):
    treatment: np.ndarray
    treatment_proxy: np.ndarray
    backdoor: np.ndarray


class NMMRTrainDataSetTorch_h(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, train_data: NMMRTrainDataSet_h):
        backdoor = None
        if train_data.backdoor is not None:
            backdoor = torch.tensor(train_data.backdoor, dtype=torch.float32)
        return NMMRTrainDataSetTorch_h(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=backdoor,
                                   outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return NMMRTrainDataSetTorch_h(treatment=self.treatment.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=backdoor,
                                   outcome=self.outcome.cuda())

class NMMRTrainDataSetTorch_q(NamedTuple):
    treatment: torch.Tensor
    treatment_target:torch.Tensor
    treatment_proxy: torch.Tensor
    outcome_proxy: torch.Tensor
    outcome: torch.Tensor
    backdoor: Optional[torch.Tensor]

    @classmethod
    def from_numpy(cls, train_data: NMMRTrainDataSet_q):
        backdoor = None
        if train_data.backdoor is not None:
            backdoor = torch.tensor(train_data.backdoor, dtype=torch.float32)
        return NMMRTrainDataSetTorch_q(treatment=torch.tensor(train_data.treatment, dtype=torch.float32),
                                   treatment_target=torch.tensor(train_data.treatment_target, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(train_data.treatment_proxy, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(train_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=backdoor,
                                   outcome=torch.tensor(train_data.outcome, dtype=torch.float32))

    def to_gpu(self):
        backdoor = None
        if self.backdoor is not None:
            backdoor = self.backdoor.cuda()
        return NMMRTrainDataSetTorch_q(treatment=self.treatment.cuda(),
                                   treatment_target=self.treatment_target.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=backdoor,
                                   outcome=self.outcome.cuda())


class NMMRTestDataSetTorch_h(NamedTuple):
    treatment: torch.Tensor
    outcome_proxy: torch.Tensor
    backdoor: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: NMMRTestDataSet_h):
        return NMMRTestDataSetTorch_h(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   outcome_proxy=torch.tensor(test_data.outcome_proxy, dtype=torch.float32),
                                   backdoor=torch.tensor(test_data.backdoor, dtype=torch.float32))

    def to_gpu(self):
        return NMMRTestDataSetTorch_h(treatment=self.treatment.cuda(),
                                   outcome_proxy=self.outcome_proxy.cuda(),
                                   backdoor=self.backdoor.cuda())

class NMMRTestDataSetTorch_q(NamedTuple):
    treatment: torch.Tensor
    treatment_proxy: torch.Tensor
    backdoor: torch.Tensor

    @classmethod
    def from_numpy(cls, test_data: NMMRTestDataSet_q):
        return NMMRTestDataSetTorch_q(treatment=torch.tensor(test_data.treatment, dtype=torch.float32),
                                   treatment_proxy=torch.tensor(test_data.treatment_proxy, dtype=torch.float32),
                                   backdoor=torch.tensor(test_data.backdoor, dtype=torch.float32))

    def to_gpu(self):
        return NMMRTestDataSetTorch_q(treatment=self.treatment.cuda(),
                                   treatment_proxy=self.treatment_proxy.cuda(),
                                   backdoor=self.backdoor.cuda())

