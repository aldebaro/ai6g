from typing import Optional

import numpy as np


class Slices:
    """
    Slices class to implement associations with basestations and UEs

    ...

    Attributes
    ----------
    max_number_slices: int
        Maximum number of supported slices
    max_number_ues: int
        Maximum number of UEs in the simulation
    ue_assoc: np.ndarray
        Numpy array associating UEs to basestations with a form SxU,
        where S and U represents the maximum number of slices and UEs
    requirements: dict
        Dictionary contaning the slice requirements defined for each slice

    Methods
    -------
    update_assoc(self, slice_assoc: Optional[np.ndarray] = None,
    ue_assoc: Optional[np.ndarray] = None)
        Update association of basestations with slices and UEs
    def update_slice_req(self, requirements: dict)
        Update slice requirements
    get_number_ue_per_slice(self)
        Return a numpy array with the number of UEs per slice
    """

    def __init__(
        self,
        max_number_slices: int,
        max_number_ues: int,
        ue_assoc: np.ndarray,
        requirements: Optional[dict] = None,
    ) -> None:
        """
        Parameters
        ----------
        max_number_slices : int
            Maximum number of slices in the simulation
        max_number_ues : int
            Maximum number of UEs in the simulation
        ue_assoc: np.ndarray
            Numpy array associating UEs to basestations with a form SxU,
            where S and U represents the maximum number of slices and UEs
        requirements: dict
            Dictionary contaning the slice requirements defined for each slice
        """
        self.max_number_slices = max_number_slices
        self.max_number_ues = max_number_ues
        self.ue_assoc = ue_assoc  # Matrix of |Slices|x|UEs|
        self.requirements = requirements

    def update_assoc(
        self,
        ue_assoc: Optional[np.ndarray] = None,
    ) -> None:
        """Update associations of slices with UEs.

        Parameters
        ----------
        ue_assoc: Optional[np.ndarray]
            Optional UE association to update the existent one
        """
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def update_slice_req(self, requirements: dict) -> None:
        """Update slices requirements.

        Parameters
        ----------
        requirements: dict
            Requirements to update the existent ones
        """
        self.requirements = requirements

    def get_number_ue_per_slice(self) -> np.ndarray:
        """Return the number of UEs per slice

        Returns
        -------
        numpy.ndarray
            An array containing the number of UEs per slice
        """
        return np.sum(self.ue_assoc, axis=1)
