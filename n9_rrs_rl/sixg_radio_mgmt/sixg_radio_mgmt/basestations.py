from typing import Optional

import numpy as np


class Basestations:
    """
    Basestations class to implement basestations associations with slices
    and UEs

    ...

    Attributes
    ----------
    max_number_basestations : int
        Maximum number of basestations in the simulation
    max_number_slices: int
        Maximum number of supported slices
    slice_assoc: np.ndarray
        Numpy array associating basestations to slices with a form BxS,
        where B is the maximum number of basestations and S is the
        maximum number of slices
    ue_assoc: np.ndarray
        Numpy array associating UEs to basestations with a form BxU,
        where U represents the maximum number of UEs
    bandwidths: np.ndarray
        Numpy array with with the bandwidth value for each basestation
    carrier_frequencies: np.ndarray
        Numpy array with carrier frequencies values for each basestation
    num_available_rbs: np.ndarray
        Numpy array with the number of resource blocks available in
        each basestation

    Methods
    -------
    get_assoc(self)
        Method that return basestations associations with slices and UEs
    update_assoc(self, slice_assoc: Optional[np.ndarray] = None,
    ue_assoc: Optional[np.ndarray] = None)
        Update association of basestations with slices and UEs
    get_number_slices_per_basestation(self)
        Return a numpy array with the number of slices per basestation
    """

    def __init__(
        self,
        max_number_basestations: int,
        max_number_slices: int,
        slice_assoc: np.ndarray,
        ue_assoc: np.ndarray,
        bandwidths: np.ndarray,
        carrier_frequencies: np.ndarray,
        num_available_rbs: np.ndarray,
    ) -> None:
        """
        Parameters
        ----------
        max_number_basestations : int
            Maximum number of basestations in the simulation
        max_number_slices: int
            Maximum number of slices int he simulation
        slice_assoc: np.ndarray
            Numpy array associating basestations to slices with a form BxS,
            where B is the maximum number of basestations and S is the
            maximum number of slices
        ue_assoc: np.ndarray
            Numpy array associating UEs to basestations with a form BxU,
            where represents the maximum number of UEs
        bandwidths: np.ndarray
            Numpy array with with the bandwidth value for each basestation
        carrier_frequencies: np.ndarray
            Numpy array with carrier frequencies values for each basestation
        num_available_rbs: np.ndarray
            Numpy array with the number of resource blocks available in
            each basestation

        """
        self.max_number_basestations = max_number_basestations
        self.max_number_slices = max_number_slices
        self.slice_assoc = slice_assoc
        self.ue_assoc = ue_assoc
        self.bandwidths = bandwidths
        self.carrier_frequencies = carrier_frequencies
        self.num_available_rbs = num_available_rbs

    def get_assoc(self) -> np.ndarray:
        """Return slices and UEs association with basestations

        Returns
        -------
        numpy.ndarray
            An array containing slice and UEs associations
        """
        return np.array([self.slice_assoc, self.ue_assoc])

    def update_assoc(
        self,
        slice_assoc: Optional[np.ndarray] = None,
        ue_assoc: Optional[np.ndarray] = None,
    ) -> None:
        """Update associations of basestations with slices and UEs.

        Parameters
        ----------
        slice_assoc: Optional[np.ndarray]
            Optional slice association to update the existent one
        ue_assoc: Optional[np.ndarray]
            Optional UE association to update the existent one
        """
        self.slice_assoc = slice_assoc if slice_assoc is not None else self.slice_assoc
        self.ue_assoc = ue_assoc if ue_assoc is not None else self.ue_assoc

    def get_number_slices_per_basestation(self) -> np.ndarray:
        """Return the number of slices per basestation

        Returns
        -------
        numpy.ndarray
            An array containing the number of slices per basestation
        """
        return np.sum(self.slice_assoc, axis=1)
