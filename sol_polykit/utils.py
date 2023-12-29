import numpy as np

# Decorator to lazily compute property
def lazy_property(fn):
    """
    Decorator function to lazily compute a property. If the property is already
    computed, it returns the previously computed value, otherwise it computes
    the value and caches it for future use.

    See https://towardsdatascience.com/what-is-lazy-evaluation-in-python-9efb1d3bfed0
    for original implementation and associated article.
    """
    attr_name = "_lazy_" + fn.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, fn(self))
        return getattr(self, attr_name)

    return _lazy_property

# Function to determine the number of indices to subtract
def n_to_subtract(atoms_removed_inds, atom_ind):
    """
    Given a list of indices of atoms removed and the index of a specific atom
    before removal, determine the number of indices that need to be subtracted
    to account for the removed atoms.
    """
    return int(np.argwhere(np.sort(atoms_removed_inds + [atom_ind]) == atom_ind)[0][0])
