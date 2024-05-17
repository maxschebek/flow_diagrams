import numpy as np
import mdtraj as md
def radial_distribution_function(
    pos, box, num_particles, n_bins=200, n_dims=3, r_range=None, **kwargs
):
    """Compute the RDF of the data, using mdtraj"""

    top = md.Topology()
    res = top.add_residue('LJ', top.add_chain())
    for i in range(num_particles): #element type does not matter
        top.add_atom('Ar', md.element.Element.getBySymbol('Ar'), res)

    # assert jnp.abs(data.max()) <= 1 and data.min() >= 0, "data should be rescaled"
    assert pos.shape[-1] == 3 and pos.shape[-2] == num_particles

    box_mean = box.mean(axis=0)
    assert box_mean.shape == (3,)

    if r_range is None:
        r_range = (0, np.sqrt(np.sum(np.power(box_mean, 2))) / 2)
        # r_range = (0, 1)

    # box is needed for PBC. assumed to be cubic/squared
    unitcell = {
        "unitcell_lengths": box,
        "unitcell_angles": np.full((pos.shape[0], n_dims), 90),
    }
    # create mdtraj traj object
    traj = md.Trajectory(pos, top, **unitcell)
    ij = np.array(np.triu_indices(num_particles, k=1)).T
    rdf = md.compute_rdf(traj, ij, r_range=r_range, n_bins=n_bins)

    return rdf