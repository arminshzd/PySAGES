import jax
from cell_list import get_cell_list, get_neighbor_ids, get_neighbors_list
from verlet_list import get_neighborhood

class NeighborList():
    """
    Wrapper for the neighbor list building methods

    Attributes:
        pos (jax.Array): The positions of the particles
        box (list): The system object
        cutoff (float): The cutoff distance for the neighbor list
        skin (float): The skin thickness for the neighbor list
        buffer_size (int): The buffer size for the neighbor list
    """

    def __init__(self, pos: jax.Array, box: list, cutoff: float,
                 buffer_size: int =20):
        """
        Initialize the NeighborList object

        Args:
            pos (jax.Array): The positions of the particles
            box (list): The system object
            cutoff (float): The cutoff distance for the neighbor list
            buffer_size (int): The buffer size for the neighbor list
        """
        self.pos = pos
        self.box = box
        self.cutoff = cutoff
        self.buffer_size = buffer_size
        self.cell_list = self.get_cell_list()
    
    def get_cell_list(self) -> jax.Array:
        return get_cell_list(self.pos, self.box, self.cutoff)

    def get_neighbors_single_particle(self, atom_id: int, mask_self: bool = True) -> jax.Array:
        """
        Get the ids of the neighbors of a single particle. 
        The returned array will contain the indices of the neighbors in the original
        position array. The returned array will always have a length equal to buffer_size.
        The extra elements will be padded with -1.

        Args:
            atom_id (int): The index of the particle for which the neighbors are to be found
            mask_self (bool): If True, the particle itself will not be included in the neighbor list

        Returns:
            list: The indices of the neighbors of the particle in the original position array
        """
        cell_neighbors =  get_neighbor_ids(box_size=self.box, cutoff=self.cutoff,
                               cell_idx=self.cell_list, idx=atom_id,
                               buffer_size_cell=self.buffer_size)
        # remove the padding
        cell_neighbors = cell_neighbors[cell_neighbors != -1]
        # get exact neighbors using verlet list
        nlist = get_neighborhood(self.pos[cell_neighbors, :], self.pos[atom_id, :],
                                 self.cutoff, self.box)
        
        nlist = cell_neighbors[jax.numpy.where(nlist)[0]]
        if mask_self:
            nlist = nlist[nlist != atom_id]
        return nlist
    
    
    def get_neighbors_particle_list(self, atom_ids: jax.Array, mask_self=True) -> list[jax.Array]:
        """
        Get the ids of the neighbors of a list of particles.
        The returned list will contain arrays with the indices of the neighbors in the original
        position array. The returned arrays will always have a length equal to buffer_size.
        The extra elements will be padded with -1.

        Args:
            atom_ids (ax.Array): The indices of the particles for which the neighbors are to be found
            mask_self (bool): If True, the particle itself will not be included in the neighbor list

        Returns:
            list[ax.Array]: list of arrays of the indices of the neighbors of the particles in the original position array
        """
        def _get_verlet_nlist(nlist, atom_id):
            nlist_verlet = get_neighborhood(self.pos[nlist, :], self.pos[atom_id, :],
                                     self.cutoff, self.box)
            return nlist_verlet
        
        cell_list_nlist =  get_neighbors_list(box_size=self.box, cutoff=self.cutoff,
                               cell_idx=self.cell_list, idxs=atom_ids,
                               buffer_size_cell=self.buffer_size, mask_self=False)
        
        
        
        verlet_nlist = jax.vmap(_get_verlet_nlist, in_axes=(0, 0))(cell_list_nlist, atom_ids)
        
        verlet_nlis_npad = []
        for i, row in enumerate(verlet_nlist):
            cell_list_nlist_npad = jax.numpy.where(cell_list_nlist[i, :] != -1)
            row = row[cell_list_nlist_npad[0]]
            true_verlet_ids = jax.numpy.where(row)[0]
            true_verlet = cell_list_nlist[i, true_verlet_ids]
            if mask_self:
                true_verlet = true_verlet[true_verlet != atom_ids[i]]
            verlet_nlis_npad.append(true_verlet)
        
        return verlet_nlis_npad