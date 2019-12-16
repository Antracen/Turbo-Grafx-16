#ifndef MEMCPY_H
#define MEMCPY_H

#include "Grid.h"
#include "EMfield.h"
#include "Particles.h"

struct field_pointers {
    FPfield* Ex_flat;
    FPfield* Ey_flat;
    FPfield* Ez_flat;
    FPfield* Bxn_flat;
    FPfield* Byn_flat;
    FPfield* Bzn_flat;
};

struct ids_pointers {
    FPinterp *rhon_flat;
    FPinterp *Jx_flat;
    FPinterp *Jy_flat;
    FPinterp *Jz_flat;
    FPinterp *pxx_flat;
    FPinterp *pxy_flat;
    FPinterp *pxz_flat;
    FPinterp *pyy_flat;
    FPinterp *pyz_flat;
    FPinterp *pzz_flat;
};

struct device_particle {
    
    int n_sub_cycles;
    FPpart qom;
    long nop;
    int NiterMover;

    FPpart *x;
    FPpart *y;
    FPpart *z;
    FPpart *u;
    FPpart *v;
    FPpart *w;
    FPpart *q;
};

void device_grd_malloc_and_initialise(grid *grd, grid **device_grd, size_t grid_size);
void device_param_malloc_and_initialise(parameters *param, parameters **device_param);
void device_field_malloc(field_pointers **device_field_pointers, field_pointers &host_field_pointers, size_t grid_size);
void device_part_malloc(particles *part, device_particle **device_part_device, device_particle *host_part_pointers, size_t start, size_t end);
void device_ids_malloc(ids_pointers **device_ids_pointers, ids_pointers *host_ids_pointers, size_t grid_size, size_t start, size_t end);

/* TRANSFER FUNCTIONS */
void device_field_transfer(EMfield *field, field_pointers &host_field_pointers, size_t grid_size, cudaMemcpyKind direction);
void device_param_transfer(parameters &param, parameters *device_param, cudaMemcpyKind direction);
void device_part_transfer(struct particles *part, struct device_particle *host_part_pointers, size_t start, size_t end, cudaMemcpyKind direction);
void device_ids_transfer(interpDensSpecies *ids, ids_pointers *host_ids_pointers, size_t grid_size, size_t start, size_t end, cudaMemcpyKind direction);

#endif