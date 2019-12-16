#include "MemCpy.h"
#include "Grid.h"
#include "EMfield.h"
#include "Particles.h"

/*
    The allocation will also transfer grid-elements which will not change over the runtime.
*/
void device_grd_malloc_and_initialise(grid *grd, grid **device_grd, size_t grid_size) {

    cudaMalloc(device_grd, sizeof(grid));

    // Backup pointers
        FPfield *XN_flat = grd->XN_flat;
        FPfield *YN_flat = grd->YN_flat;
        FPfield *ZN_flat = grd->ZN_flat;

    // Allocate memory for the dynamic arrays on the device.
    cudaMalloc(&(grd->XN_flat), grid_size*sizeof(FPfield));
    cudaMalloc(&(grd->YN_flat), grid_size*sizeof(FPfield));
    cudaMalloc(&(grd->ZN_flat), grid_size*sizeof(FPfield));
    cudaMemcpy(grd->XN_flat, XN_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd->YN_flat, YN_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice);
    cudaMemcpy(grd->ZN_flat, ZN_flat, grid_size*sizeof(FPfield), cudaMemcpyHostToDevice);

    cudaMemcpy(*device_grd, grd, sizeof(grid), cudaMemcpyHostToDevice);

    // Restore pointers
        grd->XN_flat = XN_flat;
        grd->YN_flat = YN_flat;
        grd->ZN_flat = ZN_flat;
}

void device_field_malloc(field_pointers **device_field_pointers, field_pointers &host_field_pointers, size_t grid_size) {

    cudaMalloc(device_field_pointers, sizeof(field_pointers));

    // Allocate memory for the dynamic arrays on the device.
    cudaMalloc(&(host_field_pointers.Ex_flat), grid_size*sizeof(FPfield));
    cudaMalloc(&(host_field_pointers.Ey_flat), grid_size*sizeof(FPfield));
    cudaMalloc(&(host_field_pointers.Ez_flat), grid_size*sizeof(FPfield));
    cudaMalloc(&(host_field_pointers.Bxn_flat), grid_size*sizeof(FPfield));
    cudaMalloc(&(host_field_pointers.Byn_flat), grid_size*sizeof(FPfield));
    cudaMalloc(&(host_field_pointers.Bzn_flat), grid_size*sizeof(FPfield));

    cudaMemcpy(*device_field_pointers, &host_field_pointers, sizeof(field_pointers), cudaMemcpyHostToDevice);
}

void device_field_transfer(EMfield *field, field_pointers &host_field_pointers, size_t grid_size, cudaMemcpyKind direction) {
    if(direction == cudaMemcpyHostToDevice) {
        cudaMemcpy(host_field_pointers.Ex_flat, field->Ex_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(host_field_pointers.Ey_flat, field->Ey_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(host_field_pointers.Ez_flat, field->Ez_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(host_field_pointers.Bxn_flat, field->Bxn_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(host_field_pointers.Byn_flat, field->Byn_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(host_field_pointers.Bzn_flat, field->Bzn_flat, grid_size*sizeof(FPfield), direction);
        
    } else {
        cudaMemcpy(field->Ex_flat, host_field_pointers.Ex_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(field->Ey_flat, host_field_pointers.Ey_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(field->Ez_flat, host_field_pointers.Ez_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(field->Bxn_flat, host_field_pointers.Bxn_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(field->Byn_flat, host_field_pointers.Byn_flat, grid_size*sizeof(FPfield), direction);
        cudaMemcpy(field->Bzn_flat, host_field_pointers.Bzn_flat, grid_size*sizeof(FPfield), direction);
    }
}

void device_param_malloc_and_initialise(parameters *param, parameters **device_param) {
    cudaMalloc(device_param, sizeof(parameters));
    cudaMemcpy(*device_param, param, sizeof(parameters), cudaMemcpyHostToDevice);
}

/*
    The allocation will also transfer grid-elements which will not change over the runtime.
*/
void device_part_malloc(particles *part, device_particle **device_part_pointers, device_particle *host_part_pointers, size_t start, size_t end) {

    cudaMalloc(device_part_pointers, (end-start)*sizeof(device_particle));

    // Allocate memory for the dynamic arrays on the device.
    long npmax;
    for(size_t i = start; i < end; i++) {
        npmax = part[i].npmax;
        host_part_pointers[i].n_sub_cycles = part[i].n_sub_cycles; 
        host_part_pointers[i].qom = part[i].qom;
        host_part_pointers[i].nop = part[i].nop;
        host_part_pointers[i].NiterMover = part[i].NiterMover;
        cudaMalloc(&(host_part_pointers[i].x), npmax*sizeof(FPpart));
        cudaMalloc(&(host_part_pointers[i].y), npmax*sizeof(FPpart));
        cudaMalloc(&(host_part_pointers[i].z), npmax*sizeof(FPpart));
        cudaMalloc(&(host_part_pointers[i].u), npmax*sizeof(FPpart));
        cudaMalloc(&(host_part_pointers[i].v), npmax*sizeof(FPpart));
        cudaMalloc(&(host_part_pointers[i].w), npmax*sizeof(FPpart));
        cudaMalloc(&(host_part_pointers[i].q), npmax*sizeof(FPpart));
        
    }
    cudaMemcpy(&((*device_part_pointers)[start]), &host_part_pointers[start], (end-start)*sizeof(device_particle), cudaMemcpyHostToDevice);
    
}

void device_ids_malloc(ids_pointers **device_ids_pointers, ids_pointers *host_ids_pointers, size_t grid_size, size_t start, size_t end) {

    cudaMalloc(device_ids_pointers, (end-start)*sizeof(ids_pointers));

    for(size_t i = start; i < end; i++) {
        // Allocate memory for the dynamic arrays on the device.
        cudaMalloc(&(host_ids_pointers[i].rhon_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].Jx_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].Jy_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].Jz_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].pxx_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].pxy_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].pxz_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].pyy_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].pyz_flat), grid_size*sizeof(FPinterp));
        
        cudaMalloc(&(host_ids_pointers[i].pzz_flat), grid_size*sizeof(FPinterp));
        
    }
    cudaMemcpy(&((*device_ids_pointers)[start]), &(host_ids_pointers[start]), (end-start)*sizeof(ids_pointers), cudaMemcpyHostToDevice);
    
}

void device_part_transfer(struct particles *part, struct device_particle *host_part_pointers, size_t start, size_t end, cudaMemcpyKind direction) {

    long npmax;
    if(direction == cudaMemcpyHostToDevice) {
        for(size_t i = start; i < end; i++) {
            npmax = part[i].npmax;
            cudaMemcpy(host_part_pointers[i].x, part[i].x, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(host_part_pointers[i].y, part[i].y, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(host_part_pointers[i].z, part[i].z, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(host_part_pointers[i].u, part[i].u, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(host_part_pointers[i].v, part[i].v, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(host_part_pointers[i].w, part[i].w, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(host_part_pointers[i].q, part[i].q, npmax*sizeof(FPpart), direction);
            
        }
    } else {
        for(size_t i = start; i < end; i++) {
            npmax = part[i].npmax;
            cudaMemcpy(part[i].x, host_part_pointers[i].x, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(part[i].y, host_part_pointers[i].y, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(part[i].z, host_part_pointers[i].z, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(part[i].u, host_part_pointers[i].u, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(part[i].v, host_part_pointers[i].v, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(part[i].w, host_part_pointers[i].w, npmax*sizeof(FPpart), direction);
            
            cudaMemcpy(part[i].q, host_part_pointers[i].q, npmax*sizeof(FPpart), direction);
            
        }
    }
}

void device_ids_transfer(interpDensSpecies *ids, ids_pointers *host_ids_pointers, size_t grid_size, size_t start, size_t end, cudaMemcpyKind direction) {

    if(direction == cudaMemcpyHostToDevice) {
        for(size_t i = start; i < end; i++) {
            cudaMemcpy(host_ids_pointers[i].rhon_flat, ids[i].rhon_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].Jx_flat, ids[i].Jx_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].Jy_flat, ids[i].Jy_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].Jz_flat, ids[i].Jz_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].pxx_flat, ids[i].pxx_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].pxy_flat, ids[i].pxy_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].pxz_flat, ids[i].pxz_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].pyy_flat, ids[i].pyy_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].pyz_flat, ids[i].pyz_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(host_ids_pointers[i].pzz_flat, ids[i].pzz_flat, grid_size*sizeof(FPinterp), direction);
            
        }
    } else {
        for(size_t i = start; i < end; i++) {
            cudaMemcpy(ids[i].rhon_flat, host_ids_pointers[i].rhon_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].Jx_flat, host_ids_pointers[i].Jx_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].Jy_flat, host_ids_pointers[i].Jy_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].Jz_flat, host_ids_pointers[i].Jz_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].pxx_flat, host_ids_pointers[i].pxx_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].pxy_flat, host_ids_pointers[i].pxy_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].pxz_flat, host_ids_pointers[i].pxz_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].pyy_flat, host_ids_pointers[i].pyy_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].pyz_flat, host_ids_pointers[i].pyz_flat, grid_size*sizeof(FPinterp), direction);
            
            cudaMemcpy(ids[i].pzz_flat, host_ids_pointers[i].pzz_flat, grid_size*sizeof(FPinterp), direction);  
              
        }
    }
}