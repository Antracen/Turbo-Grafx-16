/** A mixed-precision implicit Particle-in-Cell simulator for heterogeneous systems **/

// Allocator for 2D, 3D and 4D array: chain of pointers
#include "Alloc.h"

// Precision: fix precision for different quantities
#include "PrecisionTypes.h"
// Simulation Parameter - structure
#include "Parameters.h"
// Grid structure
#include "Grid.h"
// Interpolated Quantities Structures
#include "InterpDensSpecies.h"
#include "InterpDensNet.h"

// Field structure
#include "EMfield.h" // Just E and Bn
#include "EMfield_aux.h" // Bc, Phi, Eth, D

// Particles structure
#include "Particles.h"
#include "Particles_aux.h" // Needed only if dointerpolation on GPU - avoid reduction on GPU

// Initial Condition
#include "IC.h"
// Boundary Conditions
#include "BC.h"
// timing
#include "Timing.h"
// Read and output operations
#include "RW_IO.h"

#include "MemCpy.h" // MAWASS ADDITION

int main(int argc, char **argv) {
    
    // The "param" variable will store parameters from the command line.
    parameters param;

    // Read the parameters, print them and save them to file.
    readInputFile(&param,argc,argv);
    printParameters(&param);
    saveParameters(&param);
    
    // Timing variables
    double iStart = cpuSecond();
    double iMover, iInterp, eMover = 0.0, eInterp= 0.0;
    
    // Set-up the grid information
    grid grd;
    setGrid(&param, &grd);
    
    // Allocate Fields
    EMfield field;
    field_allocate(&grd,&field);
    EMfield_aux field_aux;
    field_aux_allocate(&grd,&field_aux);
    
    
    // Allocate Interpolated Quantities
    // per species
    interpDensSpecies *ids = new interpDensSpecies[param.ns];
    for(int is=0; is < param.ns; is++)
        interp_dens_species_allocate(&grd,&ids[is],is);
    // Net densities
    interpDensNet idn;
    interp_dens_net_allocate(&grd,&idn);
    
    // Allocate Particles
    particles *part = new particles[param.ns];
    // allocation


    long species_particles[param.ns];
    for(int is=0; is < param.ns; is++){
        particle_allocate(&param,&part[is],is);
        species_particles[is] = param.np[is];
    }
    
    // Initialization
    initGEM(&param,&grd,&field,&field_aux,part,ids);

    /* MAWASS ADDITION START */
            
        // Grid size is used in multiple points of the code.
        size_t grid_size = (grd.nxn*grd.nyn*grd.nzn);
        int num_species = param.ns;

        // Grid malloc data
        grid *device_grd;             // Located on device. Stores device array pointers.

        // Field malloc data
        field_pointers *field_pointers_device;          // Located on device. Stores device array pointers.
        field_pointers field_pointers_host;             // Located on host. Stores device array pointers.

        // Particle malloc data
        device_particle *device_part;              // Located on device. Stores device array pointers.
        device_particle part_pointers_host[num_species];                 // Located on host. Stores device array pointers.

        // IDS malloc data
        ids_pointers *ids_pointers_device;
        ids_pointers ids_pointers_host[num_species];

        // Parameters malloc data
        parameters *device_param;                   // Located on device.

        // Malloc all variables on device.
        device_grd_malloc_and_initialise(&grd, &device_grd, grid_size);
        device_param_malloc_and_initialise(&param, &device_param);

        device_field_malloc(&field_pointers_device, field_pointers_host, grid_size);

        // These should utilise the batch approach
        device_part_malloc(part, &device_part, part_pointers_host, 0, num_species);
        device_ids_malloc(&ids_pointers_device, ids_pointers_host, grid_size, 0, num_species);

        device_field_transfer(&field, field_pointers_host, grid_size, cudaMemcpyHostToDevice);     

        size_t free, total;

        cudaMemGetInfo(&free,&total);
        
        device_part_transfer(part, part_pointers_host, 0, num_species, cudaMemcpyHostToDevice);
        device_ids_transfer(ids, ids_pointers_host, grid_size, 0, num_species, cudaMemcpyHostToDevice);

        // We know now we have "free" bytes of available memory.
    /* MAWASS ADDITION END */

    // **********************************************************//
    // **** Start the Simulation!  Cycle index start from 1  *** //
    // **********************************************************//
    for(int cycle = param.first_cycle_n; cycle < (param.first_cycle_n + param.ncycles); cycle++) {
        
        std::cout << std::endl;
        std::cout << "***********************" << std::endl;
        std::cout << "   cycle = " << cycle << std::endl;
        std::cout << "***********************" << std::endl;
    
        // set to zero the densities - needed for interpolation
        setZeroDensities(&idn,ids,&grd,param.ns);
        
        
        // implicit mover
        iMover = cpuSecond(); // start timer for mover


        for(int is=0; is < param.ns; is++) {
            mover_PC(&(device_part[is]), field_pointers_device, device_grd, device_param, part[is].nop);
        }

        eMover += (cpuSecond() - iMover); // stop timer for mover
        
        // interpolation particle to grid
        iInterp = cpuSecond(); // start timer for the interpolation step
        

        // interpolate species
        for(int is=0; is < param.ns; is++) {
            interpP2G(&(device_part[is]), &(ids_pointers_device[is]), device_grd, part[is].nop); // modifies [ids -> ]
        }
        // apply BC to interpolated densities

        device_ids_transfer(ids, ids_pointers_host, grid_size, 0, num_species, cudaMemcpyDeviceToHost);

        for(int is=0; is < param.ns; is++)
            applyBCids(&ids[is], &grd, &param); // modifies [ids -> ]
        // sum over species
        sumOverSpecies(&idn, ids, &grd, param.ns); // modifies [idn]
        // interpolate charge density from center to node
        applyBCscalarDensN(idn.rhon, &grd, &param);
                
        // write E, B, rho to disk
        if(cycle % param.FieldOutputCycle == 0){
            VTK_Write_Vectors(cycle, &grd,&field);
            VTK_Write_Scalars(cycle, &grd, ids, &idn);
        }
        
        eInterp += (cpuSecond() - iInterp); // stop timer for interpolation  
    
    }  // end of one PIC cycle
    
    /// Release the resources
    // deallocate field
    grid_deallocate(&grd);
    field_deallocate(&grd,&field);
    // interp
    interp_dens_net_deallocate(&grd,&idn);
    
    // Deallocate interpolated densities and particles
    for(int is=0; is < param.ns; is++){
        interp_dens_species_deallocate(&grd,&ids[is]);
        particle_deallocate(&part[is]);
    }
    
    
    // stop timer
    double iElaps = cpuSecond() - iStart;
    
    // Print timing of simulation
    std::cout << std::endl;
    std::cout << "**************************************" << std::endl;
    std::cout << "   Tot. Simulation Time (s) = " << iElaps << std::endl;
    std::cout << "   Mover Time / Cycle   (s) = " << eMover/param.ncycles << std::endl;
    std::cout << "   Interp. Time / Cycle (s) = " << eInterp/param.ncycles  << std::endl;
    std::cout << "**************************************" << std::endl;
    
    /*
        @mawass
        TODO: Deallocate things from the GPU
    */

    // exit
    return 0;
}

