#ifndef DEVICEFIELD_H
#define DEVICEFIELD_H

    #include "Alloc.h"
    #include "EMfield.h"

    struct DeviceField {
        FPfield* Ex_flat;
        FPfield* Ey_flat;
        FPfield* Ez_flat;

        FPfield* Bxn_flat;
        FPfield* Byn_flat;
        FPfield* Bzn_flat;
    };

#endif