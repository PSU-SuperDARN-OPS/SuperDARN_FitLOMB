/* files to read fitlomb hdf5 files and produce fitread-like output */
/* jon klein, jtklein@alaska.edu, 07/2014 */

#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>

// crap to work with fitdata and rprm..
typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

#include "fitdata.h"
#include "rprm.h"


#define FILE "20140226.ksr.a.hdf5"

#define REV_MAJOR 0
#define REV_MINOR 1

struct LombFile {
    hid_t file_id;
    hid_t root_group;
    herr_t status; /* status of last HDF5 command */
    hsize_t npulses; /* number of pulses in file */
    hsize_t pulseidx; /* index of current pulse in array */
};

int LombFitOpen(struct LombFile *lombfile, char *filename)
{
    H5G_info_t ginfo;

    lombfile->file_id = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    lombfile->root_group = H5Gopen(lombfile->file_id, "/", H5P_DEFAULT);
    lombfile->status = H5Gget_info (lombfile->root_group, &ginfo);
    lombfile->npulses = ginfo.nlinks;
    lombfile->pulseidx = 0;
    return lombfile->status;
}

int LombFitClose(struct LombFile *lombfile)
{
    lombfile->status = H5Gclose(lombfile->root_group);
    lombfile->status = H5Fclose(lombfile->file_id);
    return lombfile->status;
}

// read an attribute, used to get scalars from a record
// send attribute group, attribute name, and pointer to store attribute data
herr_t LombFitReadAttr(struct LombFile *lombfile, char *groupname, char *attrname, void *attrdata)
{
    hid_t attr, attrtype;
    herr_t status;
    
    attr = H5Aopen_by_name(lombfile->file_id, groupname, attrname, H5P_DEFAULT, H5P_DEFAULT);
    attrtype = H5Aget_type(attr);

    status = H5Aread(attr, attrtype, attrdata);
    H5Aclose(attr);
    return status;
}

int LombFitRead(struct LombFile *lombfile, struct RadarParm *prm, struct FitData *fit)
{
    char *groupname;
    ssize_t groupnamesize;
    hid_t recordgroup;

    // get name of next record (get name size, allocate space, grab name)
    groupnamesize = 1 + H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, lombfile->pulseidx, NULL, 0, H5P_DEFAULT);
    groupname = (char *) malloc (groupnamesize);
    groupnamesize = H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, lombfile->pulseidx, groupname, (size_t) groupnamesize, H5P_DEFAULT);

    printf ("Index %d: %s\n", (int) lombfile->pulseidx, groupname);

    recordgroup = H5Gopen(lombfile->file_id, groupname, H5P_DEFAULT);

        
    uint64_t t = 0;
    LombFitReadAttr(lombfile, groupname, "epoch.time", &t);
    printf("%d\n", (int) t);

    lombfile->status = H5Gclose(recordgroup);
    lombfile->pulseidx++;
    return 0;
}
    /*
    // open group
    //
    
    // read record information into RadarParm and FitData structures
    fit->revision->major = REV_MAJOR;
    fit->revision->major = REV_MINOR;
    fit->noise->vel = 
    fit->noise->skynoise = 
    fit->noise->lag0 = 
    
    // also populate rprm struct..
    
    // determine nrang
    for(int i = 0; i < nrang, i++) {
        // doubles
        fit->rng[i]->v
        fit->rng[i]->v_err
        fit->rng[i]->p_0
        fit->rng[i]->p_l
        fit->rng[i]->p_l_err
        fit->rng[i]->w_l
        fit->rng[i]->w_l_err
        fit->rng[i]->w_s
        fit->rng[i]->w_s_err
        fit->rng[i]->w_phi0
        fit->rng[i]->w_phi0_err
        fit->rng[i]->sdev_l
        fit->rng[i]->sdev_s
        fit->rng[i]->sdev_phi
        // int32
        fit->rng[i]->qflg
        fit->rng[i]->gsct
       
        // int8
        fit->rng[i]->nump
        
        // ignore elevation for now..
        fit->xrng[i] = NULL;
        fit->elv[i] = NULL;
    }
}

int LombFitSeek(struct LombFile *lombfile, int yr,int mo,int dy,int hr,int mt,int sc,double *atme)
{
    hsize_t i;
    char *name;
    ssize_t size;

    // iterate through record names looking for index of closest record time to given time
    
}

*/



int main(void)
{
    struct LombFile lombfile;
    struct RadarParm prm;
    struct FitData fit;

    LombFitOpen(&lombfile, FILE);
    LombFitRead(&lombfile, &prm, &fit);
    LombFitRead(&lombfile, &prm, &fit);
    LombFitClose(&lombfile);

    return 0;
}
