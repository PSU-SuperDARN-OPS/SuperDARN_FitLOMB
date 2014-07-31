/* files to read fitlomb hdf5 files and produce fitread-like output */
/* jon klein, jtklein@alaska.edu, 07/2014 */

#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>

// to work with fitdata and rprm..
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

// send null pointer, lombfitreadvector will alloc space..
void * LombFitReadVector(hid_t recordgroup, char *dsetname)
{
    hid_t dset, dsettype, dsetspace;
    herr_t status;
    size_t typesize;
    size_t dsetlen;
    void *vectordata;
    dset = H5Dopen(recordgroup, dsetname, H5P_DEFAULT);

    dsettype = H5Dget_type(dset);
    typesize = H5Tget_size(dsettype);
    dsetspace = H5Dget_space(dset);
    dsetlen = H5Sget_simple_extent_npoints(dsetspace);
    vectordata = malloc(dsetlen * typesize);

    status = H5Dread(dset, dsettype, H5S_ALL, H5S_ALL, H5P_DEFAULT, vectordata);

    H5Tclose(dsettype);
    H5Sclose(dsetspace);
    H5Dclose(dset);
    return vectordata;
}


int LombFitRead(struct LombFile *lombfile, struct RadarParm *rprm, struct FitData *fit)
{
    char *groupname;
    ssize_t groupnamesize;
    hid_t recordgroup;

    // get name of next record (get name size, allocate space, grab name)
    groupnamesize = 1 + H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, lombfile->pulseidx, NULL, 0, H5P_DEFAULT);
    groupname = (char *) malloc (groupnamesize);
    groupnamesize = H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, lombfile->pulseidx, groupname, (size_t) groupnamesize, H5P_DEFAULT);
    recordgroup = H5Gopen(lombfile->file_id, groupname, H5P_DEFAULT);

    // read record information into RadarParm and FitData structures
    fit->revision.major = REV_MAJOR;
    fit->revision.major = REV_MINOR;
    rprm->revision.major = REV_MAJOR;
    rprm->revision.major = REV_MAJOR;

    // read in noise information
    fit->noise.vel = 0; // not currently produced by fitlomb 
    fit->noise.skynoise = 0; // not current produced by fitlomb
    LombFitReadAttr(lombfile, groupname, "noise.lag0", &fit->noise.lag0);
    /*
    // populate rprm origin struct
    rprm->origin.code = // char
    rprm->origin.time = // char *
    rprm->origin.command = // char *
    
    rprm->cp = 
    rprm->stid = 

    // populate rprm time struct (int16)
    rprm->time.yr =
    rprm->time.mo =
    rprm->time.dy =
    rprm->time.hr =
    rprm->time.mt =
    rprm->time.sc =
    rprm->time.us =
    
    rprm->txpow
    rprm->nave
    rprm->atten
    rprm->lagfr
    rprm->smsep
    rprm->ercod

    // populate stat struct (int16)
    rprm->stat.agc = 
    rprm->stat.lopwr = 

    // populate noise struct (float)
    rprm->noise.search =
    rprm->noise.mean = 

    rprm->channel
    rprm->bmnum
    rprm->bmazm
    rprm->scan
    rprm->rxrise

    // populate intt structure (???)
    rprm->intt.sc
    rprm->intt.us

    rprm->txpl
    rprm->mpinc
    rprm->mplgs
    rprm->mplgexs
    rprm->nrang
    rprm->frang
    rprm->rsep
    rprm->xcf
    rprm->tfreq
    rprm->offset
    rprm->offset
    rprm->ifmode

    rprm->maxpwr
    rprm->lvmax

    // copy pulse and lag vectors?
    rprm->pulse
    rprm->lag
    rprm->combf

    
    */
    // populate fit->rng vectors    
    double *v, *v_err, *p_0, *p_l, *p_l_err, *w_l, *w_l_err, *w_s, *w_s_err, *phi0, *phi0_err, *sdev_l, *sdev_s, *sdev_phi;
    int32_t *qflg, *gsct;
    int8_t *nump;
    uint16_t nrang = 0;
    uint16_t i;
    LombFitReadAttr(lombfile, groupname, "nrang", &nrang);

    v = LombFitReadVector(recordgroup, "v");
    v_err = LombFitReadVector(recordgroup, "v_e");
    p_0 = LombFitReadVector(recordgroup, "pwr0");
    p_l = LombFitReadVector(recordgroup, "p_l");
    p_l_err = LombFitReadVector(recordgroup, "p_l_e");
    w_l = LombFitReadVector(recordgroup, "w_l");
    w_l_err = LombFitReadVector(recordgroup, "w_l_e");
    w_s = LombFitReadVector(recordgroup, "w_s");
    w_s_err = LombFitReadVector(recordgroup, "w_s_e");
    phi0 = NULL; // not produced by fitlomb
    phi0_err = NULL; // not produced by fitlomb
    sdev_l = LombFitReadVector(recordgroup, "v_l_std");
    sdev_s = LombFitReadVector(recordgroup, "v_s_std");
    sdev_phi = NULL; // not produced by fitlomb

    qflg = LombFitReadVector(recordgroup, "qflg");
    gsct =  NULL; // not produced by fitlomb
    nump = NULL; // not produced by fitlomb
    
    for(i = 0; i < nrang; i++) {
        // doubles
        fit->rng[i].v = v[i];
        fit->rng[i].v_err = v_err[i];
        fit->rng[i].p_0 = p_0[i];
        fit->rng[i].p_l = p_l[i];
        fit->rng[i].p_l_err = p_l_err[i];
        fit->rng[i].w_l = w_l[i];
        fit->rng[i].w_l_err = w_l_err[i];
        fit->rng[i].w_s = w_s[i];
        fit->rng[i].w_s_err = w_s_err[i];
        fit->rng[i].sdev_l = sdev_l[i];
        fit->rng[i].sdev_s = sdev_s[i];

        // int32
        fit->rng[i].qflg = qflg[i];

        // set unsupported parameters to -1
        fit->rng[i].phi0 = -1;
        fit->rng[i].phi0_err = -1; 
        fit->rng[i].sdev_phi = -1;
        fit->rng[i].gsct = -1;
        fit->rng[i].nump = -1;

    }
    
    // ignore elevation for now..
    fit->xrng = NULL;
    fit->elv = NULL;

    free(v);
    free(v_err);
    free(p_0);
    free(p_l);
    free(p_l_err);
    free(w_l);
    free(w_l_err);
    free(w_s);
    free(w_s_err);
    free(sdev_l);
    free(sdev_s);
    free(qflg);

    lombfile->pulseidx++;
    lombfile->status = H5Gclose(recordgroup);
    return 0;
}
/*
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
