/* files to read fitlomb hdf5 files and produce fitread-like output */
/* see main for example usage */
/* jon klein, jtklein@alaska.edu, 07/2014 */

#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// to work with fitdata and rprm..
typedef int8_t int8;
typedef uint8_t uint8;
typedef int16_t int16;
typedef uint16_t uint16;
typedef int32_t int32;
typedef uint32_t uint32;
typedef int64_t int64;
typedef uint64_t uint64;

#include "rprm.h"
#include "fitdata.h"
#include "fitlombread.h"

#define FILE "20140226.ksr.a.hdf5"

#define ORIGIN "fitlombread"
#define ORIGIN_CODE 0
#define COMMENT "fitlombread" 
#define REV_MAJOR 0
#define REV_MINOR 1

// prototype copied functions..
void FitFree(struct FitData *ptr);
struct FitData * FitMake();
int FitSetRng(struct FitData *ptr,int nrang);
struct RadarParm *RadarParmMake();
int RadarParmSetOriginCommand(struct RadarParm *ptr,char *str);
int RadarParmSetCombf(struct RadarParm *ptr,char *str);
void RadarParmFree(struct RadarParm *ptr);

int32_t LombFitOpen(struct LombFile *lombfile, char *filename)
{
    H5G_info_t ginfo;

    lombfile->file_id = H5Fopen(FILE, H5F_ACC_RDONLY, H5P_DEFAULT);
    lombfile->root_group = H5Gopen(lombfile->file_id, "/", H5P_DEFAULT);
    lombfile->status = H5Gget_info (lombfile->root_group, &ginfo);
    lombfile->nrecords = ginfo.nlinks;
    lombfile->pulseidx = 0;
    return lombfile->status;
}

int32_t LombFitClose(struct LombFile *lombfile)
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

    
    // return zero if no pulses remain to be read 
    if (lombfile->pulseidx >= lombfile->nrecords || lombfile->nrecords < 0) {
        return 0;
    }

    // get name of next record (get name size, allocate space, grab name)
    groupnamesize = 1 + H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, lombfile->pulseidx, NULL, 0, H5P_DEFAULT);
    groupname = (char *) malloc (groupnamesize);
    groupnamesize = H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, lombfile->pulseidx, groupname, (size_t) groupnamesize, H5P_DEFAULT);
    recordgroup = H5Gopen(lombfile->file_id, groupname, H5P_DEFAULT);
    
    // read record information into RadarParm and FitData structures
    fit->revision.major = REV_MAJOR;
    fit->revision.minor = REV_MINOR;
    // need to malloc struct.
    //rprm->revision.major = REV_MAJOR;
    //rprm->revision.minor = REV_MINOR;

    // read in noise information
    fit->noise.vel = 0; // not currently produced by fitlomb 
    fit->noise.skynoise = 0; // not current produced by fitlomb
    LombFitReadAttr(lombfile, groupname, "noise.lag0", &fit->noise.lag0);
    // populate rprm origin struct
    //rprm->origin.time = // char *
    
    RadarParmSetOriginCommand(rprm, ORIGIN);
    RadarParmSetCombf(rprm, COMMENT);
    rprm->origin.code = ORIGIN_CODE;
    
    time_t origin_time_raw;
    struct tm *origin_time;
    time(&origin_time_raw);
    origin_time = gmtime(&origin_time_raw);
    RadarParmSetOriginTime(rprm, asctime(origin_time));

    LombFitReadAttr(lombfile, groupname, "stid", &rprm->stid);
    LombFitReadAttr(lombfile, groupname, "cp", &rprm->cp);
    // populate rprm time struct (int16)
    LombFitReadAttr(lombfile, groupname, "time.yr", &rprm->time.yr);
    LombFitReadAttr(lombfile, groupname, "time.mo", &rprm->time.mo);
    LombFitReadAttr(lombfile, groupname, "time.dy", &rprm->time.dy);
    LombFitReadAttr(lombfile, groupname, "time.mt", &rprm->time.mt);
    LombFitReadAttr(lombfile, groupname, "time.sc", &rprm->time.sc);
    LombFitReadAttr(lombfile, groupname, "time.us", &rprm->time.us);

    LombFitReadAttr(lombfile, groupname, "txpow", &rprm->txpow);
    LombFitReadAttr(lombfile, groupname, "nave", &rprm->nave);
    LombFitReadAttr(lombfile, groupname, "atten", &rprm->atten);
    LombFitReadAttr(lombfile, groupname, "lagfr", &rprm->lagfr); 
    LombFitReadAttr(lombfile, groupname, "smsep", &rprm->smsep);
    LombFitReadAttr(lombfile, groupname, "ercod", &rprm->ercod);
    
    // populate stat struct (int16)
    //LombFitReadAttr(lombfile, groupname, "stat.agc", &rprm->stat.agc);
    //LombFitReadAttr(lombfile, groupname, "stat.lopwr", &rprm->stat.lopwr);
    
    // populate noise struct (float)
    LombFitReadAttr(lombfile, groupname, "noise.search", &rprm->noise.search);
    LombFitReadAttr(lombfile, groupname, "noise.mean", &rprm->noise.mean);
    
    LombFitReadAttr(lombfile, groupname, "channel", &rprm->channel);
    LombFitReadAttr(lombfile, groupname, "bmnum", &rprm->bmnum);
    LombFitReadAttr(lombfile, groupname, "bmazm", &rprm->bmazm);
    LombFitReadAttr(lombfile, groupname, "scan", &rprm->scan);
    LombFitReadAttr(lombfile, groupname, "rxrise", &rprm->rxrise);

    // populate intt structure
    LombFitReadAttr(lombfile, groupname, "intt.sc", &rprm->intt.sc);
    LombFitReadAttr(lombfile, groupname, "intt.us", &rprm->intt.us);
    
    LombFitReadAttr(lombfile, groupname, "txpl", &rprm->txpl);
    LombFitReadAttr(lombfile, groupname, "mpinc", &rprm->mpinc);
    LombFitReadAttr(lombfile, groupname, "mppul", &rprm->mppul);
    LombFitReadAttr(lombfile, groupname, "mplgs", &rprm->mplgs);
    LombFitReadAttr(lombfile, groupname, "", &rprm->mplgexs);
    LombFitReadAttr(lombfile, groupname, "nrang", &rprm->nrang);
    LombFitReadAttr(lombfile, groupname, "frang", &rprm->frang);
    LombFitReadAttr(lombfile, groupname, "rsep", &rprm->rsep);
    LombFitReadAttr(lombfile, groupname, "xc", &rprm->xcf);
    LombFitReadAttr(lombfile, groupname, "tfreq", &rprm->tfreq);
    LombFitReadAttr(lombfile, groupname, "offset", &rprm->offset);
    LombFitReadAttr(lombfile, groupname, "", &rprm->ifmode);

    LombFitReadAttr(lombfile, groupname, "mxpwr", &rprm->mxpwr);
    LombFitReadAttr(lombfile, groupname, "lvmax", &rprm->lvmax);

    // copy pulse and lag vectors
    int16_t *ltab, *ptab;
    ltab = LombFitReadVector(recordgroup, "ltab");
    ptab = LombFitReadVector(recordgroup, "ptab");
    RadarParmSetPulse(rprm, rprm->mppul, ptab);
    RadarParmSetLag(rprm, rprm->mplgs, ltab);

    // populate fit->rng vectors  
    double *v, *v_err, *p_0, *p_l, *p_l_err, *w_l, *w_l_err, *w_s, *w_s_err, *phi0, *phi0_err, *sdev_l, *sdev_s, *sdev_phi;
    int32_t *qflg, *gsct;
    int8_t *nump;
    uint16_t nrang = 0;
    uint16_t i;
    uint16_t lomb_iterations = 0;
    LombFitReadAttr(lombfile, groupname, "nrang", &nrang);
    LombFitReadAttr(lombfile, groupname, "fitlomb.bayes.iterations", &lomb_iterations);
    FitSetRng(fit, nrang); 

    v = LombFitReadVector(recordgroup, "v");
    v_err = LombFitReadVector(recordgroup, "v_e");
    p_0 = NULL;
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
    
    for(i = 0; i < nrang; i ++) {
        // calculate index into 2d array to pull the first iteration at a range (nrang by number of lomb iterations) 
        uint32_t idx = lomb_iterations * i;

        // copy over first first fit (ignore later iterations)
        // doubles
        fit->rng[i].v = v[idx];
        fit->rng[i].v_err = v_err[idx];
        fit->rng[i].p_l = p_l[idx];
        fit->rng[i].p_l_err = p_l_err[idx];
        fit->rng[i].w_l = w_l[idx];
        fit->rng[i].w_l_err = w_l_err[idx];
        fit->rng[i].w_s = w_s[idx];
        fit->rng[i].w_s_err = w_s_err[idx];
        fit->rng[i].sdev_l = sdev_l[idx];
        fit->rng[i].sdev_s = sdev_s[idx];

        // int32
        fit->rng[i].qflg = qflg[idx];

        // set unsupported parameters to -1
        fit->rng[i].p_0 = -1;
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
    free(ltab);
    free(ptab);

    lombfile->pulseidx++;
    lombfile->status = H5Gclose(recordgroup);
    return 1;
}

/* iterate through records, get as close possible without going over */
int LombFitSeek(struct LombFile *lombfile, int yr,int mo,int dy,int hr,int mt,int sc,double *atme)
{
    // convert seek time to epoch time 
    struct tm t;
    time_t seektime;
    int64_t recordtime;
    uint16_t i;

    t.tm_year = yr - 1900;
    t.tm_mon = mo - 1;  // months are 0 to 11...
    t.tm_mday = dy;
    t.tm_hour = hr;
    t.tm_min = mt;
    t.tm_sec = sc;
    t.tm_isdst = 0;
    t.tm_gmtoff = 0;
    seektime = timegm(&t);
    *atme = 0;   
    lombfile->pulseidx = 0;

    // check epoch time on records, compare against given time
    for(i = 0; i < lombfile->nrecords; i++) {
        char *groupname;
        ssize_t groupnamesize;

        // get name of next record (get name size, allocate space, grab name)
        groupnamesize = 1 + H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, i, NULL, 0, H5P_DEFAULT);
        groupname = (char *) malloc (groupnamesize);
        groupnamesize = H5Lget_name_by_idx (lombfile->root_group, ".", H5_INDEX_NAME, H5_ITER_INC, i, groupname, (size_t) groupnamesize, H5P_DEFAULT);
        
        // read off record time
        LombFitReadAttr(lombfile, groupname, "epoch.time", &recordtime);

        // compare with seek time, break and decrement if greater
        if (recordtime > seektime) {
            uint32_t time_us;
            lombfile->pulseidx = i - 1;

            LombFitReadAttr(lombfile, groupname, "time.us", &time_us);
            *atme = (double) recordtime + ((double) time_us) / 1e6;
            break;
        }
                    
    }
    
    return lombfile->pulseidx;
}


/* copied from fit.1.35/src/fit.c */
struct FitData * FitMake() {
    struct FitData *ptr=NULL;

    ptr=malloc(sizeof(struct FitData));

    if (ptr==NULL) return NULL;

    memset(ptr,0,sizeof(struct FitData));

    ptr->rng=NULL;
    ptr->xrng=NULL;
    ptr->elv=NULL;
    return ptr;
}

void FitFree(struct FitData *ptr) {
    if (ptr==NULL) return;
    if (ptr->rng !=NULL) free(ptr->rng);
    if (ptr->xrng !=NULL) free(ptr->xrng);
    if (ptr->elv !=NULL) free(ptr->elv);
    free(ptr);
    return;
}

int FitSetRng(struct FitData *ptr,int nrang) {
    void *tmp=NULL;
    if (ptr==NULL) return -1;

    if (nrang==0) {
        if (ptr->rng !=NULL) free(ptr->rng);
            ptr->rng=NULL;
            return 0;
    }

    if (ptr->rng==NULL) tmp=malloc(sizeof(struct FitRange)*nrang);
    else tmp=realloc(ptr->rng,sizeof(struct FitRange)*nrang);

    if (tmp==NULL) return -1;
    memset(tmp,0,sizeof(struct FitRange)*nrang);
    ptr->rng=tmp;
    return 0;
}

/* copied from radar1.21/src/rprm.c */
struct RadarParm *RadarParmMake() {
  struct RadarParm *ptr=NULL;

  ptr=malloc(sizeof(struct RadarParm));
  if (ptr==NULL) return NULL;
  memset(ptr,0,sizeof(struct RadarParm));
  ptr->origin.time=NULL;
  ptr->origin.command=NULL;
  ptr->pulse=NULL;
  ptr->lag[0]=NULL;
  ptr->lag[1]=NULL;
  ptr->combf=NULL;
  return ptr;
}

int RadarParmSetOriginCommand(struct RadarParm *ptr,char *str) {
  char *tmp=NULL;
  if (ptr==NULL) return -1;

  if (str==NULL) {
    if (ptr->origin.command !=NULL) free(ptr->origin.command);
    ptr->origin.command=NULL;
    return 0;
  }

  if (ptr->origin.command==NULL) tmp=malloc(strlen(str)+1);
  else tmp=realloc(ptr->origin.command,strlen(str)+1);

  if (tmp==NULL) return -1;
  strcpy(tmp,str);
  ptr->origin.command=tmp;
  return 0;

}

int RadarParmSetCombf(struct RadarParm *ptr,char *str) {
  void *tmp=NULL;
  if (ptr==NULL) return -1;

  if (str==NULL) {
    if (ptr->combf !=NULL) free(ptr->combf);
    ptr->combf=NULL;
    return 0;
  }

  if (ptr->combf==NULL) tmp=malloc(strlen(str)+1);
  else tmp=realloc(ptr->combf,strlen(str)+1);

  if (tmp==NULL) return -1;
  strcpy(tmp,str);
  ptr->combf=tmp;
  return 0;
}

int RadarParmSetOriginTime(struct RadarParm *ptr,char *str) {
  char *tmp=NULL;
  if (ptr==NULL) return -1;

  if (str==NULL) {
    if (ptr->origin.time !=NULL) free(ptr->origin.time);
    ptr->origin.time=NULL;
    return 0;
  }

  if (ptr->origin.time==NULL) tmp=malloc(strlen(str)+1);
  else tmp=realloc(ptr->origin.time,strlen(str)+1);

  if (tmp==NULL) return -1;
  strcpy(tmp,str);
  ptr->origin.time=tmp;
  return 0;

}
int RadarParmSetPulse(struct RadarParm *ptr,int mppul,int16 *pulse) {
  void *tmp=NULL;
  if (ptr==NULL) return -1;

  if ((mppul==0) || (pulse==NULL)) {
    if (ptr->pulse !=NULL) free(ptr->pulse);
    ptr->pulse=NULL;
    ptr->mppul=0;
    return 0;
  }

  if (ptr->pulse==NULL) tmp=malloc(sizeof(int16)*mppul);
  else tmp=realloc(ptr->pulse,sizeof(int16)*mppul);

  if (tmp==NULL) return -1;
  memcpy(tmp,pulse,sizeof(int16)*mppul);
  ptr->pulse=tmp;
  ptr->mppul=mppul;
  return 0;
}

int RadarParmSetLag(struct RadarParm *ptr,int mplgs,int16 *lag) {
  int n,x;
  void *tmp=NULL;
  if (ptr==NULL) return -1;

  if ((mplgs==0) || (lag==NULL)) {
    for (n=0;n<2;n++) {
      if (ptr->lag[n] !=NULL) free(ptr->lag[n]);
      ptr->lag[n]=NULL;
    }
    return 0;
  }

  for (n=0;n<2;n++) {
    if (ptr->lag[n]==NULL) tmp=malloc(sizeof(int16)*(mplgs+1));
    else tmp=realloc(ptr->lag[n],sizeof(int16)*(mplgs+1));
    if (tmp==NULL) return -1;
    ptr->lag[n]=tmp;
    for (x=0;x<=mplgs;x++) ptr->lag[n][x]=lag[2*x+n];
  }
  return 0;
}



void RadarParmFree(struct RadarParm *ptr) {
  if (ptr==NULL) return;
  if (ptr->origin.time !=NULL) free(ptr->origin.time);
  if (ptr->origin.command !=NULL) free(ptr->origin.command);
  if (ptr->pulse !=NULL) free(ptr->pulse);
  if (ptr->lag[0] !=NULL) free(ptr->lag[0]);
  if (ptr->lag[1] !=NULL) free(ptr->lag[1]);
  if (ptr->combf !=NULL) free(ptr->combf);
  free(ptr);
}

/* example test code */
int main(void)
{
    struct LombFile lombfile;
    struct RadarParm *prm;
    struct FitData *fit;
    double atme;
    
    // create fit and prm structs, open lombfit file
    fit = FitMake();
    prm = RadarParmMake();
    LombFitOpen(&lombfile, FILE);
    
    // read the first record
    LombFitRead(&lombfile, prm, fit);
    
    // seek to a time, then read the record at that time
    LombFitSeek(&lombfile, 2014,2,26,1,20,27, &atme);
    LombFitRead(&lombfile, prm, fit);
    
    // read off remaining records..
    while(LombFitRead(&lombfile, prm, fit)) {
        printf(".");
    }
    printf("\n");
    
    // clean up..
    LombFitClose(&lombfile);
    RadarParmFree(prm);
    FitFree(fit);
    return 0;
}


