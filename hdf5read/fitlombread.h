struct LombFile {
    hid_t file_id;
    hid_t root_group;
    herr_t status; /* status of last HDF5 command */
    hsize_t nrecords; /* number of pulses in file */
    hsize_t pulseidx; /* index of current pulse in array */
};


int32_t LombFitOpen(struct LombFile *lombfile, char *filename);
int32_t LombFitClose(struct LombFile *lombfile);
herr_t LombFitReadAttr(struct LombFile *lombfile, char *groupname, char *attrname, void *attrdata);
void * LombFitReadVector(hid_t recordgroup, char *dsetname);
int LombFitRead(struct LombFile *lombfile, struct RadarParm *rprm, struct FitData *fit);
int LombFitSeek(struct LombFile *lombfile, int yr,int mo,int dy,int hr,int mt,int sc,double *atme);




