/* k3_state.c - see k3_state.h. Moved out of k3_run.c unchanged so it can be tested. */
#define _POSIX_C_SOURCE 200809L

#include <stdio.h>
#include <string.h>

#include "k3_state.h"

void k3_state_fp(const K3Cfg *c, int32_t *fp)
{
    fp[0] = c->hidden;      fp[1] = c->n_layers;  fp[2]  = c->vocab;
    fp[3] = c->kda_heads;   fp[4] = c->kda_head_dim; fp[5] = c->conv_k;
    fp[6] = c->n_heads;     fp[7] = c->qk_nope;   fp[8]  = c->qk_rope;
    fp[9] = c->v_head;      fp[10] = c->n_experts; fp[11] = c->topk;
}

int k3_state_peek(const char *path, K3StateHdr *hd)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    const size_t got = fread(hd, 1, sizeof *hd, f);
    fclose(f);
    if (got != sizeof *hd || memcmp(hd->magic, K3_STATE_MAGIC, 4) != 0) {
        fprintf(stderr, "%s is not a k3 state file\n", path);
        return -1;
    }
    if (hd->version != K3_STATE_VER) {
        fprintf(stderr, "%s is state version %d, this build writes %d\n",
                path, hd->version, K3_STATE_VER);
        return -1;
    }
    return 0;
}

int k3_state_load(const char *path, const K3Cfg *c, const K3StateHdr *hd,
                  int *seq, float *ks, float *kvc, float *ropec,
                  int n_bound, int n_mla, int kv_cap)
{
    int32_t fp[12];
    k3_state_fp(c, fp);
    if (memcmp(fp, hd->fp, sizeof fp) != 0) {
        fprintf(stderr, "REFUSING: %s was written by a different model architecture.\n"
                        "  Restoring it would produce fluent, wrong output.\n", path);
        return -1;
    }
    if (hd->n_bound != n_bound || hd->n_mla != n_mla) {
        fprintf(stderr, "REFUSING: %s holds %d bound layers and %d MLA layers, "
                        "this run has %d and %d\n",
                path, hd->n_bound, hd->n_mla, n_bound, n_mla);
        return -1;
    }
    if (hd->cached > kv_cap) {
        fprintf(stderr, "REFUSING: %s holds %d positions, this run's KV cache is %d.\n"
                        "  Raise --gen or shorten the prompt.\n", path, hd->cached, kv_cap);
        return -1;
    }
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return -1; }
    if (fseek(f, (long)sizeof *hd, SEEK_SET) != 0) { fclose(f); return -1; }

    int rc = 0;
    if (fread(seq, sizeof(int), (size_t)hd->nseq, f) != (size_t)hd->nseq) rc = -1;
    if (!rc && fread(ks, sizeof(float), (size_t)hd->kper * n_bound, f)
               != (size_t)hd->kper * (size_t)n_bound) rc = -1;
    /* Position-major inside each layer slice, so a differently-sized destination cache
     * is written slice by slice rather than as one block. */
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = kvc + (size_t)mi * kv_cap * hd->kvpp;
        const size_t n = (size_t)hd->cached * hd->kvpp;
        if (fread(dst, sizeof(float), n, f) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        float *dst = ropec + (size_t)mi * kv_cap * hd->ropepp;
        const size_t n = (size_t)hd->cached * hd->ropepp;
        if (fread(dst, sizeof(float), n, f) != n) rc = -1;
    }
    fclose(f);
    if (rc) fprintf(stderr, "%s is truncated\n", path);
    return rc;
}

int k3_state_save(const char *path, const K3Cfg *c, const int *seq, int nseq,
                  const float *ks, const float *kvc, const float *ropec,
                  int n_bound, int n_mla, int kv_cap, int cached,
                  int64_t kper, int64_t kvpp, int64_t ropepp)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); return -1; }
    K3StateHdr hd;
    memset(&hd, 0, sizeof hd);
    memcpy(hd.magic, K3_STATE_MAGIC, 4);
    hd.version = K3_STATE_VER;
    k3_state_fp(c, hd.fp);
    hd.n_bound = n_bound; hd.n_mla = n_mla; hd.cached = cached; hd.nseq = nseq;
    hd.kper = kper; hd.kvpp = kvpp; hd.ropepp = ropepp;

    int rc = 0;
    if (fwrite(&hd, sizeof hd, 1, f) != 1) rc = -1;
    if (!rc && fwrite(seq, sizeof(int), (size_t)nseq, f) != (size_t)nseq) rc = -1;
    if (!rc && fwrite(ks, sizeof(float), (size_t)kper * n_bound, f)
               != (size_t)kper * (size_t)n_bound) rc = -1;
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = kvc + (size_t)mi * kv_cap * kvpp;
        const size_t n = (size_t)cached * kvpp;
        if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
    }
    for (int mi = 0; !rc && mi < n_mla; mi++) {
        const float *src = ropec + (size_t)mi * kv_cap * ropepp;
        const size_t n = (size_t)cached * ropepp;
        if (fwrite(src, sizeof(float), n, f) != n) rc = -1;
    }
    if (fclose(f) != 0) rc = -1;
    if (rc) fprintf(stderr, "failed writing %s\n", path);
    return rc;
}
