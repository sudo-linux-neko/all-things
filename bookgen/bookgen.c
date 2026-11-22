// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 Vaibhav Agarwal

/*
 * bookgen.c - Hyper-optimized exhaustive-combination book generator
 *
 * Author : vaibhav (project owner; holds final say over this project)
 * License: TBD - if ever released publicly, intended default is GPL-v3,
 *          but ONLY the project owner (vaibhav) decides. Not MIT.
 *
 * Build  : gcc -O3 -march=native -funroll-loops -fopenmp bookgen.c -lpthread -lz -o bookgen
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stddef.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <omp.h>
#include <zlib.h>

#define MAX_THREADS   64
#define PREVIEW_LEN   76
#define CYCLE_BOOKS   4096

static const char *articles[] = { "the", "a", "an", "this", "that", "every", "each" };
static const char *pronouns[] = { "he", "she", "it", "the old man", "the young woman", "the traveler" };
static const char *pronouns_p[] = { "they", "we", "the children", "the travelers" };
static const char *auxverbs[] = { "walks", "runs", "remembers", "forgets", "discovers", "builds", "dreams", "questions",
                                 "watches", "follows", "imagines", "measures", "describes", "protects", "greets", "listens" };
static const char *auxverbs_p[] = { "walk", "run", "remember", "forget", "discover", "build", "dream", "question",
                                    "watch", "follow", "imagine", "measure", "describe", "protect", "greet", "listen" };
static const char *adj      [] = { "quiet", "ancient", "bright", "silent", "gentle", "fierce", "curious", "hollow",
                                   "golden", "forgotten", "patient", "restless", "distant", "tender", "broken", "hidden",
                                   "vivid", "pale", "endless", "narrow", "small", "vast", "weary", "calm" };
static const char *adverbs  [] = { "slowly", "quietly", "suddenly", "always", "never", "softly", "carefully", "boldly",
                                   "rarely", "often", "gently", "fiercely", "silently", "eagerly" };
static const char *preps    [] = { "through", "beyond", "beneath", "across", "within", "toward", "behind", "over",
                                   "under", "beside", "without", "among" };
static const char *conj     [] = { "and", "but", "while", "because", "although", "until", "since" };

static char    **words = NULL;
static uint32_t *wlen  = NULL;
static uint64_t  nwords = 0;

enum { SLOT_NOUN, SLOT_ART, SLOT_ADJ, SLOT_AUXV, SLOT_ADV, SLOT_PREP, SLOT_PRON, SLOT_CONJ, SLOT_PRON_P, SLOT_AUXV_P };

typedef struct { const uint8_t *slots; uint8_t nslots; } tpl_t;

static const uint8_t t0_slots[] = { SLOT_ART, SLOT_ADJ, SLOT_NOUN, SLOT_AUXV, SLOT_ADV };
static const uint8_t t1_slots[] = { SLOT_PRON, SLOT_AUXV, SLOT_ADV, SLOT_PREP, SLOT_ART, SLOT_ADJ, SLOT_NOUN };
static const uint8_t t2_slots[] = { SLOT_ART, SLOT_NOUN, SLOT_AUXV, SLOT_ADV, SLOT_CONJ, SLOT_ART, SLOT_ADJ, SLOT_NOUN, SLOT_AUXV, SLOT_ADV };
static const uint8_t t3_slots[] = { SLOT_ART, SLOT_ADJ, SLOT_NOUN, SLOT_AUXV, SLOT_ART, SLOT_NOUN };
static const uint8_t t4_slots[] = { SLOT_PRON, SLOT_AUXV, SLOT_PREP, SLOT_ART, SLOT_ADJ, SLOT_NOUN, SLOT_ADV };
static const uint8_t t5_slots[] = { SLOT_PRON_P, SLOT_AUXV_P, SLOT_ADV, SLOT_PREP, SLOT_ART, SLOT_ADJ, SLOT_NOUN };

static const tpl_t templates[] = {
    { t0_slots, sizeof t0_slots / sizeof t0_slots[0] },
    { t1_slots, sizeof t1_slots / sizeof t1_slots[0] },
    { t2_slots, sizeof t2_slots / sizeof t2_slots[0] },
    { t3_slots, sizeof t3_slots / sizeof t3_slots[0] },
    { t4_slots, sizeof t4_slots / sizeof t4_slots[0] },
    { t5_slots, sizeof t5_slots / sizeof t5_slots[0] },
};
#define NTMPL (sizeof(templates)/sizeof(templates[0]))

static uint64_t tpl_prefix[NTMPL+1];
static uint64_t sentence_space = 0;

static inline uint64_t slot_size(uint8_t s) {
    switch (s) {
        case SLOT_NOUN:   return nwords;
        case SLOT_ART:    return sizeof(articles)/sizeof(*articles);
        case SLOT_ADJ:    return sizeof(adj)/sizeof(*adj);
        case SLOT_AUXV:   return sizeof(auxverbs)/sizeof(*auxverbs);
        case SLOT_ADV:    return sizeof(adverbs)/sizeof(*adverbs);
        case SLOT_PREP:   return sizeof(preps)/sizeof(*preps);
        case SLOT_PRON:   return sizeof(pronouns)/sizeof(*pronouns);
        case SLOT_CONJ:   return sizeof(conj)/sizeof(*conj);
        case SLOT_PRON_P: return sizeof(pronouns_p)/sizeof(*pronouns_p);
        case SLOT_AUXV_P: return sizeof(auxverbs_p)/sizeof(*auxverbs_p);
    }
    return 1;
}

/* decode sentence number c into slot indices (used once per book) */
static inline void decode_sentence(uint64_t c, uint64_t idx[], int *ti) {
    int t = 0;
    while (t < (int)NTMPL - 1 && c >= tpl_prefix[t+1]) t++;
    uint64_t r = c - tpl_prefix[t];
    *ti = t;
    for (int i = templates[t].nslots - 1; i >= 0; i--) {
        uint64_t m = slot_size(templates[t].slots[i]);
        idx[i] = r % m;
        r /= m;
    }
}

/* odometer advance: next slot index tuple in current template */
static inline void odo_next(uint64_t idx[], int ti) {
    const uint8_t *sl = templates[ti].slots;
    int ns = templates[ti].nslots;
    int i = ns - 1;
    while (i >= 0) {
        idx[i]++;
        if (idx[i] < slot_size(sl[i])) return;
        idx[i] = 0;
        i--;
    }
}

/* render one sentence, return length */
static inline size_t render_sentence(const uint64_t idx[], int ti, char *buf) {
    size_t used = 0;
    const uint8_t *sl = templates[ti].slots;
    for (int i = 0; i < templates[ti].nslots; i++) {
        if (i) {
            if (sl[i] == SLOT_CONJ) buf[used++] = ',';
            buf[used++] = ' ';
        }
        const char *w; uint32_t L;
        switch (sl[i]) {
            case SLOT_NOUN:   w = words[idx[i]];    L = wlen[idx[i]]; break;
            case SLOT_ART:    w = articles[idx[i]]; L = (uint32_t)strlen(w); break;
            case SLOT_ADJ:    w = adj[idx[i]];      L = (uint32_t)strlen(w); break;
            case SLOT_AUXV:   w = auxverbs[idx[i]]; L = (uint32_t)strlen(w); break;
            case SLOT_AUXV_P: w = auxverbs_p[idx[i]]; L = (uint32_t)strlen(w); break;
            case SLOT_ADV:    w = adverbs[idx[i]];  L = (uint32_t)strlen(w); break;
            case SLOT_PREP:   w = preps[idx[i]];    L = (uint32_t)strlen(w); break;
            case SLOT_PRON:   w = pronouns[idx[i]]; L = (uint32_t)strlen(w); break;
            case SLOT_PRON_P: w = pronouns_p[idx[i]]; L = (uint32_t)strlen(w); break;
            default:          w = conj[idx[i]];     L = (uint32_t)strlen(w); break;
        }
        memcpy(buf + used, w, L);
        used += L;
    }
    buf[used++] = '.';
    buf[used++] = '\n';
    if (buf[0] >= 'a' && buf[0] <= 'z') buf[0] = (char)(buf[0] - 32);
    return used;
}

static atomic_uint_fast64_t books_done;
static atomic_uint_fast64_t bytes_done;
static char previews[MAX_THREADS][PREVIEW_LEN + 1];
static volatile int stop_flag = 0;

static int load_words(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open word list %s: %s\n", path, strerror(errno)); return -1; }
    uint64_t cap = 1 << 20, n = 0;
    words = malloc(cap * sizeof(char*));
    wlen  = malloc(cap * sizeof(uint32_t));
    char line[256];
    while (fgets(line, sizeof line, f)) {
        size_t L = strlen(line);
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (L < 3 || L > 20) continue;
        int ok = 1;
        for (size_t i = 0; i < L; i++)
            if (!((line[i] >= 'a' && line[i] <= 'z') || (line[i] >= 'A' && line[i] <= 'Z'))) { ok = 0; break; }
        if (!ok) continue;
        if (n == cap) { cap <<= 1; words = realloc(words, cap * sizeof(char*)); wlen = realloc(wlen, cap * sizeof(uint32_t)); }
        char *w = malloc(L + 1);
        memcpy(w, line, L + 1);
        words[n] = w; wlen[n] = (uint32_t)L;
        n++;
    }
    fclose(f);
    nwords = n;
    return 0;
}

static double now_sec(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void fmt_num(uint64_t v, char *out) {
    if (v >= 1000000) sprintf(out, "%.1fM", v / 1e6);
    else if (v >= 1000) sprintf(out, "%.1fk", v / 1e3);
    else sprintf(out, "%llu", (unsigned long long)v);
}

static const char spinner[] = "|/-\\";

static void *display(void *arg) {
    (void)arg;
    uint64_t last_books = 0, last_bytes = 0;
    double t0 = now_sec(), last = t0;
    int spin = 0;
    char bar[41]; bar[40] = 0;
    while (!stop_flag) {
        struct timespec ts = { 0, 80 * 1000 * 1000 };
        nanosleep(&ts, NULL);
        double t = now_sec();
        uint64_t b = atomic_load(&books_done);
        uint64_t by = atomic_load(&bytes_done);
        double dt = t - last; if (dt <= 0) dt = 1e-9;
        double bps = (b - last_books) / dt;
        double mbs = (by - last_bytes) / dt / (1024.0 * 1024.0);
        last_books = b; last_bytes = by; last = t;
        uint64_t cycle = b / CYCLE_BOOKS;
        int prog = (int)((b % CYCLE_BOOKS) * 40 / CYCLE_BOOKS);
        for (int i = 0; i < 40; i++) bar[i] = i < prog ? '=' : (i == prog ? '>' : '-');
        char nb[16], bb[16];
        fmt_num(b, nb); fmt_num(by / (1<<20), bb);
        const char *pv = previews[0];
        for (int i = 0; i < MAX_THREADS; i++) if (previews[i][0]) { pv = previews[i]; break; }
        printf("\r\033[2K\033[1;36m[%s]\033[0m %c cycle %llu/%llu | \033[1;33m%s\033[0m books | \033[1;32m%.0f\033[0m books/s | \033[1;32m%.1f\033[0m MiB/s | \033[1;37m%s\033[0m MiB total | %.0fs",
               bar, spinner[spin++ & 3], (unsigned long long)cycle, (unsigned long long)(b / CYCLE_BOOKS + 1), nb, bps, mbs, bb, t - t0);
        printf("\n\033[2K\033[35m>> %s\033[0m\033[1A", pv);
        fflush(stdout);
    }
    printf("\033[2B\n");
    return NULL;
}

static void on_sigint(int sig) {
    (void)sig;
    printf("\n\033[2K\033[1;31mInterrupted.\033[0m %llu books, %llu MiB total.\n",
           (unsigned long long)atomic_load(&books_done),
           (unsigned long long)(atomic_load(&bytes_done) >> 20));
    _exit(130);
}

int main(int argc, char **argv) {
    const char *dict = "words.txt";
    const char *outdir = "books";
    int nsent = 0;
int pages = 200;
int per_page = 12;
int raw_gz = 0;
const char *decode_path = NULL;
    int threads = omp_get_max_threads();
    if (threads > MAX_THREADS) threads = MAX_THREADS;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--dict") && i + 1 < argc) dict = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) outdir = argv[++i];
        else if (!strcmp(argv[i], "--sentences") && i + 1 < argc) nsent = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--pages") && i + 1 < argc) pages = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--per-page") && i + 1 < argc) per_page = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--gzip")) raw_gz = 1;
        else if (!strcmp(argv[i], "--decode") && i + 1 < argc) decode_path = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--help")) {
            printf("bookgen [--dict words.txt] [--out books] [--pages N] [--per-page M] [--sentences N] [--threads T] [--gzip] | bookgen --decode FILE [--dict D]\n");
            return 0;
        }
    }
    if (load_words(dict)) return 1;
    if (nsent) { pages = (nsent + per_page - 1) / per_page; }
    mkdir(outdir, 0755);

    tpl_prefix[0] = 0;
    for (uint64_t t = 0; t < NTMPL; t++) {
        uint64_t sp = 1;
        for (uint8_t i = 0; i < templates[t].nslots; i++) sp *= slot_size(templates[t].slots[i]);
        tpl_prefix[t+1] = tpl_prefix[t] + sp;
    }
    sentence_space = tpl_prefix[NTMPL];

    printf("\033[1;32mbookgen v2\033[0m by vaibhav | %llu words | %llu sentence combos | %d threads\n\n",
           (unsigned long long)nwords, (unsigned long long)sentence_space, threads);

    if (decode_path) {
        int fd = open(decode_path, O_RDONLY);
        if (fd < 0) { perror("decode"); return 1; }
        uint8_t hdr[16];
        if (read(fd, hdr, 16) != 16) { fprintf(stderr, "bad file\n"); return 1; }
        close(fd);
        uint64_t c0; memcpy(&c0, hdr, 8);
        uint32_t u1, u2; memcpy(&u1, hdr + 8, 4); memcpy(&u2, hdr + 12, 4);
        uint64_t idx2[16]; int ti2 = 0;
        decode_sentence(c0, idx2, &ti2);
        char *buf2 = malloc((size_t)u1 * u2 * 256 + (1 << 20));
        size_t used = 0;
        used += (size_t)snprintf(buf2, 1 << 20, "BOOK %llu\n==========\n\n", (unsigned long long)(c0 / ((uint64_t)u1 * u2)));
        for (uint32_t p = 0; p < u1; p++) {
            used += (size_t)snprintf(buf2 + used, (1 << 20), "Page %u\n\n", p + 1);
            for (uint32_t s2 = 0; s2 < u2; s2++) {
                used += render_sentence(idx2, ti2, buf2 + used);
                if ((s2 & 5) == 5) buf2[used++] = '\n';
                odo_next(idx2, ti2);
            }
            buf2[used++] = '\n';
            if (used > (1 << 24)) { fwrite(buf2, 1, used, stdout); used = 0; }
        }
        fwrite(buf2, 1, used, stdout);
        return 0;
    }

    memset(previews, 0, sizeof previews);
    atomic_store(&books_done, 0);
    atomic_store(&bytes_done, 0);
    signal(SIGINT, on_sigint);

    pthread_t dt; pthread_create(&dt, NULL, display, NULL);

    uint64_t next_book = 0;
    #pragma omp parallel num_threads(threads) shared(next_book)
    {
        int tid = omp_get_thread_num();
        uint64_t idx[16]; int ti = 0;
        size_t cap = (size_t)pages * per_page * 256 + (1 << 20);
        char *buf = malloc(cap);
        int dirfd = open(outdir, O_RDONLY);
        for (;;) {
            uint64_t b;
            #pragma omp critical
            b = next_book++;
            char path[1024];
            int pl = snprintf(path, sizeof path, raw_gz ? "book_%08llu.txt.gz" : "book_%08llu.bkg", (unsigned long long)b);
            (void)pl;
            int npages = pages;
            uint64_t c0 = (b * (uint64_t)npages * per_page) % sentence_space;
            decode_sentence(c0, idx, &ti);
            size_t used = 0;
            used += (size_t)snprintf(buf, cap, "BOOK %llu\n==========\n\n", (unsigned long long)b);
            int preview_done = 0;
            for (int p = 0; p < npages; p++) {
                used += (size_t)snprintf(buf + used, cap - used, "Page %d\n\n", p + 1);
                for (int s = 0; s < per_page; s++) {
                    size_t L = render_sentence(idx, ti, buf + used);
                    used += L;
                    if ((s & 5) == 5) buf[used++] = '\n';
                    odo_next(idx, ti);
                    if (!preview_done && used > PREVIEW_LEN + 24) {
                        const char *src = strstr(buf, "\n\n");
                        src = src ? src + 2 : buf;
                        for (size_t i = 0; i < PREVIEW_LEN; i++) previews[tid][i] = (src[i] == '\n' ? ' ' : src[i]);
                        previews[tid][PREVIEW_LEN] = 0;
                        preview_done = 1;
                    }
                }
                buf[used++] = '\n';
            }
            if (!raw_gz) {
                uint8_t hdr[16];
                memcpy(hdr, &c0, 8);
                uint32_t u1 = (uint32_t)npages, u2 = (uint32_t)per_page;
                memcpy(hdr + 8, &u1, 4); memcpy(hdr + 12, &u2, 4);
                int fd0 = openat(dirfd, path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
                if (fd0 >= 0) { if (write(fd0, hdr, 16) != 16) {} close(fd0); }
                atomic_fetch_add(&books_done, 1);
                atomic_fetch_add(&bytes_done, 16);
                continue;
            }
            uLongf cp = (uLongf)(used + (used >> 12) + (used >> 14) + 11);
            char *cbuf = malloc(cp != 0 ? cp : 1);
            z_stream zs; memset(&zs, 0, sizeof zs);
            if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 + 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) cp = 0;
            else {
                zs.next_in = (Bytef*)buf; zs.avail_in = used;
                zs.next_out = (Bytef*)cbuf; zs.avail_out = cp;
                if (deflate(&zs, Z_FINISH) != Z_STREAM_END) cp = 0; else cp = zs.total_out;
                deflateEnd(&zs);
            }
            int fd = openat(dirfd, path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) {
                const char *p2 = cbuf; size_t left = cp;
                while (left) { ssize_t w = write(fd, p2, left); if (w <= 0) break; p2 += w; left -= (size_t)w; }
                close(fd);
            }
            free(cbuf);
            atomic_fetch_add(&books_done, 1);
            atomic_fetch_add(&bytes_done, used);
        }
    }
    stop_flag = 1;
    pthread_join(dt, NULL);
    return 0;
}
