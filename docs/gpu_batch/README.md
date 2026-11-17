# FIDWAC batch copy (GPU + lockstep) — prototype, NOT production

Trwała kopia prototypu z `/dev/shm/fidwac_batch_copy`. Produkcja (`/mnt/g/temp/FIDWAC/fidwac_2026_recenzje/FIDWAC_v2`) nie jest zmieniona.

## Co jest w kopii (względem produkcji)

- `engine/u8_lockstep.py`
  - `_search_row` — jeden wiersz (blok) wyszukiwania uint8 (binarne po połówkach, backscan, weryfikacja, kwantyzacja).
  - `search_u8_par` (`prange`), `search_u8` (wejście: GPU → prange; `performance.gpu = off` wyłącza GPU).
  - `tie_rows` — wiersze z remisem zaokrąglenia w weryfikacji całkowitej (IDCT scipy vs suma bazy).
- `engine/gpu_u8/`
  - `search_u8.cu` — kernel CUDA C odwzorowujący `_search_row` (float64, `-fmad=false`).
  - `__init__.py` — ładowanie `libfidwac_u8.so` przez ctypes, `available()`, fallback.
- `engine/blocks.py`
  - `_fix_tie_rows` — wiersze z remisem liczy produkcyjne `refine_dct_array`.
  - `_batched_refine` (cm=5) i `_rgb_batch_lockstep` (cm=6) — wsadowe ścieżki (domyślnie aktywne).

## Build GPU

```
NV=/home/infostrateg/.fidwac/venv/lib/python3.13/site-packages/nvidia/cu13
$NV/bin/nvcc -O3 -arch=sm_75 -fmad=false -shared -Xcompiler -fPIC -cudart shared \
  -L$NV/lib engine/gpu_u8/search_u8.cu -o engine/gpu_u8/libfidwac_u8.so
```
Uruchomienie wymaga `LD_LIBRARY_PATH=$NV/lib` (lub wrapper sam ładuje `libcudart.so.13`).

## Uruchomienie

- Ścieżka wsadowa z GPU: `python3 compress.py -i ... -o ... --uint8-accuracy 5 --block-size 8 --rgb-channel-indices 1,2,3 --ycbcr-per-block -p 1 --quiet`
- Bez GPU: `performance.gpu = off`. Ścieżka tablicowa jest domyślna; per blok działa dla przypadków, których tablice nie obsługują.

## Stan weryfikacji (na danych `test/multi_uint_6ch`)

- Parzystość GPU vs szeregowo vs `prange` na danych rzeczywistych i stresowych: identyczne `LF`, `ERR`, `RES` (`tests_batch/test_gpu_parity.py`).
- cm=6 różnicowo (ta sama instancja, pętla per-blok vs lockstep, 10 000 bloków): 0 różnic (`tests_batch/cm6_diff.py`).
- End-to-end z GPU + lockstep, archiwa i zdekodowane piksele identyczne z produkcją dla 3 plików.
- Czas, plik 1 w tych samych warunkach: produkcja 143,7 s, kopia 53,4 s (ok. 2,7×). Plik 2 i 3: 60,7 s i 58,8 s (bez porównania w tych samych warunkach).

## Znane ograniczenia

- Kernel GPU działa jeden wątek na wiersz (FP64); na samym wyszukiwaniu nie jest szybszy od `prange`. Planowany jest wariant pixel-parallel.
- Dane nie zostają w VRAM między kanałami/plikami (transfer na batch).
- Nie wyizolowano udziału GPU od udziału lockstepu i `prange` w czasie e2e.
- Ścieżka bez GPU po ostatnich zmianach nie była testowana end-to-end.
- Poprawka bramki ulp w `_rgb_batch_lockstep` (blok `near`) nie była przyczyną rozbieżności i może zostać usunięta.

## Skrypty testowe (`tests_batch/`)

Mają zaszyte ścieżki (`/dev/shm/fidwac_batch_copy`, `/mnt/g/.../FIDWAC_v2`, katalog scratchpad sesji). Przed użyciem trzeba je zaktualizować do tej lokalizacji.
