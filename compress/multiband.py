"""Band container for the cm=7 (.fpred) and cm=8 (.sz3) payloads.

A single-band raster is stored as one payload. A multi-band raster is
stored as MBND | uint32 band_count | (uint64 length | payload) per band. Every payload is a
complete single-band stream, so each band is quantized, predicted or compressed, and checked
on its own.
"""

import struct

MULTI_MAGIC = b"MBND"


def pack_bands(payloads):
    if len(payloads) == 1:
        return payloads[0]
    parts = [MULTI_MAGIC, struct.pack("<I", len(payloads))]
    for payload in payloads:
        parts.append(struct.pack("<Q", len(payload)))
        parts.append(payload)
    return b"".join(parts)


def split_bands(data: bytes):
    if data[:4] != MULTI_MAGIC:
        return [data]
    (count,) = struct.unpack_from("<I", data, 4)
    pos = 8
    payloads = []
    for _ in range(count):
        if pos + 8 > len(data):
            raise ValueError("truncated multi-band file")
        (length,) = struct.unpack_from("<Q", data, pos)
        pos += 8
        if pos + length > len(data):
            raise ValueError("truncated multi-band file")
        payloads.append(data[pos : pos + length])
        pos += length
    return payloads
