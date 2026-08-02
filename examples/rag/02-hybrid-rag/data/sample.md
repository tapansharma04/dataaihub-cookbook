# Acme Edge Platform — Support Handbook

Tiny on purpose. Each ## section is one retrieval chunk so dense and lexical
signals can disagree honestly.

## Restoring service when the network keeps dropping

When users report that packets vanish at random times of day, begin with
environmental triage. Confirm whether other devices share the symptom, reseat
cables, and reboot the customer edge router. Most transient outages clear after
a simple restart cycle. Capture environment notes before escalating to platform
engineering. This guidance covers general connectivity instability only.

## Error code E_CONN_42

E_CONN_42. EdgeGateway. TLS handshake against regional relay failed.
Remediation for E_CONN_42: rotate gateway client certificate; confirm
relay-west.acme.internal is reachable on port 8443; run `edgectl session flush`.
Do not treat E_CONN_42 as a generic timeout or packet-loss incident.

## Softening slow backend profile reads

Clients should treat remote user data as eventually consistent. Prefer an
optimistic UI, keep a cached profile snapshot, and retry with backoff when the
service is sluggish. Do not freeze the main application thread while waiting on
the network. No vendor-specific method names are required for this pattern.

## NebulaAPI v2.3 release notes

NebulaAPI v2.3 introduces `getUserProfileAsync` as the preferred call for
non-blocking profile fetches. The older `getUserProfile` API is deprecated in
v2.3 and may throw under high concurrency. Migration for NebulaAPI v2.3:
replace synchronous reads with getUserProfileAsync and handle the Promise.

## Product SKU NX-4400-PRO

NX-4400-PRO ships with dual power supplies. Rack mounting requires NX-RAIL-2.
Supported firmware baseline for NX-4400-PRO is 14.2.1 or newer. Field swaps
must match the NX-4400-PRO SKU exactly.
