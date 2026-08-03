# Acme Edge Platform — Support Handbook

Tiny on purpose. Each ## section is one retrieval chunk. Several topics share
vocabulary (TLS, profiles, NebulaAPI, timeouts) so hybrid retrieval can surface
plausible candidates while a cross-encoder decides final relevance.

## Restoring service when the network keeps dropping

When users report that packets vanish at random times of day, begin with
environmental triage. Confirm whether other devices share the symptom, reseat
cables, and reboot the customer edge router. Most transient outages clear after
a simple restart cycle. Capture environment notes before escalating to platform
engineering. This guidance covers general connectivity instability only — not
certificate or TLS handshake error codes.

## Error code E_CONN_42

E_CONN_42. EdgeGateway. TLS handshake against regional relay failed.
Remediation for E_CONN_42: rotate the gateway client certificate used by the
EdgeGateway process; confirm relay-west.acme.internal is reachable on port
8443; run `edgectl session flush`. Do not treat E_CONN_42 as a generic timeout,
packet-loss incident, or routine certificate calendar rotation.

## Softening slow backend profile reads

Clients should treat remote user data as eventually consistent. Prefer an
optimistic UI, keep a cached profile snapshot, and retry with backoff when the
service is sluggish. Do not freeze the main application thread while waiting on
the network. No vendor-specific method names are required for this pattern.

## NebulaAPI v2.3 asynchronous profile fetches

NebulaAPI v2.3 introduces `getUserProfileAsync` as the preferred call for
non-blocking profile fetches. Migration for NebulaAPI v2.3: replace synchronous
reads with getUserProfileAsync and handle the Promise. Under high concurrency,
this is the supported path for loading user profiles without blocking the UI.

## NebulaAPI legacy synchronous profile reads

The older `getUserProfile` API remains documented for NebulaAPI compatibility
review. In v2.3 it is deprecated and may throw under high concurrency. Prefer
migrating callers away from getUserProfile rather than increasing thread-pool
size around the synchronous call.

## Gateway TLS certificate rotation schedule

Production EdgeGateway fleets rotate TLS client certificates on a 90-day
calendar. Operators schedule the rotation window, stage the next certificate in
`/etc/acme/gateway/certs/next`, and promote it with `edgectl cert promote`.
Calendar rotation is preventive maintenance. It is not the remediation procedure
for live E_CONN_42 handshake failures.

## EdgeGateway timeout configuration

EdgeGateway exposes `relay_connect_timeout_ms` (default 5000) and
`relay_handshake_timeout_ms` (default 8000). Raise these values only when
latency to relay-west.acme.internal is known to be high. Timeout configuration
does not fix certificate mismatch errors and should not be used as the first
response to E_CONN_42.

## Troubleshooting intermittent TLS handshake failures

When TLS handshakes fail intermittently without a stable error code, capture
packet traces, verify NTP clock skew under 2 seconds, and confirm intermediate
CA bundles are complete. This playbook covers undiagnosed handshake flakiness.
If the gateway logs E_CONN_42 specifically, follow the E_CONN_42 remediation
instead of this general checklist.

## Product SKU NX-4400-PRO

NX-4400-PRO ships with dual power supplies. Rack mounting requires NX-RAIL-2.
Supported firmware baseline for NX-4400-PRO is 14.2.1 or newer. Field swaps
must match the NX-4400-PRO SKU exactly.

## Caching strategies for user profile data

Edge clients may cache NebulaAPI profile payloads for up to five minutes.
Invalidate the cache after logout or role change. Caching reduces load but does
not replace choosing the correct NebulaAPI fetch method for concurrent UIs.
