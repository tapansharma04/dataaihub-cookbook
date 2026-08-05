# Acme Edge Platform — Support Handbook

Tiny on purpose. Each ## section is one retrieval chunk. Topics share some
vocabulary (TLS, profiles, NebulaAPI, timeouts) while other sections deliberately
use handbook jargon that end-user questions rarely contain — useful when
comparing original-only retrieval against multi-query transformation.

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

NebulaAPI v2.3 introduces `getUserProfileAsync` as the preferred call when many
UI screens issue profile reads at once. Migration for NebulaAPI v2.3: replace
synchronous reads with getUserProfileAsync and handle the Promise. Under high
concurrency this avoids freezing the main application thread and is the supported
path for loading user profiles without blocking the UI.

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

## Relay idle session keepalive termination

Regional relays tear down idle EdgeGateway tunnels when
`relay_keepalive_miss_threshold` consecutive keepalive probes fail (default 3).
Operators typically observe TCP `RST` or `ECONNRESET` against
relay-west.acme.internal after long idle periods with no application traffic.
Remediation: set `relay_keepalive_interval_ms` to 15000 or lower, confirm UDP
4500 is not filtered, and review `edgectl relay probe --keepalive`. This
playbook addresses idle keepalive termination only — not E_CONN_42 certificate
failures and not customer LAN packet loss.

## NebulaAuth token TTL and silent credential refresh

NebulaAuth access tokens expire after eight hours of inactivity. When the TTL
elapses, Edge clients receive HTTP 401 with reason code `AUTH_TOKEN_EXPIRED`
and the Edge UI returns the operator to the credential prompt. Enable silent
credential refresh with `nebula auth refresh --background` so tokens renew
without interactive password prompts. Do not treat `AUTH_TOKEN_EXPIRED` as an
SSO outage, password-reset incident, workstation screen-lock event, or
NebulaAPI profile-fetch failure.

## Firmware baseline drift on NX series appliances

NX-series appliances that fall more than two minor versions behind the fleet
baseline emit `FW_DRIFT_WARN` during inventory sync. Schedule
`edgectl firmware align --sku NX-4400-PRO` during the next maintenance window.
Firmware drift does not cause relay keepalive resets or AUTH_TOKEN_EXPIRED
responses.

## Edge log shipping to Central Observability

EdgeGateways ship JSON logs to Central Observability over mTLS on port 6514.
Backpressure raises `LOG_SHIP_LAG_SECONDS`. Increase
`log_ship_batch_size` only after confirming collector capacity. Log shipping
lag is unrelated to NebulaAPI profile latency.

## Rate limits for NebulaAPI public endpoints

Public NebulaAPI endpoints enforce 120 requests per minute per client key.
Responses include `Retry-After` when limit `RATE_LIMIT_EXCEEDED` is hit. Burst
UI traffic should use client-side coalescing; raising the limit requires a
platform ticket. Rate limits are separate from getUserProfileAsync concurrency
guidance.

## DNS resolver settings for edge sites

Edge sites must pin recursive resolvers to the Acme anycast pair
`dns-a.acme.internal` / `dns-b.acme.internal`. Incorrect resolvers produce
intermittent NXDOMAIN for relay hostnames. Validate with
`edgectl dns check --relay relay-west.acme.internal`. DNS misconfiguration is
not a keepalive or certificate issue.

## Disk pressure alerts on EdgeGateway hosts

When root filesystem usage exceeds 85%, EdgeGateway emits `DISK_PRESSURE_HIGH`
and may pause non-critical collectors. Free space under `/var/log/edgectl` and
confirm log rotation. Disk pressure does not terminate relay tunnels by itself.

## Metrics retention in Central Observability

Central Observability retains high-resolution Edge metrics for 14 days and
downsamples older series. Changing retention does not alter EdgeGateway
keepalive behavior or NebulaAuth token TTL.

## Password complexity requirements

Acme directory passwords must be at least 14 characters with mixed case and a
symbol. Complexity rules apply at password-reset time only. They do not explain
HTTP 401 responses from Edge clients and do not enable silent credential
refresh.

## SSO identity-provider outage runbook

When the corporate IdP is unreachable, interactive SSO sign-in fails with a
browser error page. Follow the IdP status page and fall back to break-glass
accounts for operators. An IdP outage is distinct from `AUTH_TOKEN_EXPIRED`
returned by an already-authenticated Edge client.

## VPN reconnect after laptop sleep

Managed laptops often drop the corporate VPN after sleep or lid-close. Users
must reconnect GlobalProtect before internal tools load. VPN sleep reconnect is
unrelated to relay keepalive termination and unrelated to NebulaAuth token
refresh.

## Browser cookie retention for internal portals

Some internal portals rely on browser session cookies that expire after a fixed
wall-clock duration. Cookie retention settings are managed by the portal team.
They do not configure NebulaAuth access-token TTL on Edge clients.

## Warehouse barcode scanner firmware notes

Handheld barcode scanners in receiving docks run firmware 9.4.2. Docking
cradles must supply at least 5V/2A. Scanner firmware is unrelated to Edge
client authentication and unrelated to NebulaAPI profile fetches.
