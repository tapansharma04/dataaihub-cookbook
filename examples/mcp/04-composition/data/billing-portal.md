# Billing Portal

Service documentation for the Billing Portal.

The Billing Portal exposes invoices, usage meters, and plan changes for
Acme AI customers. This resource is static fixture content owned by the
MCP server.

## Related services

- billing-api
- identity-api
- knowledge-platform

## Surfaces

- Invoice list and PDF retrieval
- Usage meters by product
- Plan upgrade and downgrade requests

## Operational notes

Authentication is handled by identity-api. Live availability is published
on `acme://status/services`. Invoice PDF generation is the component most
often cited when billing-api is degraded.
