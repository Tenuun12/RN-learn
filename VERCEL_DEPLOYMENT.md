# Vercel Services deployment

This repository is configured as one Vercel project with two services:

- `frontend`: the public Vite application at `/`
- `backend`: the public FastAPI application at `/api/*`

The browser calls same-origin `/api/*` URLs. No service binding is declared because
there is no server-side service-to-service call. Vercel service bindings exist only
inside runtime functions; they are not available to browser code or during a Vite build.

## Import from GitHub

1. Keep **Root Directory** set to `./`.
2. Keep **Application Preset** set to **Services**.
3. Do not import `backend` or `frontend` as a standalone project.
4. Deploy the grouped project after confirming the service names and public paths in
   `vercel.json`.

No `VITE_API_URL` environment variable is needed on Vercel. The production frontend
uses the shared project domain and the `/api/*` rewrite.

## Local verification

With Vercel CLI 48.1.8 or newer installed, run:

```powershell
vercel dev -L
```

## Persistence limitation

Vercel Functions have a read-only deployment filesystem. The app therefore uses
`/tmp` for test-key edits and generated result files when `VERCEL` is present. This
makes deployment and a single warm-instance workflow function, but `/tmp` is not a
durable database: teacher-created tests and edits can disappear after a cold start or
when traffic reaches another instance.

Before using this for permanent production records, connect a durable database and
replace the JSON `TestRepository` implementation. Do not set `OMR_DATA_PATH` to a path
inside the deployment bundle. `BLOB_READ_WRITE_TOKEN` is intentionally not required by
this configuration because a Vercel Blob store has not yet been selected or created.
