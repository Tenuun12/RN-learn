# Vercel Services deployment

This repository is configured as one Vercel project with two services:

- `frontend`: the public Vite application at `/`
- `backend`: the public FastAPI application at `/api/*`

The browser calls same-origin `/api/*` URLs. No service binding is declared because
there is no server-side service-to-service call. Vercel service bindings exist only
inside runtime functions; they are not available to browser code or during a Vite build.

The REST API is discoverable after deployment at:

- `/api` - JSON endpoint index
- `/api/docs` - interactive Swagger UI
- `/api/redoc` - ReDoc reference
- `/api/openapi.json` - OpenAPI schema

## Import from GitHub

1. Keep **Root Directory** set to `./`.
2. Keep **Application Preset** set to **Services**.
3. Do not import `backend` or `frontend` as a standalone project.
4. Deploy the grouped project after confirming the service names and public paths in
   `vercel.json`.

No `VITE_API_URL` environment variable is needed on Vercel. The production frontend
uses the shared project domain and the `/api/*` rewrite.

## QR decoding

The grading endpoint scans the original upload for QR codes as well as grading the
OMR sheet. It uses OpenCV already included in the backend service, so Vercel does not
need an additional binary or system package. The first decoded code is returned as
`qr_data`/`qr_url`; `qr_codes` contains every unique code OpenCV decoded. Only HTTP
and HTTPS payloads become clickable links. An unreadable or damaged QR code is
reported as not detected and never prevents OMR grading.
The response also includes a compact `qr_image_data_url` preview from the original
upload so the code remains visible beside the calibrated OMR verification image.

## Supported OMR templates

The test-creation screen exposes two independently calibrated formats:

- `legacy_red_60_30_v1`: the original red sheet with 60 Part 1 and 30 Part 2 rows.
- `school21_70_32_v1`: the School 21 sheet with 70 Part 1 and 32 Part 2 rows.

Each saved answer key records its `layout_id`. Grading uses that selected layout and
rejects an image whose registration lattice or circle grid belongs to another format.
For the School 21 template, the handwritten code and variant fields are intentionally
ignored; only the Part 1 and Part 2 answer bubbles are graded.

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

The frontend also keeps completed answer keys in the current browser's local storage.
The **Answer keys** screen can restore and select those keys after a cold start, and
the selected key is submitted with each grading request so grading does not depend on
temporary server state. Browser storage is device-specific; it is not a replacement
for a shared database when several teachers or devices need the same key library.

Before using this for permanent production records, connect a durable database and
replace the JSON `TestRepository` implementation. Do not set `OMR_DATA_PATH` to a path
inside the deployment bundle. `BLOB_READ_WRITE_TOKEN` is intentionally not required by
this configuration because a Vercel Blob store has not yet been selected or created.
